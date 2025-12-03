from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import multi_positive_infonce, multilabel_bce


# =============================================================================
# Frequency Baseline & KL Divergence Utilities
# =============================================================================


class FrequencyBaseline:
    """
    Frequency-based baseline for scaffold prediction.
    For each MHC, predicts gene distribution based on training set frequencies.
    """

    def __init__(self, samples: List[Dict[str, Any]], gene_vocab: Dict[str, Dict[str, int]]):
        """
        Args:
            samples: List of sample dicts with 'mhc', 'h_v', 'h_j', 'l_v', 'l_j' keys.
            gene_vocab: Dict mapping gene type -> {gene_name: gene_id}.
        """
        self.gene_vocab = gene_vocab
        self.gene_types = ["h_v", "h_j", "l_v", "l_j"]

        # Build MHC -> gene count distributions
        self.mhc_to_gene_counts: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: {g: Counter() for g in self.gene_types}
        )
        # Global distribution (fallback for unseen MHC)
        self.global_counts: Dict[str, Counter] = {g: Counter() for g in self.gene_types}

        for sample in samples:
            mhc = sample.get("mhc", "")
            if not mhc:
                continue
            for gtype in self.gene_types:
                gene_name = sample.get(gtype, "")
                if gene_name and gene_name in gene_vocab.get(gtype, {}):
                    gene_id = gene_vocab[gtype][gene_name]
                    self.mhc_to_gene_counts[mhc][gtype][gene_id] += 1
                    self.global_counts[gtype][gene_id] += 1

        # Convert to probability distributions
        self.mhc_to_gene_dist: Dict[str, Dict[str, torch.Tensor]] = {}
        for mhc, gene_counts in self.mhc_to_gene_counts.items():
            self.mhc_to_gene_dist[mhc] = {}
            for gtype, counts in gene_counts.items():
                self.mhc_to_gene_dist[mhc][gtype] = self._counts_to_dist(counts, len(gene_vocab[gtype]))

        # Global fallback distributions
        self.global_dist: Dict[str, torch.Tensor] = {}
        for gtype, counts in self.global_counts.items():
            self.global_dist[gtype] = self._counts_to_dist(counts, len(gene_vocab[gtype]))

    @staticmethod
    def _counts_to_dist(counts: Counter, vocab_size: int, smoothing: float = 1e-8) -> torch.Tensor:
        """Convert count dict to probability distribution tensor."""
        dist = torch.zeros(vocab_size) + smoothing
        total = sum(counts.values()) + smoothing * vocab_size
        for gene_id, count in counts.items():
            if 0 <= gene_id < vocab_size:
                dist[gene_id] = count + smoothing
        return dist / total

    def get_distribution(self, mhc: str, gene_type: str) -> torch.Tensor:
        """Get p(gene | MHC) for a specific gene type."""
        if mhc in self.mhc_to_gene_dist:
            return self.mhc_to_gene_dist[mhc][gene_type]
        return self.global_dist[gene_type]

    def predict_topk(self, mhc: str, gene_type: str, k: int = 10) -> List[int]:
        """Predict top-k gene IDs for a given MHC."""
        dist = self.get_distribution(mhc, gene_type)
        topk = dist.topk(min(k, len(dist))).indices.tolist()
        return topk


def compute_kl_divergence(
    model_logits: torch.Tensor,
    empirical_dist: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute KL(empirical || model) divergence.

    Args:
        model_logits: [B, num_genes] raw logits from model
        empirical_dist: [B, num_genes] empirical probability distribution
        reduction: 'mean', 'sum', or 'none'

    Returns:
        KL divergence value(s)
    """
    model_log_probs = F.log_softmax(model_logits, dim=-1)
    # KL(P || Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
    kl = F.kl_div(model_log_probs, empirical_dist, reduction="none").sum(dim=-1)
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl


def evaluate_frequency_baseline_recall(
    baseline: FrequencyBaseline,
    samples: List[Dict[str, Any]],
    gene_vocab: Dict[str, Dict[str, int]],
    k_list: List[int] = [1, 5, 10, 20],
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate frequency baseline R@K for each gene type.

    Args:
        baseline: FrequencyBaseline instance
        samples: Validation samples
        gene_vocab: Gene vocabulary
        k_list: List of K values for R@K

    Returns:
        Dict[gene_type, Dict[k, recall]]
    """
    results: Dict[str, Dict[int, float]] = {g: {k: 0.0 for k in k_list} for g in baseline.gene_types}

    for gtype in baseline.gene_types:
        hits = {k: [] for k in k_list}
        for sample in samples:
            mhc = sample.get("mhc", "")
            gene_name = sample.get(gtype, "")
            if not mhc or not gene_name:
                continue
            true_id = gene_vocab[gtype].get(gene_name, -1)
            if true_id < 0:
                continue

            for k in k_list:
                topk_ids = baseline.predict_topk(mhc, gtype, k)
                hit = 1.0 if true_id in topk_ids else 0.0
                hits[k].append(hit)

        for k in k_list:
            if hits[k]:
                results[gtype][k] = sum(hits[k]) / len(hits[k])

    return results


def _mask_pos(pos_mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Apply validity mask to a [B,B] pos_mask."""
    v = valid.unsqueeze(0) & valid.unsqueeze(1)
    return pos_mask & v


def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    pos_weight: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Train one epoch with:
    - MHC-group InfoNCE: only on samples with MHC
    - pMHC-group InfoNCE: only on samples with MHC  
    - Multi-label BCE: only on samples with MHC
    - Peptide-only InfoNCE: on ALL samples (weak weight)
    """
    model.train()
    total_loss = total_nce_mhc = total_nce_pmhc = total_nce_pep = total_bce = 0.0
    num_batches = 0
    num_mhc_batches = 0  # batches with at least one MHC sample

    for batch in loader:
        pmhc_tokens = batch["pmhc_tokens"].to(device)
        pmhc_mask = batch["pmhc_mask"].to(device)
        hv_tokens = batch["hv_tokens"].to(device)
        hv_mask = batch["hv_mask"].to(device)
        hj_tokens = batch["hj_tokens"].to(device)
        hj_mask = batch["hj_mask"].to(device)
        lv_tokens = batch["lv_tokens"].to(device)
        lv_mask = batch["lv_mask"].to(device)
        lj_tokens = batch["lj_tokens"].to(device)
        lj_mask = batch["lj_mask"].to(device)
        allele_id = batch["allele_id"].to(device)

        pos_mask_mhc = batch["pos_mask_mhc"].to(device)
        pos_mask_pmhc = batch["pos_mask_pmhc"].to(device)
        pos_mask_pep = batch["pos_mask_pep"].to(device)
        has_mhc = batch["has_mhc"].to(device)

        target_hv = batch["target_hv_mhc"].to(device)
        target_hj = batch["target_hj_mhc"].to(device)
        target_lv = batch["target_lv_mhc"].to(device)
        target_lj = batch["target_lj_mhc"].to(device)

        valid_hv = batch["valid_hv_seq"].to(device)
        valid_hj = batch["valid_hj_seq"].to(device)
        valid_lv = batch["valid_lv_seq"].to(device)
        valid_lj = batch["valid_lj_seq"].to(device)

        out = model(
            pmhc_tokens,
            pmhc_mask,
            allele_id,
            hv_tokens,
            hv_mask,
            hj_tokens,
            hj_mask,
            lv_tokens,
            lv_mask,
            lj_tokens,
            lj_mask,
        )

        # =====================================================================
        # Loss 1: MHC-group InfoNCE (only on samples with MHC)
        # =====================================================================
        loss_nce_mhc = torch.tensor(0.0, device=device)
        loss_nce_pmhc = torch.tensor(0.0, device=device)
        loss_bce = torch.tensor(0.0, device=device)
        
        if has_mhc.any():
            num_mhc_batches += 1
            # Combine has_mhc with valid_* masks
            valid_hv_mhc = valid_hv & has_mhc
            valid_hj_mhc = valid_hj & has_mhc
            valid_lv_mhc = valid_lv & has_mhc
            valid_lj_mhc = valid_lj & has_mhc

            # MHC-group InfoNCE
        loss_nce_mhc = (
                multi_positive_infonce(out["z_pmhc"], out["z_hv"], _mask_pos(pos_mask_mhc, valid_hv_mhc), args.tau)
                + multi_positive_infonce(out["z_pmhc"], out["z_hj"], _mask_pos(pos_mask_mhc, valid_hj_mhc), args.tau)
                + multi_positive_infonce(out["z_pmhc"], out["z_lv"], _mask_pos(pos_mask_mhc, valid_lv_mhc), args.tau)
                + multi_positive_infonce(out["z_pmhc"], out["z_lj"], _mask_pos(pos_mask_mhc, valid_lj_mhc), args.tau)
        )
            
            # pMHC-group InfoNCE
        loss_nce_pmhc = (
                multi_positive_infonce(out["z_pmhc"], out["z_hv"], _mask_pos(pos_mask_pmhc, valid_hv_mhc), args.tau)
                + multi_positive_infonce(out["z_pmhc"], out["z_hj"], _mask_pos(pos_mask_pmhc, valid_hj_mhc), args.tau)
                + multi_positive_infonce(out["z_pmhc"], out["z_lv"], _mask_pos(pos_mask_pmhc, valid_lv_mhc), args.tau)
                + multi_positive_infonce(out["z_pmhc"], out["z_lj"], _mask_pos(pos_mask_pmhc, valid_lj_mhc), args.tau)
        )

            # Multi-label BCE (only for samples with MHC)
            # has_mhc acts as additional validity mask
            valid_hv_target = (target_hv.sum(dim=1) > 0).float() * has_mhc.float()
            valid_hj_target = (target_hj.sum(dim=1) > 0).float() * has_mhc.float()
            valid_lv_target = (target_lv.sum(dim=1) > 0).float() * has_mhc.float()
            valid_lj_target = (target_lj.sum(dim=1) > 0).float() * has_mhc.float()

        loss_bce = (
            multilabel_bce(out["hv_logits"], target_hv, pos_weight["h_v"], valid_hv_target)
            + multilabel_bce(out["hj_logits"], target_hj, pos_weight["h_j"], valid_hj_target)
            + multilabel_bce(out["lv_logits"], target_lv, pos_weight["l_v"], valid_lv_target)
            + multilabel_bce(out["lj_logits"], target_lj, pos_weight["l_j"], valid_lj_target)
        )

        # =====================================================================
        # Loss 2: Peptide-only InfoNCE (on ALL samples, weak weight)
        # =====================================================================
        loss_nce_pep = (
            multi_positive_infonce(out["z_pmhc"], out["z_hv"], _mask_pos(pos_mask_pep, valid_hv), args.tau)
            + multi_positive_infonce(out["z_pmhc"], out["z_hj"], _mask_pos(pos_mask_pep, valid_hj), args.tau)
            + multi_positive_infonce(out["z_pmhc"], out["z_lv"], _mask_pos(pos_mask_pep, valid_lv), args.tau)
            + multi_positive_infonce(out["z_pmhc"], out["z_lj"], _mask_pos(pos_mask_pep, valid_lj), args.tau)
        )

        # =====================================================================
        # Combined loss
        # =====================================================================
        lambda_pep = getattr(args, 'lambda_pep', 0.1)  # default 0.1
        loss = (
            loss_nce_mhc 
            + args.lambda_pmhc * loss_nce_pmhc 
            + args.lambda_bce * loss_bce
            + lambda_pep * loss_nce_pep
        )

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_nce_mhc += loss_nce_mhc.item()
        total_nce_pmhc += loss_nce_pmhc.item()
        total_nce_pep += loss_nce_pep.item()
        total_bce += loss_bce.item()
        num_batches += 1

    if num_batches == 0:
        num_batches = 1
    if num_mhc_batches == 0:
        num_mhc_batches = 1
    return {
        "loss": total_loss / num_batches,
        "nce_mhc": total_nce_mhc / num_mhc_batches,
        "nce_pmhc": total_nce_pmhc / num_mhc_batches,
        "nce_pep": total_nce_pep / num_batches,
        "bce": total_bce / num_mhc_batches,
    }


def _recall_hits_at_k(
    z_query: torch.Tensor,
    z_gene: torch.Tensor,
    pos_mask: torch.Tensor,
    k_list: List[int],
) -> Dict[int, Dict[str, float]]:
    """
    Compute hit counts and denominators for Recall@K across a batch.

    Returns:
        Dict where each k maps to {"hits": float, "count": int}.
    """
    pos_mask = pos_mask.to(z_query.device)
    if pos_mask.numel() == 0:
        return {k: {"hits": 0.0, "count": 0} for k in k_list}

    valid_rows = pos_mask.sum(dim=1) > 0
    if not valid_rows.any():
        return {k: {"hits": 0.0, "count": 0} for k in k_list}

    zq = F.normalize(z_query, p=2, dim=-1)
    zg = F.normalize(z_gene, p=2, dim=-1)
    sim = torch.matmul(zq, zg.T)
    max_k = min(max(k_list), sim.size(1))
    topk_all = sim.topk(max_k, dim=-1).indices

    out: Dict[int, Dict[str, float]] = {k: {"hits": 0.0, "count": 0} for k in k_list}
    for i in range(sim.size(0)):
        if not valid_rows[i]:
            continue
        positives = pos_mask[i].nonzero(as_tuple=False).squeeze(-1)
        for k in k_list:
            cutoff = min(k, topk_all.size(1))
            topk = topk_all[i, :cutoff]
        hit = torch.isin(topk, positives).any().float()
            out[k]["hits"] += hit.item()
            out[k]["count"] += 1
    return out


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    args,
    pos_weight: Dict[str, torch.Tensor],
    frequency_baseline: Optional[FrequencyBaseline] = None,
    k_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Evaluate model with loss, R@K, and optional KL divergence metrics.
    Only evaluates on samples with MHC for MHC-related metrics.
    """
    if k_list is None:
        k_list = [10]

    model.eval()
    total_loss = total_nce_mhc = total_nce_pmhc = total_nce_pep = total_bce = 0.0
    num_batches = 0
    num_mhc_batches = 0

    recall_hits = {
        "hv": {k: {"hits": 0.0, "count": 0} for k in k_list},
        "hj": {k: {"hits": 0.0, "count": 0} for k in k_list},
        "lv": {k: {"hits": 0.0, "count": 0} for k in k_list},
        "lj": {k: {"hits": 0.0, "count": 0} for k in k_list},
    }

    all_logits = {"hv": [], "hj": [], "lv": [], "lj": []}
    all_mhc_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            pmhc_tokens = batch["pmhc_tokens"].to(device)
            pmhc_mask = batch["pmhc_mask"].to(device)
            hv_tokens = batch["hv_tokens"].to(device)
            hv_mask = batch["hv_mask"].to(device)
            hj_tokens = batch["hj_tokens"].to(device)
            hj_mask = batch["hj_mask"].to(device)
            lv_tokens = batch["lv_tokens"].to(device)
            lv_mask = batch["lv_mask"].to(device)
            lj_tokens = batch["lj_tokens"].to(device)
            lj_mask = batch["lj_mask"].to(device)
            allele_id = batch["allele_id"].to(device)

            pos_mask_mhc = batch["pos_mask_mhc"].to(device)
            pos_mask_pmhc = batch["pos_mask_pmhc"].to(device)
            pos_mask_pep = batch["pos_mask_pep"].to(device)
            has_mhc = batch["has_mhc"].to(device)

            target_hv = batch["target_hv_mhc"].to(device)
            target_hj = batch["target_hj_mhc"].to(device)
            target_lv = batch["target_lv_mhc"].to(device)
            target_lj = batch["target_lj_mhc"].to(device)

            valid_hv = batch["valid_hv_seq"].to(device)
            valid_hj = batch["valid_hj_seq"].to(device)
            valid_lv = batch["valid_lv_seq"].to(device)
            valid_lj = batch["valid_lj_seq"].to(device)

            out = model(
                pmhc_tokens,
                pmhc_mask,
                allele_id,
                hv_tokens,
                hv_mask,
                hj_tokens,
                hj_mask,
                lv_tokens,
                lv_mask,
                lj_tokens,
                lj_mask,
            )

            # MHC-related losses (only on samples with MHC)
            loss_nce_mhc = torch.tensor(0.0, device=device)
            loss_nce_pmhc = torch.tensor(0.0, device=device)
            loss_bce = torch.tensor(0.0, device=device)
            
            if has_mhc.any():
                num_mhc_batches += 1
                valid_hv_mhc = valid_hv & has_mhc
                valid_hj_mhc = valid_hj & has_mhc
                valid_lv_mhc = valid_lv & has_mhc
                valid_lj_mhc = valid_lj & has_mhc

            loss_nce_mhc = (
                    multi_positive_infonce(out["z_pmhc"], out["z_hv"], _mask_pos(pos_mask_mhc, valid_hv_mhc), args.tau)
                    + multi_positive_infonce(out["z_pmhc"], out["z_hj"], _mask_pos(pos_mask_mhc, valid_hj_mhc), args.tau)
                    + multi_positive_infonce(out["z_pmhc"], out["z_lv"], _mask_pos(pos_mask_mhc, valid_lv_mhc), args.tau)
                    + multi_positive_infonce(out["z_pmhc"], out["z_lj"], _mask_pos(pos_mask_mhc, valid_lj_mhc), args.tau)
            )
            loss_nce_pmhc = (
                    multi_positive_infonce(out["z_pmhc"], out["z_hv"], _mask_pos(pos_mask_pmhc, valid_hv_mhc), args.tau)
                    + multi_positive_infonce(out["z_pmhc"], out["z_hj"], _mask_pos(pos_mask_pmhc, valid_hj_mhc), args.tau)
                    + multi_positive_infonce(out["z_pmhc"], out["z_lv"], _mask_pos(pos_mask_pmhc, valid_lv_mhc), args.tau)
                    + multi_positive_infonce(out["z_pmhc"], out["z_lj"], _mask_pos(pos_mask_pmhc, valid_lj_mhc), args.tau)
            )

                valid_hv_target = (target_hv.sum(dim=1) > 0).float() * has_mhc.float()
                valid_hj_target = (target_hj.sum(dim=1) > 0).float() * has_mhc.float()
                valid_lv_target = (target_lv.sum(dim=1) > 0).float() * has_mhc.float()
                valid_lj_target = (target_lj.sum(dim=1) > 0).float() * has_mhc.float()

            loss_bce = (
                multilabel_bce(out["hv_logits"], target_hv, pos_weight["h_v"], valid_hv_target)
                + multilabel_bce(out["hj_logits"], target_hj, pos_weight["h_j"], valid_hj_target)
                + multilabel_bce(out["lv_logits"], target_lv, pos_weight["l_v"], valid_lv_target)
                + multilabel_bce(out["lj_logits"], target_lj, pos_weight["l_j"], valid_lj_target)
            )

            # Peptide-only InfoNCE (all samples)
            loss_nce_pep = (
                multi_positive_infonce(out["z_pmhc"], out["z_hv"], _mask_pos(pos_mask_pep, valid_hv), args.tau)
                + multi_positive_infonce(out["z_pmhc"], out["z_hj"], _mask_pos(pos_mask_pep, valid_hj), args.tau)
                + multi_positive_infonce(out["z_pmhc"], out["z_lv"], _mask_pos(pos_mask_pep, valid_lv), args.tau)
                + multi_positive_infonce(out["z_pmhc"], out["z_lj"], _mask_pos(pos_mask_pep, valid_lj), args.tau)
            )

            lambda_pep = getattr(args, 'lambda_pep', 0.1)
            loss = (
                loss_nce_mhc 
                + args.lambda_pmhc * loss_nce_pmhc 
                + args.lambda_bce * loss_bce
                + lambda_pep * loss_nce_pep
            )

            total_loss += loss.item()
            total_nce_mhc += loss_nce_mhc.item()
            total_nce_pmhc += loss_nce_pmhc.item()
            total_nce_pep += loss_nce_pep.item()
            total_bce += loss_bce.item()
            num_batches += 1

            # Only collect MHC samples for R@K evaluation
            if has_mhc.any():
                valid_hv_mhc = valid_hv & has_mhc
                valid_hj_mhc = valid_hj & has_mhc
                valid_lv_mhc = valid_lv & has_mhc
                valid_lj_mhc = valid_lj & has_mhc

                hv_hits = _recall_hits_at_k(out["z_pmhc"], out["z_hv"], _mask_pos(pos_mask_mhc, valid_hv_mhc), k_list)
                hj_hits = _recall_hits_at_k(out["z_pmhc"], out["z_hj"], _mask_pos(pos_mask_mhc, valid_hj_mhc), k_list)
                lv_hits = _recall_hits_at_k(out["z_pmhc"], out["z_lv"], _mask_pos(pos_mask_mhc, valid_lv_mhc), k_list)
                lj_hits = _recall_hits_at_k(out["z_pmhc"], out["z_lj"], _mask_pos(pos_mask_mhc, valid_lj_mhc), k_list)

                for k in k_list:
                    recall_hits["hv"][k]["hits"] += hv_hits[k]["hits"]
                    recall_hits["hv"][k]["count"] += hv_hits[k]["count"]
                    recall_hits["hj"][k]["hits"] += hj_hits[k]["hits"]
                    recall_hits["hj"][k]["count"] += hj_hits[k]["count"]
                    recall_hits["lv"][k]["hits"] += lv_hits[k]["hits"]
                    recall_hits["lv"][k]["count"] += lv_hits[k]["count"]
                    recall_hits["lj"][k]["hits"] += lj_hits[k]["hits"]
                    recall_hits["lj"][k]["count"] += lj_hits[k]["count"]

            # Collect logits for KL computation
            all_logits["hv"].append(out["hv_logits"].cpu())
            all_logits["hj"].append(out["hj_logits"].cpu())

            if "mhc_list" in batch:
                all_mhc_ids.extend(batch["mhc_list"])

    if num_batches == 0:
        num_batches = 1
    if num_mhc_batches == 0:
        num_mhc_batches = 1

    # Concatenate for recall (only samples with MHC contribute to pos_mask)
    metrics: Dict[str, float] = {
        "loss": total_loss / num_batches,
        "nce_mhc": total_nce_mhc / num_mhc_batches,
        "nce_pmhc": total_nce_pmhc / num_mhc_batches,
        "nce_pep": total_nce_pep / num_batches,
        "bce": total_bce / num_mhc_batches,
    }

    for gtype in ["hv", "hj", "lv", "lj"]:
        for k in k_list:
            denom = recall_hits[gtype][k]["count"]
            key = f"recall@{k}_{gtype}"
            metrics[key] = recall_hits[gtype][k]["hits"] / denom if denom > 0 else 0.0

    # Compute KL divergence if baseline is provided (only for samples with real MHC)
    if frequency_baseline is not None and all_mhc_ids:
        hv_logits = torch.cat(all_logits["hv"], dim=0)
        hj_logits = torch.cat(all_logits["hj"], dim=0)

        # Filter to only real MHC (not __NO_MHC_*)
        valid_mhc_mask = [not mhc.startswith("__NO_MHC_") for mhc in all_mhc_ids]
        valid_indices = [i for i, v in enumerate(valid_mhc_mask) if v]
        
        if valid_indices:
            hv_logits_mhc = hv_logits[valid_indices]
            hj_logits_mhc = hj_logits[valid_indices]
            mhc_ids_valid = [all_mhc_ids[i] for i in valid_indices]

            hv_emp = torch.stack([frequency_baseline.get_distribution(mhc, "h_v") for mhc in mhc_ids_valid])
            hj_emp = torch.stack([frequency_baseline.get_distribution(mhc, "h_j") for mhc in mhc_ids_valid])

            metrics["kl_hv"] = compute_kl_divergence(hv_logits_mhc, hv_emp).item()
            metrics["kl_hj"] = compute_kl_divergence(hj_logits_mhc, hj_emp).item()

    return metrics


