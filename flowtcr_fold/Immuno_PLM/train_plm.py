"""
Training entrypoint for the Immuno-PLM hybrid encoder.

Implements a simple InfoNCE-style contrastive objective using anchor vs decoy negatives.
Replace with full MLM + hard-negative loss when ready.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from flowtcr_fold.data.dataset import FlowDataset
from flowtcr_fold.data.tokenizer import vocab_size, BasicTokenizer
from flowtcr_fold.Immuno_PLM.immuno_plm import ImmunoPLM
from flowtcr_fold.common.utils import save_checkpoint, EarlyStopper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/trn.csv")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--use_esm", action="store_true")
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--mlm_weight", type=float, default=1.0)
    p.add_argument("--out_dir", type=str, default="checkpoints/plm")
    return p.parse_args()


def collate(batch):
    tokens_pos = [item["tokens_pos"] for item in batch]
    masks_pos = [item["mask_pos"] for item in batch]
    tokens_neg = [item["tokens_neg"] for item in batch if item["tokens_neg"] is not None]
    masks_neg = [item["mask_neg"] for item in batch if item["mask_neg"] is not None]
    slices_pos = [item["slices_pos"] for item in batch]
    slices_neg = [item["slices_neg"] for item in batch if item["slices_neg"] is not None]
    neg_types = [item["meta"]["neg_type"] for item in batch]

    def pad(list_tokens):
        if not list_tokens:
            return None, None
        max_len = max(t.size(0) for t in list_tokens)
        padded = torch.zeros(len(list_tokens), max_len, dtype=torch.long)
        mask = torch.zeros(len(list_tokens), max_len, dtype=torch.long)
        for i, t in enumerate(list_tokens):
            padded[i, : t.size(0)] = t
            mask[i, : t.size(0)] = 1
        return padded, mask

    pos_padded, pos_mask = pad(tokens_pos)
    neg_padded, neg_mask = pad(tokens_neg)

    topo_pos = slices_pos if len(slices_pos) > 0 else None
    topo_neg = slices_neg if len(slices_neg) > 0 else None

    return {
        "tokens_pos": pos_padded,
        "mask_pos": pos_mask,
        "tokens_neg": neg_padded,
        "mask_neg": neg_mask,
        "slices_pos": topo_pos,
        "slices_neg": topo_neg,
        "neg_types": neg_types,
    }


def info_nce(anchor, positive, negatives, tau: float = 0.1):
    """
    anchor: [B, D]
    positive: [B, D]
    negatives: [B_neg, D] or None
    """
    sims_pos = torch.cosine_similarity(anchor, positive, dim=-1) / tau
    denom = torch.exp(sims_pos)
    if negatives is not None and negatives.size(0) > 0:
        neg_expand = negatives.unsqueeze(0)  # [1, Bneg, D]
        anchor_expand = anchor.unsqueeze(1)  # [B,1,D]
        sims_neg = torch.cosine_similarity(anchor_expand, neg_expand, dim=-1) / tau  # [B, Bneg]
        denom = denom + torch.exp(sims_neg).sum(dim=-1)
    loss = -torch.log(torch.exp(sims_pos) / denom).mean()
    return loss


def mask_tokens(tokens: torch.Tensor, tokenizer, mask_prob: float = 0.15):
    """
    BERT-style masking for MLM. Only supports BasicTokenizer fallback.
    """
    if not isinstance(tokenizer, BasicTokenizer):
        return tokens, tokens  # skip MLM if not basic
    labels = tokens.clone()
    mask_token = tokenizer.stoi["[MASK]"]
    special = set([tokenizer.stoi[t] for t in ["[PAD]", "[CLS]", "[SEP]"] if t in tokenizer.stoi])
    probability_matrix = torch.full(labels.shape, mask_prob, device=tokens.device)
    for sp in special:
        probability_matrix[tokens == sp] = 0
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # ignore index
    tokens = tokens.clone()
    tokens[masked_indices] = mask_token
    return tokens, labels


def main():
    args = parse_args()
    ds = FlowDataset(args.data, split="train")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImmunoPLM(use_esm=args.use_esm, vocab_size=vocab_size(ds.tokenizer)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    logger = []
    stopper = EarlyStopper(patience=100)

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            pos_tokens = batch["tokens_pos"].to(device)
            pos_mask = batch["mask_pos"].to(device)
            neg_tokens = batch["tokens_neg"].to(device) if batch["tokens_neg"] is not None else None
            neg_mask = batch["mask_neg"].to(device) if batch["mask_neg"] is not None else None

            # Forward pass 1: Anchor
            out_anchor = model(pos_tokens, pos_mask, region_slices=batch["slices_pos"])
            anchor = out_anchor["pooled"]
            
            # Forward pass 2: Positive (same input, different dropout mask)
            out_pos = model(pos_tokens, pos_mask, region_slices=batch["slices_pos"])
            positive = out_pos["pooled"]

            negatives = None
            if neg_tokens is not None:
                out_neg = model(neg_tokens, neg_mask, region_slices=batch["slices_neg"])
                negatives = out_neg["pooled"]

            loss_nce = info_nce(anchor, positive, negatives, tau=args.tau)

            loss_mlm = torch.tensor(0.0, device=device)
            if not model.use_esm and hasattr(model, "decoder"):
                masked_tokens, labels = mask_tokens(pos_tokens, ds.tokenizer)
                masked_tokens = masked_tokens.to(device)
                labels = labels.to(device)
                out_mlm = model(masked_tokens, pos_mask)
                logits = model.decoder(out_mlm["s"])
                loss_mlm = ce(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss = loss_nce + args.mlm_weight * loss_mlm
            opt.zero_grad()
            loss.backward()
            opt.step()
            logger.append(
                {
                    "loss_nce": loss_nce.item(),
                    "loss_mlm": loss_mlm.item(),
                    "neg_types": batch["neg_types"],
                }
            )
        print(f"epoch {epoch} nce {loss_nce.item():.4f} mlm {loss_mlm.item():.4f}")
        # checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, opt, args.out_dir, epoch + 1, tag="plm")
        # early stopping on total loss
        if stopper.update(loss.item()):
            print(f"Early stopping at epoch {epoch+1} (no improvement for {stopper.patience} epochs)")
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), out_dir / "immuno_plm.pt")


if __name__ == "__main__":
    main()
