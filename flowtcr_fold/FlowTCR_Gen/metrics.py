"""
Evaluation metrics for FlowTCR-Gen.

Metrics:
- Recovery Rate: Exact match with ground truth CDR3β
- Diversity: Unique sequences among generated samples
- Perplexity: Approximate log-likelihood based metric
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_recovery_rate(
    generated: List[str],
    ground_truth: List[str],
) -> Dict[str, float]:
    """
    Compute recovery rate metrics.
    
    Args:
        generated: List of generated CDR3β sequences
        ground_truth: List of ground truth CDR3β sequences
    
    Returns:
        Dict with 'exact_match', 'partial_match_80', 'partial_match_90'
    """
    assert len(generated) == len(ground_truth), "Lists must have same length"
    
    exact_matches = 0
    partial_80 = 0
    partial_90 = 0
    
    for gen, gt in zip(generated, ground_truth):
        if gen == gt:
            exact_matches += 1
            partial_80 += 1
            partial_90 += 1
        else:
            # Compute sequence identity
            identity = compute_sequence_identity(gen, gt)
            if identity >= 0.9:
                partial_90 += 1
                partial_80 += 1
            elif identity >= 0.8:
                partial_80 += 1
    
    n = len(generated)
    return {
        'exact_match': exact_matches / n if n > 0 else 0.0,
        'partial_match_80': partial_80 / n if n > 0 else 0.0,
        'partial_match_90': partial_90 / n if n > 0 else 0.0,
    }


def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity (alignment-free, by position)."""
    if not seq1 or not seq2:
        return 0.0
    
    # Simple position-wise comparison (no gaps)
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    if max_len == 0:
        return 0.0
    
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    return matches / max_len


def compute_diversity(
    sequences: List[str],
    n_total: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute diversity metrics.
    
    Args:
        sequences: List of generated CDR3β sequences
        n_total: Total number of attempts (for unique ratio)
    
    Returns:
        Dict with 'unique_count', 'unique_ratio', 'entropy'
    """
    if not sequences:
        return {'unique_count': 0, 'unique_ratio': 0.0, 'entropy': 0.0}
    
    unique_seqs = set(sequences)
    unique_count = len(unique_seqs)
    
    n_total = n_total or len(sequences)
    unique_ratio = unique_count / n_total if n_total > 0 else 0.0
    
    # Compute entropy of sequence distribution
    import math
    seq_counts = Counter(sequences)
    total = sum(seq_counts.values())
    entropy = 0.0
    for count in seq_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p + 1e-10)
    
    return {
        'unique_count': unique_count,
        'unique_ratio': unique_ratio,
        'entropy': entropy,
    }


def compute_length_distribution(
    sequences: List[str],
    reference: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compare length distribution of generated vs reference sequences.
    
    Returns:
        Dict with 'mean_len', 'std_len', 'kl_divergence' (if reference provided)
    """
    lengths = [len(s) for s in sequences]
    
    if not lengths:
        return {'mean_len': 0.0, 'std_len': 0.0}
    
    mean_len = sum(lengths) / len(lengths)
    std_len = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5
    
    result = {
        'mean_len': mean_len,
        'std_len': std_len,
    }
    
    if reference:
        ref_lengths = [len(s) for s in reference]
        ref_mean = sum(ref_lengths) / len(ref_lengths)
        ref_std = (sum((l - ref_mean) ** 2 for l in ref_lengths) / len(ref_lengths)) ** 0.5
        
        result['ref_mean_len'] = ref_mean
        result['ref_std_len'] = ref_std
        result['len_diff'] = abs(mean_len - ref_mean)
    
    return result


def compute_amino_acid_distribution(
    sequences: List[str],
    reference: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compare amino acid frequency distribution.
    
    Returns:
        Dict with 'kl_divergence' (if reference provided)
    """
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    def count_aa(seqs: List[str]) -> Dict[str, int]:
        counts = Counter()
        for seq in seqs:
            counts.update(seq)
        return counts
    
    gen_counts = count_aa(sequences)
    total_gen = sum(gen_counts.values())
    
    if total_gen == 0:
        return {'kl_divergence': 0.0}
    
    if reference:
        ref_counts = count_aa(reference)
        total_ref = sum(ref_counts.values())
        
        if total_ref == 0:
            return {'kl_divergence': 0.0}
        
        # Compute KL divergence
        import math
        kl = 0.0
        for aa in AMINO_ACIDS:
            p_gen = (gen_counts.get(aa, 0) + 1) / (total_gen + len(AMINO_ACIDS))
            p_ref = (ref_counts.get(aa, 0) + 1) / (total_ref + len(AMINO_ACIDS))
            kl += p_gen * math.log(p_gen / p_ref + 1e-10)
        
        return {'aa_kl_divergence': kl}
    
    return {}


def tokens_to_sequence(tokens: torch.Tensor, vocab: List[str]) -> str:
    """Convert token indices to sequence string."""
    return ''.join(vocab[t] for t in tokens.tolist() if 0 <= t < len(vocab))


class FlowTCRGenEvaluator:
    """
    Evaluator for FlowTCR-Gen model.
    
    Computes recovery, diversity, and perplexity metrics.
    """

    def __init__(
        self,
        vocab: List[str],
        pad_idx: int = 0,
    ):
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.generated_seqs = []
        self.ground_truth_seqs = []
        self.flow_costs = []

    def add_batch(
        self,
        generated_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        flow_cost: Optional[torch.Tensor] = None,
    ):
        """
        Add a batch of generated sequences.
        
        Args:
            generated_tokens: [B, L] generated token indices
            target_tokens: [B, L] target token indices
            pad_mask: [B, L] 1 for valid, 0 for pad
            flow_cost: [B] flow matching cost per sample
        """
        B = generated_tokens.shape[0]
        
        for i in range(B):
            gen = generated_tokens[i]
            tgt = target_tokens[i]
            
            if pad_mask is not None:
                mask = pad_mask[i].bool()
                gen = gen[mask]
                tgt = tgt[mask]
            
            gen_seq = tokens_to_sequence(gen, self.vocab)
            tgt_seq = tokens_to_sequence(tgt, self.vocab)
            
            self.generated_seqs.append(gen_seq)
            self.ground_truth_seqs.append(tgt_seq)
        
        if flow_cost is not None:
            self.flow_costs.extend(flow_cost.tolist())

    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}
        
        # Recovery
        recovery = compute_recovery_rate(self.generated_seqs, self.ground_truth_seqs)
        results.update({f'recovery/{k}': v for k, v in recovery.items()})
        
        # Diversity
        diversity = compute_diversity(self.generated_seqs)
        results.update({f'diversity/{k}': v for k, v in diversity.items()})
        
        # Length distribution
        length = compute_length_distribution(self.generated_seqs, self.ground_truth_seqs)
        results.update({f'length/{k}': v for k, v in length.items()})
        
        # AA distribution
        aa = compute_amino_acid_distribution(self.generated_seqs, self.ground_truth_seqs)
        results.update({f'aa/{k}': v for k, v in aa.items()})
        
        # Perplexity (from flow cost)
        if self.flow_costs:
            import math
            mean_cost = sum(self.flow_costs) / len(self.flow_costs)
            # Approximate perplexity as exp(cost) with safety clamp
            results['perplexity'] = min(math.exp(min(mean_cost, 10.0)), 1e10)
        
        return results


def evaluate_recovery(
    model,
    encoder,
    flow_matcher,
    val_loader,
    vocab: List[str],
    n_samples: int = 1,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate recovery rate on validation set.
    
    Args:
        model: Complete FlowTCR-Gen model
        val_loader: Validation data loader
        vocab: Token vocabulary
        n_samples: Number of samples per input
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluator = FlowTCRGenEvaluator(vocab=vocab)
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            for _ in range(n_samples):
                # Generate sequences
                # This would need to be implemented based on actual model interface
                pass
    
    return evaluator.compute_metrics()


def evaluate_diversity(
    model,
    encoder,
    flow_matcher,
    val_loader,
    vocab: List[str],
    n_samples: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate diversity by generating multiple samples per input.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_generated = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            for _ in range(n_samples):
                # Generate and collect
                pass
    
    return compute_diversity(all_generated, n_total=len(val_loader) * n_samples)


if __name__ == "__main__":
    # Quick test
    vocab = list('ACDEFGHIKLMNPQRSTVWY') + ['<PAD>']
    
    # Test recovery
    gen = ['CASSLGQFF', 'CASSIRSTDTQYF', 'CASSLGQFF']
    gt = ['CASSLGQFF', 'CASSIRSTDTQYF', 'CASSLGQAF']
    
    recovery = compute_recovery_rate(gen, gt)
    print(f"✅ Recovery metrics: {recovery}")
    
    # Test diversity
    diversity = compute_diversity(gen)
    print(f"✅ Diversity metrics: {diversity}")
    
    # Test evaluator
    evaluator = FlowTCRGenEvaluator(vocab=vocab)
    
    gen_tokens = torch.tensor([
        [2, 0, 18, 18, 11, 6, 16, 5, 5],
        [2, 0, 18, 18, 8, 17, 18, 19, 3],
    ])
    tgt_tokens = torch.tensor([
        [2, 0, 18, 18, 11, 6, 16, 5, 5],
        [2, 0, 18, 18, 8, 17, 18, 19, 4],
    ])
    
    evaluator.add_batch(gen_tokens, tgt_tokens)
    metrics = evaluator.compute_metrics()
    print(f"✅ Evaluator metrics: {metrics}")
