"""
Sampling script for FlowTCR-Gen.

Usage:
    python -m flowtcr_fold.FlowTCR_Gen.sample \
        --model checkpoints/flow_gen/flow_model_best.pt \
        --retriever_ckpt checkpoints/scaffold_v1/model_best.pt \
        --data flowtcr_fold/data/tst.jsonl \
        --limit 5
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from flowtcr_fold.FlowTCR_Gen.flow_gen import ConditionEmbedder, FlowMatchingModel
from flowtcr_fold.data.tokenizer import BasicTokenizer, get_tokenizer, vocab_size


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path to trained flow model (state_dict).")
    p.add_argument("--retriever_ckpt", type=str, required=True, help="Path to scaffold retriever checkpoint.")
    p.add_argument("--data", type=str, default="flowtcr_fold/data/tst.jsonl", help="Prompt JSONL/CSV with pMHC + scaffold.")
    p.add_argument("--limit", type=int, default=4, help="Number of records to sample from the data file.")
    p.add_argument("--seq_len", type=int, default=None, help="Override sequence length; default uses cdr3_b length.")
    p.add_argument("--steps", type=int, default=10, help="Flow steps during sampling.")
    p.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature when decoding.")
    p.add_argument("--use_esm", action="store_true", help="Use ESM backbone (must match retriever training)")
    p.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D", help="ESM model name")
    p.add_argument("--use_lora", action="store_true", help="Enable LoRA (must match retriever training)")
    p.add_argument("--lora_rank", type=int, default=8)
    return p.parse_args()


def load_records(path: Path, limit: int) -> List[Dict]:
    records: List[Dict] = []
    if path.suffix == ".csv":
        import csv

        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
                if len(records) >= limit:
                    break
    else:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if len(records) >= limit:
                    break
    return records


def _pad_id(tokenizer) -> int:
    if hasattr(tokenizer, "pad_token_id"):
        return int(tokenizer.pad_token_id)
    if isinstance(tokenizer, BasicTokenizer):
        return tokenizer.stoi["[PAD]"]
    return 0


def decode_tokens(tokenizer, tokens: torch.Tensor) -> str:
    vocab = tokenizer.itos if isinstance(tokenizer, BasicTokenizer) else []
    specials = set(["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[TRA]", "[TRB]", "[PEP]", "[MHC]"])
    seq_chars: List[str] = []
    for idx in tokens.tolist():
        if 0 <= idx < len(vocab):
            tok = vocab[idx]
            if tok in specials or len(tok) != 1:
                continue
            seq_chars.append(tok)
    return "".join(seq_chars)


def build_model(args, tokenizer, cond_dim: int) -> FlowMatchingModel:
    config_path = Path(args.model).with_name("flow_config.json")
    cfg: Dict[str, Optional[int]] = {}
    if config_path.exists():
        with config_path.open() as f:
            cfg = json.load(f)
    model = FlowMatchingModel(
        vocab_size=vocab_size(tokenizer),
        hidden_dim=cfg.get("hidden_dim", 256),
        n_layers=cfg.get("n_layers", 4),
        cond_dim=cond_dim,
        pad_id=cfg.get("pad_id", _pad_id(tokenizer)),
    )
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    prompts = load_records(Path(args.data), args.limit)
    if not prompts:
        print("No records loaded from", args.data)
        return

    cond_embedder = ConditionEmbedder(
        ckpt_path=args.retriever_ckpt,
        tokenizer=tokenizer,
        device=device,
        use_esm=args.use_esm,
        esm_model=args.esm_model,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
    )
    model = build_model(args, tokenizer, cond_embedder.cond_dim)

    metas: List[Dict] = []
    seq_lens: List[int] = []
    for rec in prompts:
        metas.append(rec)
        if args.seq_len:
            seq_lens.append(args.seq_len)
        else:
            seq_lens.append(len(rec.get("cdr3_b", rec.get("cdr3", ""))) or 16)

    with torch.no_grad():
        cond = cond_embedder(metas, device=device)
        max_len = max(seq_lens)
        tokens = model.sample(cond, seq_len=max_len, steps=args.steps, temperature=args.temperature)

    for i, rec in enumerate(prompts):
        decoded = decode_tokens(tokenizer, tokens[i])
        print(f"Input peptide={rec.get('peptide','')} mhc={rec.get('mhc','')}")
        print(f"Scaffold HV={rec.get('h_v_seq','')} HJ={rec.get('h_j_seq','')}")
        print(f"Generated CDR3b: {decoded}")
        print("-" * 40)


if __name__ == "__main__":
    main()
