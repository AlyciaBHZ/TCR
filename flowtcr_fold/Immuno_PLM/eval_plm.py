"""
Evaluation scaffold for Immuno-PLM embeddings.

Computes simple retrieval metrics: cosine similarity of anchor vs negatives,
and reports separation.
"""

import argparse
import torch
from torch.utils.data import DataLoader

from flowtcr_fold.data.dataset import FlowDataset
from flowtcr_fold.data.tokenizer import vocab_size
from flowtcr_fold.Immuno_PLM.immuno_plm import ImmunoPLM


def collate(batch):
    tokens = torch.stack([item["tokens_pos"] for item in batch])
    mask = torch.stack([item["mask_pos"] for item in batch])
    return {"tokens": tokens, "mask": mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/val.csv")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    ds = FlowDataset(args.data, split="val")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImmunoPLM(use_esm=False, vocab_size=vocab_size(ds.tokenizer)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    sims = []
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            out = model(tokens, mask)
            emb = out["pooled"]  # [B, D]
            norm = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            sim = norm @ norm.t()
            sims.append(sim.cpu())
    sims = torch.cat([s.flatten() for s in sims])
    print(f"Similarity mean {sims.mean():.4f}, std {sims.std():.4f}")


if __name__ == "__main__":
    main()
