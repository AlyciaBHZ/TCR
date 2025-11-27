"""
Flow training stub for CDR3 generation.

Implements a minimal loop over FlowMatchingModel with dummy targets; replace
conditioning, schedules, and loss with the full flow-matching objective.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from flowtcr_fold.data.dataset import FlowDataset
from flowtcr_fold.data.tokenizer import vocab_size
from flowtcr_fold.FlowTCR_Gen.flow_gen import FlowMatchingModel
from flowtcr_fold.common.utils import save_checkpoint, EarlyStopper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/trn.csv")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", type=str, default="checkpoints/flow")
    return p.parse_args()


def collate(batch):
    tokens = [item["tokens_pos"] for item in batch]
    max_len = max(t.size(0) for t in tokens)
    padded = torch.zeros(len(tokens), max_len, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded[i, : t.size(0)] = t
    return padded


def main():
    args = parse_args()
    ds = FlowDataset(args.data, split="train")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMatchingModel(vocab_size=vocab_size(ds.tokenizer)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=100)

    for epoch in range(args.epochs):
        for tokens in loader:
            tokens = tokens.to(device)
            loss = model.flow_matching_loss(tokens, conditioning=None)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, opt, args.out_dir, epoch + 1, tag="flow")
        if stopper.update(loss.item()):
            print(f"Early stopping at epoch {epoch+1}")
            break

    save_checkpoint(model, opt, args.out_dir, epoch + 1, tag="flow_final")


if __name__ == "__main__":
    main()
