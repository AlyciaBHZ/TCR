"""
Unified structure training entrypoint (stub) combining geometry + energy regression.
"""

import argparse

import torch

from flowtcr_fold.TCRFold_Light.tcrfold_light import TCRFoldLight
from flowtcr_fold.common.utils import save_checkpoint, EarlyStopper


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default="checkpoints/struct")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCRFoldLight().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=100)

    for epoch in range(args.epochs):
        s = torch.randn(args.len, model.s_dim).to(device)
        z = torch.randn(args.len, args.len, model.z_dim).to(device)
        out = model(s, z)
        lbl_dist = torch.zeros_like(out["distance"])
        lbl_contact = torch.zeros_like(out["contact"])
        lbl_energy = torch.zeros(1, device=device)
        loss = (out["distance"] - lbl_dist).pow(2).mean() + torch.nn.functional.binary_cross_entropy(out["contact"], lbl_contact) + (out["energy"] - lbl_energy).pow(2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, opt, args.out_dir, epoch + 1, tag="struct")
        if stopper.update(loss.item()):
            print(f"Early stopping at epoch {epoch+1}")
            break


if __name__ == "__main__":
    main()
