"""
Domain finetuning stub for TCR-specific structural supervision.

Populate with STCRDab/TCR3d loaders and interface/contact losses.
"""

import argparse

import torch

from flowtcr_fold.TCRFold_Light.tcrfold_light import TCRFoldLight
from flowtcr_fold.common.utils import save_checkpoint, EarlyStopper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", type=str, default="checkpoints/struct_tcr")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCRFoldLight().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=100)

    for epoch in range(args.epochs):
        # Placeholder tensors; replace with real batches
        s = torch.randn(64, model.s_dim).to(device)
        z = torch.randn(64, 64, model.z_dim).to(device)
        out = model(s, z)
        lbl_contact = torch.zeros_like(out["contact"])
        lbl_energy = torch.zeros(1, device=device)
        loss_contact = torch.nn.functional.binary_cross_entropy(out["contact"], lbl_contact)
        loss_energy = (out["energy"] - lbl_energy).pow(2)
        loss = loss_contact + loss_energy
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, opt, args.out_dir, epoch + 1, tag="struct_tcr")
        if stopper.update(loss.item()):
            print(f"Early stopping at epoch {epoch+1}")
            break


if __name__ == "__main__":
    main()
