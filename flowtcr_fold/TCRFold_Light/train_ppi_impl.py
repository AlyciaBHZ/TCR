"""
Structure pretraining stub on generic PPI data.

Replace the placeholder loss/loader with real PDB parsing and interface labels.
"""

import argparse

import torch

from flowtcr_fold.TCRFold_Light.tcrfold_light import TCRFoldLight
from flowtcr_fold.common.utils import save_checkpoint, EarlyStopper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dummy_len", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", type=str, default="checkpoints/struct_ppi")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCRFoldLight().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=100)

    for epoch in range(args.epochs):
        # TODO: replace with real loader yielding labels {"distance","contact","energy"}
        s = torch.randn(args.dummy_len, model.s_dim).to(device)
        z = torch.randn(args.dummy_len, args.dummy_len, model.z_dim).to(device)
        out = model(s, z)
        # placeholder labels (zeros) until real structure/energy is available
        lbl_dist = torch.zeros_like(out["distance"])
        lbl_contact = torch.zeros_like(out["contact"])
        lbl_energy = torch.zeros(1, device=device)
        loss_dist = (out["distance"] - lbl_dist).pow(2).mean()
        loss_contact = torch.nn.functional.binary_cross_entropy(out["contact"], lbl_contact)
        loss_energy = (out["energy"] - lbl_energy).pow(2)
        loss = loss_dist + loss_contact + loss_energy
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, opt, args.out_dir, epoch + 1, tag="struct_ppi")
        if stopper.update(loss.item()):
            print(f"Early stopping at epoch {epoch+1}")
            break


if __name__ == "__main__":
    main()
