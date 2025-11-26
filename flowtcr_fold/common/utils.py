"""
Shared training utilities.
- save_checkpoint: save model/optimizer
- EarlyStopper: stop after patience epochs without improvement
Prefs: ckpt every 50 epochs; early stop after 100 epochs (handled in scripts).
"""

import os
import torch


def save_checkpoint(model, optimizer, out_dir: str, epoch: int, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_epoch_{epoch}.pt"))
    torch.save(optimizer.state_dict(), os.path.join(out_dir, f"{tag}_epoch_{epoch}.opt"))


class EarlyStopper:
    def __init__(self, patience: int = 100):
        self.patience = patience
        self.best = None
        self.counter = 0

    def update(self, metric: float) -> bool:
        if self.best is None or metric < self.best:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
