"""
FlowTCR-Gen training for CDR3β generation (conditional flow matching).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from flowtcr_fold.FlowTCR_Gen.flow_gen import ConditionEmbedder, FlowMatchingModel
from flowtcr_fold.common.utils import EarlyStopper
from flowtcr_fold.data.dataset import FlowDataset
from flowtcr_fold.data.tokenizer import BasicTokenizer, get_tokenizer, vocab_size


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="flowtcr_fold/data/trn.jsonl")
    p.add_argument("--val_data", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=10, help="Flow steps for sampling (not used in training loss)")
    p.add_argument("--retriever_ckpt", type=str, default="checkpoints/scaffold_v1/model_best.pt")
    p.add_argument("--retriever_hidden_dim", type=int, default=256)
    p.add_argument("--use_esm", action="store_true", help="Use ESM backbone inside retriever encoder")
    p.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D", help="ESM model name")
    p.add_argument("--use_lora", action="store_true", help="Enable LoRA inside retriever encoder")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="checkpoints/flow_gen")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def _pad_id(tokenizer) -> int:
    if hasattr(tokenizer, "pad_token_id"):
        return int(tokenizer.pad_token_id)
    if isinstance(tokenizer, BasicTokenizer):
        return tokenizer.stoi["[PAD]"]
    return 0


def _collate_fn(tokenizer):
    pad_id = _pad_id(tokenizer)

    def collate(batch: List[Dict]):
        cdr3_seqs = []
        masks = []
        metas = []
        for item in batch:
            sl = item["slices_pos"]["cdr3b"]
            cdr3 = item["tokens_pos"][sl]
            cdr3_seqs.append(cdr3)
            metas.append(item["meta"])
        max_len = max(t.size(0) for t in cdr3_seqs)
        padded = torch.full((len(cdr3_seqs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros_like(padded, dtype=torch.long)
        for i, seq in enumerate(cdr3_seqs):
            padded[i, : seq.size(0)] = seq
            mask[i, : seq.size(0)] = 1
        return {"cdr3": padded, "mask": mask, "meta": metas}

    return collate


def save_config(out_dir: Path, config: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "flow_config.json").open("w") as f:
        json.dump(config, f, indent=2)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    train_ds = FlowDataset(args.data, split="train", tokenizer=tokenizer)
    val_ds = FlowDataset(args.val_data, split="val", tokenizer=tokenizer) if args.val_data else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_fn(tokenizer),
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn(tokenizer))
        if val_ds
        else None
    )

    cond_embedder = ConditionEmbedder(
        ckpt_path=args.retriever_ckpt,
        tokenizer=tokenizer,
        device=device,
        hidden_dim=args.retriever_hidden_dim,
        use_esm=args.use_esm,
        esm_model=args.esm_model,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
    )

    model = FlowMatchingModel(
        vocab_size=vocab_size(tokenizer),
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        cond_dim=cond_embedder.cond_dim,
        pad_id=_pad_id(tokenizer),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=args.patience)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            tokens = batch["cdr3"].to(device)
            mask = batch["mask"].to(device)
            cond = cond_embedder(batch["meta"], device=device)
            loss = model.flow_matching_loss(tokens, cond_emb=cond, pad_mask=mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if step % args.log_every == 0:
                print(f"[epoch {epoch+1} step {step}] loss={loss.item():.4f}")

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        if val_loader:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch in val_loader:
                    tokens = batch["cdr3"].to(device)
                    mask = batch["mask"].to(device)
                    cond = cond_embedder(batch["meta"], device=device)
                    val_loss += model.flow_matching_loss(tokens, cond_emb=cond, pad_mask=mask).item()
                val_loss /= max(len(val_loader), 1)
            print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), out_dir / "flow_model_best.pt")
        else:
            if best_val is None or avg_loss < best_val:
                best_val = avg_loss
                torch.save(model.state_dict(), out_dir / "flow_model_best.pt")

        current_metric = val_loss if val_loader else avg_loss
        if stopper.update(current_metric):
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), out_dir / "flow_model_final.pt")
    save_config(
        out_dir,
        {
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "vocab_size": vocab_size(tokenizer),
            "pad_id": _pad_id(tokenizer),
            "retriever_ckpt": args.retriever_ckpt,
            "retriever_hidden_dim": args.retriever_hidden_dim,
            "use_esm": args.use_esm,
            "esm_model": args.esm_model,
            "use_lora": args.use_lora,
            "lora_rank": args.lora_rank,
        },
    )


if __name__ == "__main__":
    main()
