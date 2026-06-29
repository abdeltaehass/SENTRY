"""Training loop for the incident-report model.

Image-captioning-style fine-tune: freeze the backbone, train LoRA on the OPT
decoder + the language_projection layer, learning to generate the report text
from the image. A manual loop (rather than Trainer) keeps device handling
explicit and predictable on Apple MPS.

Usage:
    # quick end-to-end / overfit sanity check
    PYTHONPATH=src python -m model.train --config configs/default.yaml --subset 64 --max-steps 80
    # full run (e.g. on a cloud GPU)
    PYTHONPATH=src python -m model.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from config import load_config
from data.dataset import IncidentReportDataset, ReportCollator

from .model import build_model, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the incident-report generator")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--subset", type=int, default=None, help="train on first N samples only")
    p.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--target", default=None, help="override data.target (the text field to generate)")
    p.add_argument("--output-dir", default=None, help="override training.output_dir")
    return p.parse_args()


def init_wandb(cfg):
    import os

    mode = os.environ.get("WANDB_MODE") or cfg.get("wandb.mode", "online")
    if not cfg.get("wandb.enabled", False) or mode == "disabled":
        return None
    try:
        import wandb

        return wandb.init(
            project=cfg.get("wandb.project", "sentry"),
            entity=cfg.get("wandb.entity"),
            mode=mode,
            config=cfg.raw,
        )
    except Exception as exc:  # e.g. not logged in — keep training regardless
        print(f"[sentry] W&B disabled ({exc})")
        return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.target:
        cfg.raw.setdefault("data", {})["target"] = args.target
    if args.output_dir:
        cfg.raw.setdefault("training", {})["output_dir"] = args.output_dir
    torch.manual_seed(cfg.get("project.seed", 42))

    device = pick_device()
    epochs = args.epochs or cfg.training.epochs
    max_steps = args.max_steps or cfg.get("training.max_steps")
    print(f"[sentry] device={device} | approach={cfg.model.approach} | target={cfg.data.target}")

    model, processor = build_model(cfg, device=device)
    model.to(device)
    model.train()

    train_ds: torch.utils.data.Dataset = IncidentReportDataset(
        cfg.data.train_split, text_field=cfg.data.target
    )
    if args.subset:
        train_ds = Subset(train_ds, list(range(min(args.subset, len(train_ds)))))
    collator = ReportCollator(
        processor,
        max_target_tokens=cfg.data.max_target_tokens,
        prompt=cfg.get("model.prompt", ""),   # prompt-conditioned (masked in the loss)
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    print(f"[sentry] train samples={len(train_ds)} | batch={cfg.training.batch_size} | epochs={epochs}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    run = init_wandb(cfg)
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            step += 1
            if step % 5 == 0 or step == 1:
                print(f"  epoch {epoch} step {step:4d} | loss {loss.item():.4f}")
            if run is not None:
                run.log({"train/loss": loss.item(), "epoch": epoch}, step=step)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    out_dir = cfg.training.output_dir
    model.save_pretrained(out_dir)        # saves LoRA adapter + language_projection only
    processor.save_pretrained(out_dir)
    print(f"[sentry] done ({step} steps). adapter saved -> {out_dir}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
