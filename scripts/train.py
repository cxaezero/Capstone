#!/usr/bin/env python
"""Train the anomaly classifier on pre-extracted X3D features.

Example (train from scratch, validate every epoch, keep best by ROC-AUC):
  python scripts/train.py --train-root UCF_synth/De_X3D_Videos --val-root UCF_synth/De_X3D_Videos_T

Fine-tune the shipped weights instead:   --init-weights weights/De_final_model.pth
Resume an interrupted run:               --resume runs/<run>/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from capstone import config  # noqa: E402
from capstone.data import NPYPairedDataset, collate_fn, list_path_for  # noqa: E402
from capstone.evaluation import compute_metrics, evaluate, unpack_batch  # noqa: E402
from capstone.losses import CombinedLoss  # noqa: E402
from capstone.models import build_classifier  # noqa: E402

log = logging.getLogger("train")


def parse_args(argv=None):
    d = config.TRAIN_DEFAULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-root", required=True)
    p.add_argument("--train-list", default="ucf_x3d_train.txt")
    p.add_argument("--val-root", default=None, help="enable per-epoch validation on this feature root")
    p.add_argument("--val-list", default="ucf_x3d_test.txt")
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"], help="pairs per batch (2x samples)")
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--num-workers", type=int, default=d["num_workers"])
    p.add_argument("--init-weights", default=None, help="start from this state_dict (default: random init)")
    p.add_argument("--resume", default=None, help="checkpoint.pt written by a previous run")
    p.add_argument("--out", type=Path, default=None, help=f"output dir (default {config.RUNS_DIR}/<timestamp>)")
    p.add_argument("--missing", choices=["error", "skip"], default="error")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def train_one_epoch(loader, model, optimizer, loss_fn, device, epoch: int) -> dict:
    model.train()
    predictions, targets, losses = [], [], []
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        if batch is None:
            continue
        inputs, labels = unpack_batch(batch, device)
        scores, feats = model(inputs)
        loss = loss_fn(scores, feats, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        predictions.extend(torch.sigmoid(scores.reshape(-1)).detach().cpu().tolist())
        targets.extend(labels.reshape(-1).cpu().tolist())

    metrics = compute_metrics(predictions, targets)  # note: computed in train mode (dropout on)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    torch.manual_seed(args.seed)
    device = config.get_device(args.device)
    out_dir = args.out or config.RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("device=%s out=%s", device, out_dir)

    model = build_classifier(weights=args.init_weights, device=device, eval_mode=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # stepped once per epoch
    loss_fn = CombinedLoss()

    start_epoch, best_score = 0, float("-inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_score = ckpt["epoch"] + 1, ckpt.get("best_score", best_score)
        log.info("resumed from %s at epoch %d", args.resume, start_epoch)

    train_set = NPYPairedDataset(list_path_for(args.train_list, config.LISTS_DIR), root=args.train_root,
                                 missing=args.missing)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = None
    if args.val_root:
        val_set = NPYPairedDataset(list_path_for(args.val_list, config.LISTS_DIR), root=args.val_root,
                                   test_mode=True, missing=args.missing)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn)

    history = []
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(train_loader, model, optimizer, loss_fn, device, epoch)
        scheduler.step()
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train": train_metrics}
        if val_loader is not None:
            record["val"] = evaluate(val_loader, model, device, progress=False)
            score = record["val"]["roc_auc"]
        else:
            score = -train_metrics["loss"]
        history.append(record)
        log.info("epoch %d: %s", epoch, json.dumps(record))

        torch.save(model.state_dict(), out_dir / "last.pth")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_score": best_score},
                   out_dir / "checkpoint.pt")
        if score == score and score > best_score:  # NaN-safe
            best_score = score
            torch.save(model.state_dict(), out_dir / "best.pth")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    log.info("done. best=%.4f weights in %s (best.pth / last.pth)", best_score, out_dir)
    return history


if __name__ == "__main__":
    main()
