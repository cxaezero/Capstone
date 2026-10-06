#!/usr/bin/env python
"""Fine-tune ESDNet + X3D + classifier jointly on raw videos.

Batches are (normal, abnormal) clip pairs so the triplet loss assumption holds, clips start at a
random position in each video, and X3D BatchNorm statistics are frozen. Weights are written as
three separate state_dicts (deweather.pth / x3d.pth / classifier.pth) that the demo can load via
CAPSTONE_DEWEATHER_WEIGHTS / CAPSTONE_CLASSIFIER_WEIGHTS and --x3d-weights.

Example:
  python scripts/train_e2e.py --video-root data/UCF_Crimes/Videos --epochs 10
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
from torch import nn, optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from capstone import config  # noqa: E402
from capstone.data import PairedVideoDataset, collate_fn  # noqa: E402
from capstone.evaluation import compute_metrics, unpack_batch  # noqa: E402
from capstone.losses import CombinedLoss  # noqa: E402
from capstone.models import build_classifier, build_deweather, build_x3d  # noqa: E402
from capstone.preprocess import normalize_clip  # noqa: E402

log = logging.getLogger("train_e2e")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-root", required=True, help="directory with <class>/<video>.mp4")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2, help="pairs per batch (2x clips)")
    p.add_argument("--clip-len", type=int, default=15, help="frames per clip (original end-to-end script: 15)")
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--variant", default=config.X3D_VARIANT)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--deweather-weights", default=config.DEWEATHER_WEIGHTS)
    p.add_argument("--classifier-weights", default=None, help="default: train the classifier from scratch")
    p.add_argument("--no-freeze-bn", action="store_true", help="let X3D BatchNorm statistics update")
    p.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (smoke test)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def set_train_mode(x3d: nn.Module, freeze_bn: bool) -> None:
    x3d.train()
    if freeze_bn:
        for m in x3d.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def forward_clips(clips, deweather, x3d, classifier):
    """(B, 3, T, H, W) raw clips -> (logits, feats) through all three models."""
    b, c, t, h, w = clips.shape
    frames = clips.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    clean = deweather(frames)[0]
    clean = clean.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    feats = x3d(normalize_clip(clean))
    return classifier(feats)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    torch.manual_seed(args.seed)
    device = config.get_device(args.device)
    out_dir = args.out or config.RUNS_DIR / ("e2e-" + time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    deweather = build_deweather(weights=args.deweather_weights, device=device, eval_mode=False)
    x3d = build_x3d(args.variant, device=device)
    classifier = build_classifier(weights=args.classifier_weights, device=device, eval_mode=False)

    dataset = PairedVideoDataset(args.video_root, clip_len=args.clip_len, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_fn)
    params = list(deweather.parameters()) + list(x3d.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = CombinedLoss()

    step, history = 0, []
    for epoch in range(args.epochs):
        deweather.train()
        classifier.train()
        set_train_mode(x3d, freeze_bn=not args.no_freeze_bn)
        predictions, targets, losses = [], [], []
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            if batch is None:
                continue
            clips, labels = unpack_batch(batch, device)
            scores, feats = forward_clips(clips, deweather, x3d, classifier)
            loss = loss_fn(scores, feats, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            predictions.extend(torch.sigmoid(scores.reshape(-1)).detach().cpu().tolist())
            targets.extend(labels.reshape(-1).cpu().tolist())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        scheduler.step()
        metrics = compute_metrics(predictions, targets)
        metrics["loss"] = float(sum(losses) / max(len(losses), 1))
        history.append({"epoch": epoch, **metrics})
        log.info("epoch %d: %s", epoch, json.dumps(history[-1]))
        if args.max_steps and step >= args.max_steps:
            break

    torch.save(deweather.state_dict(), out_dir / "deweather.pth")
    torch.save(x3d.state_dict(), out_dir / "x3d.pth")
    torch.save(classifier.state_dict(), out_dir / "classifier.pth")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    log.info("saved deweather.pth / x3d.pth / classifier.pth to %s", out_dir)
    return history


if __name__ == "__main__":
    main()
