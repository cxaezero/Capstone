#!/usr/bin/env python
"""Evaluate the anomaly classifier on pre-extracted X3D features.

Example:
  python scripts/evaluate.py --root UCF_synth/X3D_Videos_T --list ucf_x3d_test.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader  # noqa: E402

from capstone import config  # noqa: E402
from capstone.data import NPYPairedDataset, collate_fn, list_path_for  # noqa: E402
from capstone.evaluation import evaluate  # noqa: E402
from capstone.models import build_classifier  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="directory containing the .npy features")
    p.add_argument("--list", default="ucf_x3d_test.txt", help=f"list file (name under {config.LISTS_DIR} or a path)")
    p.add_argument("--weights", default=config.CLASSIFIER_WEIGHTS)
    p.add_argument("--batch-size", type=int, default=config.TRAIN_DEFAULTS["batch_size"])
    p.add_argument("--num-workers", type=int, default=config.TRAIN_DEFAULTS["num_workers"])
    p.add_argument("--missing", choices=["error", "skip"], default="error", help="what to do with missing feature files")
    p.add_argument("--device", default=None)
    p.add_argument("--json", type=Path, default=None, help="also write metrics to this JSON file")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    device = config.get_device(args.device)

    model = build_classifier(weights=args.weights, device=device)
    dataset = NPYPairedDataset(list_path_for(args.list, config.LISTS_DIR), root=args.root, test_mode=True,
                               missing=args.missing)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn)

    metrics = evaluate(loader, model, device)
    print(f"[Evaluation] n={metrics['n']} Accuracy: {metrics['accuracy']:.4f}, "
          f"PR AUC: {metrics['pr_auc']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
