#!/usr/bin/env python
"""Extract clip-level X3D features from (de-weathered) videos into .npy files.

For every video in the list, clips are sampled uniformly, each frame is passed through ESDNet,
the clip goes through X3D (head removed) and the per-clip features are max-pooled into one
array saved as ``<save-root>/<list entry>.npy``.

Requires the optional extraction dependencies:
  pip install av "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git"

Defaults reproduce the original script (x3d_m, 224 crop, 15 frames, assumed 15 fps). Note the
demo uses x3d_s at 160x160 -- see REFACTORING_PLAN.md B1.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from capstone import config  # noqa: E402
from capstone.data import label_from_path, list_path_for, read_list  # noqa: E402
from capstone.models import build_deweather, build_x3d  # noqa: E402
from capstone.preprocess import normalize_clip  # noqa: E402

log = logging.getLogger("extract")


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-root", required=True)
    p.add_argument("--save-root", required=True)
    p.add_argument("--list", default="Anomaly_Train.txt")
    p.add_argument("--variant", default=config.EXTRACT_X3D_VARIANT, choices=sorted(config.X3D_TRANSFORM_PARAMS))
    p.add_argument("--fps", type=float, default=config.EXTRACT_ASSUMED_FPS,
                   help="fps assumed when converting frames x sampling_rate into a clip duration")
    p.add_argument("--deweather-weights", default=config.DEWEATHER_WEIGHTS)
    p.add_argument("--x3d-weights", default=None, help="default: weights/<variant>_weights.pth if present, else torch.hub download")
    p.add_argument("--no-skip-existing", action="store_true", help="re-extract even if the .npy exists")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def build_transform(params):
    from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import CenterCrop, Compose, Lambda

    return ApplyTransformToKey("video", Compose([
        UniformTemporalSubsample(params["num_frames"]),
        Lambda(lambda x: x / 255.0),
        Permute((1, 0, 2, 3)),
        ShortSideScale(params["side_size"]),
        CenterCrop((params["crop_size"], params["crop_size"])),
        Permute((1, 0, 2, 3)),
    ]))


def collect_jobs(list_file, video_root, save_root, skip_existing: bool):
    jobs = []
    for rel in read_list(list_file):
        video_path = os.path.join(video_root, rel)
        out_path = os.path.join(save_root, os.path.splitext(rel)[0] + ".npy")
        if skip_existing and os.path.isfile(out_path):
            continue
        if not os.path.isfile(video_path):
            log.warning("missing video: %s", video_path)
            continue
        jobs.append((video_path, {"label": label_from_path(rel), "out_path": out_path}))
    return jobs


def save_feature(path, feature):
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, feature)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler

    device = config.get_device(args.device)
    deweather = build_deweather(weights=args.deweather_weights, device=device)  # raises if missing
    x3d = build_x3d(args.variant, weights_path=args.x3d_weights, device=device)

    params = config.X3D_TRANSFORM_PARAMS[args.variant]
    clip_duration = params["num_frames"] * params["sampling_rate"] / args.fps
    jobs = collect_jobs(list_path_for(args.list, config.LISTS_DIR), args.video_root, args.save_root,
                        skip_existing=not args.no_skip_existing)
    log.info("%d videos to process (clip duration %.2fs)", len(jobs), clip_duration)
    if not jobs:
        return

    dataset = LabeledVideoDataset(jobs, UniformClipSampler(clip_duration), build_transform(params), decode_audio=False)
    loader = DataLoader(dataset, batch_size=1)

    current_path, current_feat = None, None
    with torch.no_grad():
        for inputs in tqdm(loader):
            video = inputs["video"].to(device)                      # (B, 3, T, H, W) in [0, 1]
            b, c, t, h, w = video.shape
            frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            clean = deweather(frames)[0]                            # full-resolution output
            clip = clean.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            feats = x3d(normalize_clip(clip)).cpu().numpy()          # (B, 192, T', H', W')

            for feat, out_path in zip(feats, inputs["out_path"]):
                if out_path != current_path:
                    save_feature(current_path, current_feat)
                    current_path, current_feat = out_path, feat
                else:
                    current_feat = np.maximum(current_feat, feat)   # max-pool over clips of one video
    save_feature(current_path, current_feat)
    log.info("done")


if __name__ == "__main__":
    main()
