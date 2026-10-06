"""Datasets for the anomaly classifier.

Labels come from one place: ``label_from_path`` (a path containing ``Normal`` is normal).
"""
from __future__ import annotations

import logging
import os
import random
from glob import glob
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from .config import CLIP_LEN, IMG_SIZE
from .preprocess import frame_to_tensor

log = logging.getLogger(__name__)


def label_from_path(path) -> float:
    """0.0 for normal videos (path contains 'Normal'), 1.0 otherwise."""
    return 0.0 if "Normal" in str(path) else 1.0


def read_list(list_path) -> list[str]:
    with open(list_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def collate_fn(batch):
    """``default_collate`` that drops ``None`` samples; returns ``None`` if nothing is left."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


def _split_by_label(paths: Sequence[str]) -> tuple[list[int], list[int]]:
    normal = [i for i, p in enumerate(paths) if label_from_path(p) == 0.0]
    abnormal = [i for i, p in enumerate(paths) if label_from_path(p) == 1.0]
    if not normal or not abnormal:
        raise ValueError(f"need both normal and abnormal samples, got {len(normal)} normal / {len(abnormal)} abnormal")
    return normal, abnormal


def _random_index(n: int) -> int:
    # torch's RNG is seeded per DataLoader worker and per epoch, unlike a module-level random.Random
    return int(torch.randint(n, (1,)).item())


class NPYPairedDataset(Dataset):
    """Pre-extracted X3D features (``.npy``) listed in a text file.

    * ``test_mode=False``: ``__getitem__`` returns ``(normal_feat, 0.0, abnormal_feat, 1.0)``.
      Every normal sample is visited once per epoch; the abnormal partner is drawn at random
      each time, so pairs differ between epochs and all abnormal samples get used.
    * ``test_mode=True``: returns ``(feat, label)`` for every line in order.

    Missing files raise at construction (``missing="error"``) or are dropped with a warning
    (``missing="skip"``) instead of being silently swallowed at ``__getitem__`` time.
    """

    def __init__(self, list_path, root=None, test_mode: bool = False, missing: str = "error"):
        self.test_mode = test_mode
        rel_paths = read_list(list_path)
        paths = [os.path.join(root, p) if root else p for p in rel_paths]

        absent = [p for p in paths if not os.path.isfile(p)]
        if absent:
            msg = f"{len(absent)}/{len(paths)} feature files missing, e.g. {absent[:3]}"
            if missing == "error":
                raise FileNotFoundError(msg + " (pass missing='skip' to drop them)")
            if missing != "skip":
                raise ValueError("missing must be 'error' or 'skip'")
            log.warning("%s -- skipping them", msg)
            paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            raise ValueError(f"no usable entries in {list_path}")

        self.files = paths
        self.labels = [label_from_path(p) for p in paths]
        if test_mode:
            self.length = len(paths)
        else:
            self.normal, self.abnormal = _split_by_label(paths)
            self.length = min(len(self.normal), len(self.abnormal))

    def __len__(self):
        return self.length

    def _load(self, i: int) -> torch.Tensor:
        # allow_pickle=True kept for compatibility with features saved by older numpy
        return torch.from_numpy(np.load(self.files[i], allow_pickle=True).astype(np.float32))

    def __getitem__(self, index):
        if self.test_mode:
            return self._load(index), torch.tensor(self.labels[index])
        n = self.normal[index % len(self.normal)]
        a = self.abnormal[_random_index(len(self.abnormal))]
        return self._load(n), torch.tensor(self.labels[n]), self._load(a), torch.tensor(self.labels[a])


class PairedVideoDataset(Dataset):
    """Pairs of (normal clip, abnormal clip) decoded from videos with OpenCV, for end-to-end training.

    Each clip of ``clip_len`` consecutive frames starts at a uniformly random position inside
    the video (the original script always took the first 15 frames). Returns
    ``(normal_clip, 0.0, abnormal_clip, 1.0)`` with clips shaped ``(3, T, size, size)``, or
    ``None`` when a video cannot be decoded (use ``collate_fn`` to drop those).
    """

    def __init__(self, video_root, clip_len: int = CLIP_LEN, img_size: int = IMG_SIZE, pattern: str = "*/*.mp4"):
        paths = sorted(glob(os.path.join(video_root, pattern)))
        if not paths:
            raise FileNotFoundError(f"no videos matching {pattern} under {video_root}")
        self.paths = paths
        self.clip_len = clip_len
        self.img_size = img_size
        self.normal, self.abnormal = _split_by_label(paths)

    def __len__(self):
        return min(len(self.normal), len(self.abnormal))

    def read_clip(self, path: str) -> torch.Tensor | None:
        cap = cv2.VideoCapture(path)
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_frames < self.clip_len:
                log.warning("%s has %d frames (< clip_len %d), skipping", path, n_frames, self.clip_len)
                return None
            start = random.randint(0, n_frames - self.clip_len)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            while len(frames) < self.clip_len:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame_to_tensor(frame, self.img_size))
        finally:
            cap.release()
        if len(frames) < self.clip_len:
            log.warning("%s: decoded %d/%d frames from %d, skipping", path, len(frames), self.clip_len, start)
            return None
        return torch.stack(frames, dim=1)

    def __getitem__(self, index):
        n_path = self.paths[self.normal[index % len(self.normal)]]
        a_path = self.paths[self.abnormal[_random_index(len(self.abnormal))]]
        n_clip, a_clip = self.read_clip(n_path), self.read_clip(a_path)
        if n_clip is None or a_clip is None:
            return None
        return n_clip, torch.tensor(label_from_path(n_path)), a_clip, torch.tensor(label_from_path(a_path))


def list_path_for(name: str, lists_dir: Path) -> Path:
    """Resolve ``name`` against ``lists_dir`` unless it is already an existing path."""
    p = Path(name)
    return p if p.is_file() else lists_dir / name
