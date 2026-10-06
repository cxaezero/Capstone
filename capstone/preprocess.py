"""Frame / clip preprocessing shared by the demo, feature extraction and training."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import IMG_SIZE, MEAN, STD


def frame_to_tensor(frame_bgr: np.ndarray, size: int | tuple[int, int] = IMG_SIZE) -> torch.Tensor:
    """uint8 BGR (H, W, 3) -> float RGB (3, size, size) in [0, 1].

    Antialiased bilinear resize on the uint8 tensor, which matches the original
    ``PIL -> Resize -> ToTensor`` path without the PIL round trip.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    hw = [size, size] if isinstance(size, int) else list(size)
    t = TF.resize(t, hw, interpolation=InterpolationMode.BILINEAR, antialias=True)
    return t.float().div_(255.0)


def normalize_clip(clip: torch.Tensor, mean=MEAN, std=STD) -> torch.Tensor:
    """Normalize a clip whose channel axis is at ``dim=-4``: ``(..., 3, T, H, W)``.

    Equivalent to applying ``torchvision.transforms.Normalize`` to every frame, in one op.
    """
    m = torch.as_tensor(mean, device=clip.device, dtype=clip.dtype).view(3, 1, 1, 1)
    s = torch.as_tensor(std, device=clip.device, dtype=clip.dtype).view(3, 1, 1, 1)
    return (clip - m) / s


def tensor_to_bgr(img: torch.Tensor, size: tuple[int, int] | None = None) -> np.ndarray:
    """float RGB (3, H, W) (any range) -> uint8 BGR (H, W, 3), optionally resized to ``(w, h)``."""
    arr = img.detach().float().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if size is not None:
        arr = cv2.resize(arr, size)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
