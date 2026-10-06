"""X3D backbone (facebookresearch/pytorchvideo) used as a clip feature extractor."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..config import X3D_VARIANT, X3D_WEIGHTS
from .weights import load_state_dict_file

HUB_REPO = "facebookresearch/pytorchvideo"
X3D_FEATURE_DIM = 192  # channels of the last residual stage for x3d_s / x3d_m

log = logging.getLogger(__name__)


def build_x3d(variant: str = X3D_VARIANT, weights_path=None, device="cpu",
              download_pretrained: bool | None = None) -> torch.nn.Module:
    """Return X3D with its classification head removed: input (B, 3, T, H, W) -> (B, 192, T, H', W').

    ``weights_path``: local state_dict (saved with or without the head). When omitted, the file
    in ``config.X3D_WEIGHTS[variant]`` is used if it exists; otherwise (or if
    ``download_pretrained=True``) the pretrained weights are downloaded through torch.hub,
    which is what the original scripts always did.
    """
    if weights_path is None:
        candidate = X3D_WEIGHTS.get(variant)
        if candidate is not None and Path(candidate).is_file():
            weights_path = candidate
    if download_pretrained is None:
        download_pretrained = weights_path is None

    model = torch.hub.load(HUB_REPO, variant, pretrained=download_pretrained, verbose=False, skip_validation=True)
    head_prefix = f"blocks.{len(model.blocks) - 1}."
    del model.blocks[-1]

    if weights_path is not None:
        state = load_state_dict_file(weights_path)
        state = {k: v for k, v in state.items() if not k.startswith(head_prefix)}
        model.load_state_dict(state, strict=True)
        log.info("X3D %s weights loaded from %s", variant, weights_path)
    else:
        log.info("X3D %s pretrained weights downloaded via torch.hub", variant)

    return model.eval().to(device)
