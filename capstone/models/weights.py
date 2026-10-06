"""Checkpoint loading helpers (fail loudly, never fall back to random weights)."""
from __future__ import annotations

from pathlib import Path

import torch


def load_state_dict_file(path, map_location="cpu") -> dict:
    """Load a state_dict saved with ``torch.save`` (plain dict or ``{'state_dict': ...}``).

    Raises FileNotFoundError with a hint instead of silently continuing (REFACTORING_PLAN A8/A10).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"weights file not found: {path}\n"
            "Set CAPSTONE_WEIGHTS_DIR (or the specific CAPSTONE_*_WEIGHTS variable), or pass --weights."
        )
    obj = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not contain a state_dict (got {type(obj).__name__})")
    return obj
