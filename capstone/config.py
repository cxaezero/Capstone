"""Single source of truth for paths and hyperparameters.

Values are intentionally identical to the original scripts (the trained weights depend on
them). Paths can be overridden with environment variables so nothing is hard-coded to one
machine. See REFACTORING_PLAN.md (B1) for the known train/serve preprocessing mismatch.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


# --------------------------------------------------------------------------- paths
WEIGHTS_DIR = _env_path("CAPSTONE_WEIGHTS_DIR", ROOT / "weights")
LISTS_DIR = _env_path("CAPSTONE_LISTS_DIR", ROOT / "data" / "lists")
RUNS_DIR = _env_path("CAPSTONE_RUNS_DIR", ROOT / "runs")

DEWEATHER_WEIGHTS = _env_path("CAPSTONE_DEWEATHER_WEIGHTS", WEIGHTS_DIR / "deweathering_model.pth")
CLASSIFIER_WEIGHTS = _env_path("CAPSTONE_CLASSIFIER_WEIGHTS", WEIGHTS_DIR / "De_final_model.pth")
X3D_WEIGHTS = {
    "x3d_s": WEIGHTS_DIR / "x3d_s_weights.pth",
    "x3d_m": WEIGHTS_DIR / "x3d_m_weights.pth",
}

DEMO_VIDEO = _env_path("CAPSTONE_DEMO_VIDEO", ROOT / "demo" / "demo_video.mp4")
RTMP_BASE_URL = os.environ.get("CAPSTONE_RTMP_URL", "rtmp://localhost:1935/live")

# --------------------------------------------------------------------------- models
ESDNET_KWARGS = dict(en_feature_num=48, en_inter_num=32, de_feature_num=64, de_inter_num=32, sam_number=1)
CLASSIFIER_KWARGS = dict(ff_mult=1, dims=(32, 32), depths=(1, 1))

# X3D feature extractor used by the demo / end-to-end training (NOT the same as the
# offline feature extractor below -- known limitation, see REFACTORING_PLAN.md B1).
X3D_VARIANT = "x3d_s"
CLIP_LEN = 13          # frames per clip fed to X3D
IMG_SIZE = 160         # frames are resized to IMG_SIZE x IMG_SIZE (must be a multiple of 32 for ESDNet)
MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

# Offline feature extraction (scripts/extract_features.py). Kept exactly as the original
# script so re-extracted features stay compatible with the shipped classifier weights.
EXTRACT_X3D_VARIANT = "x3d_m"
EXTRACT_ASSUMED_FPS = 15   # original assumption; UCF-Crime videos are actually 30 fps
X3D_TRANSFORM_PARAMS = {
    "x3d_s": {"side_size": 160, "crop_size": 160, "num_frames": 13, "sampling_rate": 6},
    "x3d_m": {"side_size": 224, "crop_size": 224, "num_frames": 15, "sampling_rate": 5},
    "x3d_l": {"side_size": 320, "crop_size": 320, "num_frames": 16, "sampling_rate": 5},
}

# --------------------------------------------------------------------------- training
TRAIN_DEFAULTS = dict(epochs=100, batch_size=4, lr=1e-4, weight_decay=1e-5, num_workers=0)
TRIPLET_MARGIN = 100.0   # see REFACTORING_PLAN.md B10 (kept for weight compatibility)
TRIPLET_ALPHA = 0.01

# --------------------------------------------------------------------------- demo
ANOMALY_THRESHOLD = 0.90   # sigmoid score at/above which a clip counts as anomalous
ALERT_FRAMES = 10          # consecutive anomalous clips before the red border + log
MAX_LOGS = 25
LOG_INTERVAL = 5.0         # seconds between alert log lines per stream
DISPLAY_SIZE = (352, 288)  # (width, height) of the enhanced stream shown on the dashboard
DEMO_PORT = int(os.environ.get("CAPSTONE_DEMO_PORT", "5050"))
SHARE_MODELS = os.environ.get("CAPSTONE_SHARE_MODELS", "0") == "1"  # one model set for all streams


def get_device(override: str | None = None) -> torch.device:
    """CUDA if available, else CPU. ``CAPSTONE_DEVICE`` (or ``override``) forces a device."""
    name = override or os.environ.get("CAPSTONE_DEVICE")
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
