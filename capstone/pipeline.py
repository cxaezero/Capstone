"""Inference pipeline shared by the demo and scripts: frame -> ESDNet -> clip -> X3D -> classifier."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .config import ANOMALY_THRESHOLD, CLIP_LEN
from .preprocess import normalize_clip


@dataclass
class StepResult:
    clean: torch.Tensor          # enhanced frame (3, H, W), float, roughly in [0, 1]
    score: float | None          # latest anomaly probability; None until the first full clip
    streak: int                  # consecutive clips with score >= threshold


class AnomalyPipeline:
    """Stateful per-stream pipeline. Call ``step`` once per frame (in display order).

    The clip buffer slides by one frame, so once it is full every frame triggers an X3D +
    classifier forward pass (as in the original demo).
    """

    def __init__(self, deweather, feature_extractor, classifier, *,
                 clip_len: int = CLIP_LEN, threshold: float = ANOMALY_THRESHOLD, device=None):
        self.deweather = deweather
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.clip_len = clip_len
        self.threshold = threshold
        self.device = torch.device(device) if device is not None else next(classifier.parameters()).device
        self.reset()

    def reset(self) -> None:
        self.buffer: deque[torch.Tensor] = deque(maxlen=self.clip_len)
        self.streak = 0
        self.last_score: float | None = None

    @torch.no_grad()
    def enhance(self, frame: torch.Tensor) -> torch.Tensor:
        """(3, H, W) or (1, 3, H, W) -> enhanced (3, H, W)."""
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        out_full, _, _ = self.deweather(frame.to(self.device))
        return out_full[0]

    @torch.no_grad()
    def classify(self, clip: torch.Tensor) -> float:
        """(3, T, H, W) enhanced clip -> anomaly probability."""
        x = normalize_clip(clip.to(self.device)).unsqueeze(0)
        feats = self.feature_extractor(x)
        logits, _ = self.classifier(feats)
        return torch.sigmoid(logits).item()

    @torch.no_grad()
    def step(self, frame: torch.Tensor) -> StepResult:
        clean = self.enhance(frame)
        self.buffer.append(clean)
        if len(self.buffer) == self.clip_len:
            score = self.classify(torch.stack(list(self.buffer), dim=1))
            self.last_score = score
            self.streak = self.streak + 1 if score >= self.threshold else 0
        return StepResult(clean=clean, score=self.last_score, streak=self.streak)
