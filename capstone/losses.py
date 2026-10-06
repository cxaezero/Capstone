"""Losses for the anomaly classifier (one definition instead of three copies)."""
from __future__ import annotations

import torch
from torch import nn

from .config import TRIPLET_ALPHA, TRIPLET_MARGIN


class TripletLoss(nn.Module):
    """Pull normal features together, push abnormal features at least ``margin`` away.

    ``feats`` must be ``(2N, D)`` with the first N rows normal and the last N abnormal, which is
    how ``NPYPairedDataset`` batches are concatenated in the training scripts.

    Note: the shipped margin (100) is far larger than the distance range of LayerNorm'd
    features, so the hinge term never switches off (REFACTORING_PLAN B10, kept as-is).
    """

    def __init__(self, margin: float = TRIPLET_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        half = feats.size(0) // 2
        normal, abnormal = feats[:half], feats[half:]
        dist_n = torch.cdist(normal, normal, p=2)
        dist_a = torch.cdist(normal, abnormal, p=2)
        max_n = dist_n.max(dim=0).values
        min_a = dist_a.min(dim=0).values
        return max_n.mean() + torch.clamp(self.margin - min_a, min=0).mean()


class CombinedLoss(nn.Module):
    """BCE on logits + ``alpha`` * TripletLoss on features."""

    def __init__(self, alpha: float = TRIPLET_ALPHA, margin: float = TRIPLET_MARGIN):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss(margin)

    def forward(self, scores: torch.Tensor, feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(scores.reshape(-1), targets.reshape(-1).float())
        return loss_bce + self.alpha * self.triplet(feats)
