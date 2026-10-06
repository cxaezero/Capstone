"""Anomaly classifier on top of X3D features.

Input: X3D features ``(B, 192, T, H, W)`` (head removed). Output: ``(logits (B, 1), feats (B, D))``.

Architecture (for the shipped config dims=(32, 32), depths=(1, 1)):
    LayerNorm(192) -> Linear(192, 32) -> ConvBlock -> [LayerNorm, Linear] -> AttnBlock (Performer)
    -> AdaptiveMaxPool3d -> Dropout -> LayerNorm -> Linear(32, 1)

Module attribute names must stay as they are: ``weights/De_final_model.pth`` is keyed by them.

KNOWN ISSUE (not fixed, see REFACTORING_PLAN.md A2): ``DECOUPLED.forward`` reinterprets a
``(B, T, H, W, C)`` tensor with ``view`` instead of ``permute``, so its 2D/1D convolutions do
not see a proper channel-first layout. The shipped weights were trained with this behaviour,
so it is kept for compatibility.
"""
import torch
from torch import nn
from performer_pytorch import Performer

X3D_FEATURE_DIM = 192


class AttnBlock(nn.Module):
    def __init__(self, dim, depth, dropout, attn_dropout, heads=16, ff_mult=2):
        super().__init__()
        self.performer = Performer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim // heads,
            causal=False,
            ff_mult=ff_mult,
            local_attn_heads=8,
            local_window_size=dim // 8,
            ff_dropout=dropout,
            attn_dropout=attn_dropout,
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = self.performer(x.reshape(B, -1, C))
        return x.view(B, T, H, W, C)


class ConvBlock(nn.Module):
    def __init__(self, *, dim, ff_mult=2, dropout=0.0, heads=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.conv = DECOUPLED(dim, heads)
        self.ff = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(
        self,
        *,
        dropout=0.2,
        attn_dropout=0.1,
        ff_mult=4,
        dims=(192, 128),
        depths=(3, 3),
        block_types=("c", "a"),
        in_dim=X3D_FEATURE_DIM,
    ):
        super().__init__()
        self.init_dim, *_, last_dim = dims

        self.stages = nn.ModuleList()
        for ind, (depth, block_type) in enumerate(zip(depths, block_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]

            if block_type == "c":
                for _ in range(depth):
                    self.stages.append(ConvBlock(dim=stage_dim, ff_mult=ff_mult, dropout=dropout))
            elif block_type == "a":
                for _ in range(depth):
                    self.stages.append(AttnBlock(stage_dim, 1, dropout, attn_dropout, ff_mult=ff_mult))
            else:
                raise ValueError(f"unknown block type {block_type!r} (expected 'c' or 'a')")

            if not is_last:
                self.stages.append(nn.Sequential(nn.LayerNorm(stage_dim), nn.Linear(stage_dim, dims[ind + 1])))

        self.norm0 = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, dims[0])
        self.norm = nn.LayerNorm(last_dim)
        self.fc = nn.Linear(last_dim, 1)
        self.drop_out = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, x):
        """x: (B, C, T, H, W) -> (logits (B, 1), feats (B, last_dim))."""
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        if x.shape[4] != self.init_dim:
            x = self.linear(self.norm0(x))

        for stage in self.stages:
            x = stage(x)

        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        # flatten(1) instead of squeeze(): squeeze() also dropped the batch axis when B == 1
        x = self.pooling(x).flatten(1)

        x = self.drop_out(x)
        x = self.norm(x)
        logits = self.fc(x)
        return logits, x


AnomalyClassifier = Model


def FeedForward(dim, repe=4, dropout=0.0):
    return nn.Sequential(
        nn.Linear(dim, dim * repe),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * repe, dim),
        nn.GELU(),
    )


class DECOUPLED(nn.Module):
    """Intended as a grouped 2D (spatial) conv followed by a grouped 1D (temporal) conv.

    The ``view`` calls below do NOT move the channel axis first (that would need
    ``x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)`` etc.), so the convolutions run over a
    scrambled layout. Kept as-is because the shipped weights depend on it (REFACTORING_PLAN A2).
    """

    def __init__(self, dim, heads, kernel=3):
        super().__init__()
        self.heads = heads
        self.norm2d = nn.BatchNorm2d(dim)
        self.norm1d = nn.BatchNorm1d(dim)
        self.conv2d = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=heads)
        self.conv1d = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=heads)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, C, H, W)  # NOTE: layout bug kept for weight compatibility (see class doc)
        x = self.conv2d(self.norm2d(x))
        x = x.reshape(B * H * W, C, T)
        x = self.conv1d(self.norm1d(x))
        return x.view(B, T, H, W, C)
