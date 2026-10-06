import torch
from torch import nn

from capstone.pipeline import AnomalyPipeline


class StubDeweather(nn.Module):
    def forward(self, x):
        return x * 0.5, None, None


class StubX3D(nn.Module):
    def forward(self, x):  # (B, 3, T, H, W) -> (B, 192, T, 1, 1)
        return torch.zeros(x.shape[0], 192, x.shape[2], 1, 1)


class StubClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)  # gives the pipeline a device to infer
        self.logit = 0.0

    def forward(self, feats):
        b = feats.shape[0]
        return torch.full((b, 1), self.logit), torch.zeros(b, 32)


def make_pipeline(clip_len=3, threshold=0.9):
    return AnomalyPipeline(StubDeweather(), StubX3D(), StubClassifier(), clip_len=clip_len, threshold=threshold)


def test_no_score_until_buffer_full_then_streak_logic():
    p = make_pipeline()
    frame = torch.ones(3, 32, 32)
    r1, r2 = p.step(frame), p.step(frame)
    assert r1.score is None and r2.score is None and r2.streak == 0
    assert torch.allclose(r1.clean, frame * 0.5)

    p.classifier.logit = 10.0  # sigmoid ~ 1.0 >= threshold
    r3 = p.step(frame)
    assert r3.score is not None and r3.score > 0.99 and r3.streak == 1
    assert p.step(frame).streak == 2

    p.classifier.logit = -10.0
    r5 = p.step(frame)
    assert r5.streak == 0 and r5.score < 0.01

    p.reset()
    assert p.step(frame).score is None and len(p.buffer) == 1


def test_classify_accepts_clip():
    p = make_pipeline()
    p.classifier.logit = 0.0
    assert abs(p.classify(torch.rand(3, 3, 32, 32)) - 0.5) < 1e-6
