import torch

from capstone.losses import CombinedLoss, TripletLoss


def test_triplet_loss_value():
    normal = torch.zeros(2, 2)
    abnormal = torch.tensor([[3.0, 4.0], [3.0, 4.0]])  # distance 5 from every normal row
    loss = TripletLoss(margin=100.0)(torch.cat([normal, abnormal]))
    assert torch.isclose(loss, torch.tensor(95.0))      # max_n = 0, mean(clamp(100 - 5)) = 95


def test_triplet_loss_zero_when_margin_satisfied():
    normal = torch.zeros(2, 2)
    abnormal = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
    loss = TripletLoss(margin=1.0)(torch.cat([normal, abnormal]))
    assert loss.item() == 0.0


def test_combined_loss_accepts_column_and_flat_scores():
    feats = torch.randn(4, 8)
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    scores = torch.randn(4, 1)
    loss_fn = CombinedLoss()
    a = loss_fn(scores, feats, targets)
    b = loss_fn(scores.reshape(-1), feats, targets)
    assert torch.isfinite(a) and torch.isclose(a, b)
    assert a.item() > 0
