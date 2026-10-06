import pytest
import torch

from capstone import config
from capstone.models.classifier import Model
from capstone.models.esdnet import ESDNet

TINY_ESDNET = dict(en_feature_num=4, en_inter_num=2, de_feature_num=4, de_inter_num=2, sam_number=1)


def test_esdnet_output_shapes():
    net = ESDNet(**TINY_ESDNET).eval()
    x = torch.rand(2, 3, 64, 32)
    with torch.no_grad():
        full, half, quarter = net(x)
    assert full.shape == (2, 3, 64, 32)
    assert half.shape == (2, 3, 32, 16)
    assert quarter.shape == (2, 3, 16, 8)


def test_esdnet_rejects_sizes_not_multiple_of_32():
    net = ESDNet(**TINY_ESDNET).eval()
    with pytest.raises(ValueError, match="multiples of 32"):
        net(torch.rand(1, 3, 48, 32))


@pytest.mark.parametrize("batch", [1, 4])
def test_classifier_shapes_including_batch_one(batch):
    model = Model(**config.CLASSIFIER_KWARGS).eval()
    x = torch.randn(batch, 192, 13, 5, 5)
    with torch.no_grad():
        logits, feats = model(x)
    assert logits.shape == (batch, 1)          # squeeze() used to collapse this to (1,) for batch 1
    assert feats.shape == (batch, config.CLASSIFIER_KWARGS["dims"][-1])


def test_classifier_rejects_unknown_block_type():
    with pytest.raises(ValueError):
        Model(dims=(32,), depths=(1,), block_types=("x",))


@pytest.mark.skipif(not config.CLASSIFIER_WEIGHTS.is_file() or not config.DEWEATHER_WEIGHTS.is_file(),
                    reason="shipped weights not present")
def test_shipped_weights_load_strictly():
    from capstone.models import build_classifier, build_deweather

    clf = build_classifier()
    net = build_deweather()
    assert not clf.training and not net.training
    with torch.no_grad():
        out = net(torch.rand(1, 3, 32, 32))[0]
    assert out.shape == (1, 3, 32, 32)
