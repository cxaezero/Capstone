import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from capstone.preprocess import frame_to_tensor, normalize_clip, tensor_to_bgr


def test_frame_to_tensor_shape_range_and_channel_order():
    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    frame[..., 0] = 255  # pure blue in BGR
    t = frame_to_tensor(frame, 160)
    assert t.shape == (3, 160, 160) and t.dtype == torch.float32
    assert t.min() >= 0 and t.max() <= 1
    assert torch.all(t[2] == 1.0) and torch.all(t[0] == 0.0)  # RGB: blue is channel 2


def test_frame_to_tensor_matches_original_pil_pipeline():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(288, 352, 3), dtype=np.uint8)
    rgb = frame[..., ::-1].copy()
    reference = Compose([Resize((160, 160)), ToTensor()])(Image.fromarray(rgb))
    ours = frame_to_tensor(frame, 160)
    assert (ours - reference).abs().max().item() <= 2 / 255


def test_normalize_clip_equals_per_frame_normalize():
    clip = torch.rand(3, 5, 8, 8)
    per_frame = torch.stack([Normalize([0.45] * 3, [0.225] * 3)(clip[:, t]) for t in range(5)], dim=1)
    assert torch.allclose(normalize_clip(clip), per_frame)
    batched = torch.rand(2, 3, 5, 8, 8)
    assert torch.allclose(normalize_clip(batched)[1], normalize_clip(batched[1]))


def test_tensor_to_bgr_swaps_channels_clips_and_resizes():
    img = torch.zeros(3, 4, 4)
    img[0] = 1.5  # red, out of range -> clipped to 255
    bgr = tensor_to_bgr(img)
    assert bgr.shape == (4, 4, 3) and bgr.dtype == np.uint8
    assert bgr[0, 0].tolist() == [0, 0, 255]
    assert tensor_to_bgr(img, size=(8, 6)).shape == (6, 8, 3)
