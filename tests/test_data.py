import cv2
import numpy as np
import pytest
import torch

from capstone.data import (NPYPairedDataset, PairedVideoDataset, collate_fn, label_from_path, list_path_for)

FEAT_SHAPE = (192, 2, 2, 2)


@pytest.fixture
def feature_root(tmp_path):
    names = ["Normal_Videos/N1.npy", "Normal_Videos/N2.npy", "Fighting/F1.npy", "Fighting/F2.npy", "Fighting/F3.npy"]
    for name in names:
        p = tmp_path / "feats" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, np.random.rand(*FEAT_SHAPE).astype(np.float32))
    (tmp_path / "list.txt").write_text("\n".join(names) + "\n")
    return tmp_path


def test_label_from_path():
    assert label_from_path("Normal_Videos/N1.npy") == 0.0
    assert label_from_path("Fighting/F1.npy") == 1.0


def test_paired_dataset_train_mode(feature_root):
    ds = NPYPairedDataset(feature_root / "list.txt", root=feature_root / "feats")
    assert len(ds) == 2  # min(2 normal, 3 abnormal)
    n_feat, n_lbl, a_feat, a_lbl = ds[0]
    assert n_feat.shape == FEAT_SHAPE and a_feat.shape == FEAT_SHAPE
    assert n_lbl.item() == 0.0 and a_lbl.item() == 1.0
    batch = collate_fn([ds[0], ds[1]])
    assert batch[0].shape == (2, *FEAT_SHAPE)


def test_paired_dataset_test_mode_keeps_order(feature_root):
    ds = NPYPairedDataset(feature_root / "list.txt", root=feature_root / "feats", test_mode=True)
    assert len(ds) == 5
    assert [ds[i][1].item() for i in range(5)] == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_missing_files_raise_or_skip(feature_root, caplog):
    lst = feature_root / "list_missing.txt"
    lst.write_text((feature_root / "list.txt").read_text() + "Fighting/Missing.npy\n")
    with pytest.raises(FileNotFoundError, match="1/6"):
        NPYPairedDataset(lst, root=feature_root / "feats", test_mode=True)
    ds = NPYPairedDataset(lst, root=feature_root / "feats", test_mode=True, missing="skip")
    assert len(ds) == 5
    assert "skipping" in caplog.text


def test_collate_drops_none():
    assert collate_fn([None, None]) is None
    out = collate_fn([None, (torch.zeros(2), torch.tensor(1.0))])
    assert out[0].shape == (1, 2)


def test_list_path_for(tmp_path):
    (tmp_path / "x.txt").write_text("a\n")
    assert list_path_for(str(tmp_path / "x.txt"), tmp_path / "lists") == tmp_path / "x.txt"
    assert list_path_for("y.txt", tmp_path / "lists") == tmp_path / "lists" / "y.txt"


def _write_video(path, n_frames=20, size=32):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), i * 10, dtype=np.uint8))
    writer.release()
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) >= n_frames - 1
    cap.release()
    return ok


def test_paired_video_dataset_random_clip(tmp_path):
    if not (_write_video(tmp_path / "Normal_Videos/a.mp4") and _write_video(tmp_path / "Fighting/b.mp4")):
        pytest.skip("OpenCV build cannot write/read mp4")
    ds = PairedVideoDataset(tmp_path, clip_len=5, img_size=16)
    assert len(ds) == 1
    n_clip, n_lbl, a_clip, a_lbl = ds[0]
    assert n_clip.shape == (3, 5, 16, 16) and a_clip.shape == (3, 5, 16, 16)
    assert (n_lbl.item(), a_lbl.item()) == (0.0, 1.0)
    assert ds.read_clip(str(tmp_path / "Fighting/b.mp4")).shape == (3, 5, 16, 16)
