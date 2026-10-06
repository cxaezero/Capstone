"""Smoke tests that run the CLI scripts end-to-end on tiny synthetic features."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_features(tmp_path):
    names = [f"Normal_Videos/N{i}.npy" for i in range(4)] + [f"Fighting/F{i}.npy" for i in range(4)]
    for name in names:
        p = tmp_path / "feats" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, np.random.rand(192, 2, 2, 2).astype(np.float32))
    (tmp_path / "list.txt").write_text("\n".join(names) + "\n")
    return tmp_path


def run(script, *args):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr


def test_train_then_evaluate(synthetic_features):
    out = synthetic_features / "run"
    log = run("train.py", "--train-root", synthetic_features / "feats", "--train-list", synthetic_features / "list.txt",
              "--val-root", synthetic_features / "feats", "--val-list", synthetic_features / "list.txt",
              "--epochs", 2, "--batch-size", 2, "--out", out, "--device", "cpu")
    assert (out / "best.pth").is_file() and (out / "last.pth").is_file() and (out / "checkpoint.pt").is_file()
    history = json.loads((out / "history.json").read_text())
    assert len(history) == 2 and "val" in history[0]
    assert history[1]["lr"] < history[0]["lr"]  # cosine schedule stepped per epoch

    # resume for one more epoch
    run("train.py", "--train-root", synthetic_features / "feats", "--train-list", synthetic_features / "list.txt",
        "--epochs", 3, "--batch-size", 2, "--out", out, "--resume", out / "checkpoint.pt", "--device", "cpu")
    assert len(json.loads((out / "history.json").read_text())) == 1  # new run writes its own history

    log = run("evaluate.py", "--root", synthetic_features / "feats", "--list", synthetic_features / "list.txt",
              "--weights", out / "last.pth", "--json", out / "metrics.json", "--device", "cpu")
    assert "[Evaluation] n=8" in log
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics) >= {"accuracy", "pr_auc", "roc_auc"}


def test_scripts_help():
    for script in ["train.py", "evaluate.py", "train_e2e.py", "extract_features.py"]:
        assert "usage:" in run(script, "--help")
