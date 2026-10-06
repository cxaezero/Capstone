"""Metrics and evaluation loop for the anomaly classifier."""
from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def compute_metrics(predictions, targets, threshold: float = 0.5) -> dict:
    """Accuracy at ``threshold`` plus PR-AUC / ROC-AUC (NaN if only one class is present)."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    out = {"n": int(len(targets)), "accuracy": float(np.mean((predictions >= threshold) == (targets >= 0.5)))}
    if len(np.unique(targets)) < 2:
        out.update(pr_auc=math.nan, roc_auc=math.nan)
        return out
    fpr, tpr, _ = roc_curve(targets, predictions)
    precision, recall, _ = precision_recall_curve(targets, predictions)
    out.update(pr_auc=float(auc(recall, precision)), roc_auc=float(auc(fpr, tpr)))
    return out


def unpack_batch(batch, device):
    """Accept ``(n_in, n_lbl, a_in, a_lbl)`` pair batches or plain ``(inputs, labels)`` batches."""
    if len(batch) == 4:
        n_input, n_label, a_input, a_label = batch
        inputs = torch.cat((n_input, a_input), dim=0)
        labels = torch.cat((n_label, a_label), dim=0)
    elif len(batch) == 2:
        inputs, labels = batch
    else:
        raise ValueError(f"unexpected batch of length {len(batch)}")
    return inputs.to(device), labels.to(device).float()


@torch.no_grad()
def evaluate(loader, model, device, progress: bool = True) -> dict:
    """Run ``model`` in eval mode over ``loader`` and return ``compute_metrics`` on sigmoid scores."""
    from tqdm import tqdm

    model.eval()
    predictions, targets = [], []
    iterator = tqdm(loader, desc="Evaluating") if progress else loader
    for batch in iterator:
        if batch is None:
            continue
        inputs, labels = unpack_batch(batch, device)
        scores, _ = model(inputs)
        predictions.extend(torch.sigmoid(scores.reshape(-1)).cpu().tolist())
        targets.extend(labels.reshape(-1).cpu().tolist())
    return compute_metrics(predictions, targets)
