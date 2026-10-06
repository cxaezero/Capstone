"""Model construction. All factories load weights explicitly and fail if a file is missing."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .. import config
from .classifier import AnomalyClassifier, Model
from .esdnet import ESDNet
from .weights import load_state_dict_file
from .x3d import X3D_FEATURE_DIM, build_x3d

__all__ = [
    "ESDNet", "Model", "AnomalyClassifier", "X3D_FEATURE_DIM", "ModelBundle",
    "build_deweather", "build_classifier", "build_x3d", "load_models", "load_state_dict_file",
]


def build_deweather(weights=config.DEWEATHER_WEIGHTS, device="cpu", eval_mode: bool = True) -> ESDNet:
    """ESDNet with the shipped config. ``weights=None`` builds an untrained network."""
    model = ESDNet(**config.ESDNET_KWARGS)
    if weights is not None:
        model.load_state_dict(load_state_dict_file(weights))
    model = model.to(device)
    return model.eval() if eval_mode else model.train()


def build_classifier(weights=config.CLASSIFIER_WEIGHTS, device="cpu", eval_mode: bool = True) -> Model:
    """Anomaly classifier with the shipped config. ``weights=None`` builds an untrained network."""
    model = Model(**config.CLASSIFIER_KWARGS)
    if weights is not None:
        model.load_state_dict(load_state_dict_file(weights))
    model = model.to(device)
    return model.eval() if eval_mode else model.train()


@dataclass
class ModelBundle:
    deweather: torch.nn.Module
    feature_extractor: torch.nn.Module
    classifier: torch.nn.Module


def load_models(device="cpu", variant: str = config.X3D_VARIANT) -> ModelBundle:
    """The three inference models used by the demo, in eval mode on ``device``."""
    return ModelBundle(
        deweather=build_deweather(device=device),
        feature_extractor=build_x3d(variant, device=device),
        classifier=build_classifier(device=device),
    )
