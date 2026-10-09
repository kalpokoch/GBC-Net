"""Model registry for the binary task.

Each entry gives a factory plus functions returning the sub-modules that
belong to each optimizer group (backbone / attention / classifier) and the
layer used for Grad-CAM. One generic training loop handles every model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch.nn as nn

from .baselines import make_densenet121, make_efficientnet_b0, make_resnet50
from .convnext import ConvNeXtTinyCBAM, ConvNeXtTinyCBAMMSAM, ConvNeXtTinyPlain

Modules = Callable[[nn.Module], List[nn.Module]]


@dataclass(frozen=True)
class ModelSpec:
    factory: Callable[..., nn.Module]
    backbone: Modules
    attention: Modules
    classifier: Modules
    cam_layer: Callable[[nn.Module], nn.Module]


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "ConvNeXtTiny_CBAM_MSAM": ModelSpec(
        factory=lambda pretrained=True, dropout=0.5: ConvNeXtTinyCBAMMSAM(1, pretrained, dropout),
        backbone=lambda m: [m.features],
        attention=lambda m: [m.cbam, m.msam],
        classifier=lambda m: [m.classifier],
        cam_layer=lambda m: m.features[-1],
    ),
    "ConvNeXtTiny_CBAM": ModelSpec(
        factory=lambda pretrained=True, dropout=0.5: ConvNeXtTinyCBAM(1, pretrained, dropout),
        backbone=lambda m: [m.features],
        attention=lambda m: [m.cbam],
        classifier=lambda m: [m.classifier],
        cam_layer=lambda m: m.features[-1],
    ),
    "ConvNeXtTiny_Plain": ModelSpec(
        factory=lambda pretrained=True, dropout=0.5: ConvNeXtTinyPlain(1, pretrained, dropout),
        backbone=lambda m: [m.features],
        attention=lambda m: [],
        classifier=lambda m: [m.classifier],
        cam_layer=lambda m: m.features[-1],
    ),
    "DenseNet121": ModelSpec(
        factory=make_densenet121,
        backbone=lambda m: [m.features],
        attention=lambda m: [],
        classifier=lambda m: [m.classifier],
        cam_layer=lambda m: m.features[-1],
    ),
    "ResNet50": ModelSpec(
        factory=make_resnet50,
        backbone=lambda m: [m.conv1, m.bn1, m.layer1, m.layer2, m.layer3, m.layer4],
        attention=lambda m: [],
        classifier=lambda m: [m.fc],
        cam_layer=lambda m: m.layer4[-1],
    ),
    "EfficientNet_B0": ModelSpec(
        factory=make_efficientnet_b0,
        backbone=lambda m: [m.features],
        attention=lambda m: [],
        classifier=lambda m: [m.classifier],
        cam_layer=lambda m: m.features[-1],
    ),
}


def get_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def build_model(name: str, pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    return get_spec(name).factory(pretrained=pretrained, dropout=dropout)
