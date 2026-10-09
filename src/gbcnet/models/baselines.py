"""Plain torchvision baselines adapted to 1-channel input and a single logit."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .convnext import grayscale_conv


def make_densenet121(pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    m.features.conv0 = grayscale_conv(m.features.conv0, pretrained)
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier.in_features, 1))
    return m


def make_resnet50(pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    m.conv1 = grayscale_conv(m.conv1, pretrained)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
    return m


def make_efficientnet_b0(pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    m.features[0][0] = grayscale_conv(m.features[0][0], pretrained)
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, 1))
    return m
