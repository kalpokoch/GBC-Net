"""ConvNeXt-Tiny variants for 1-channel grayscale CT, binary output."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .attention import CBAM, MultiScaleAttention

CONVNEXT_TINY_FEATURES = 768


def grayscale_conv(original: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    """Replace a 3-channel stem conv by a 1-channel one (RGB weights averaged)."""
    new_conv = nn.Conv2d(
        1,
        original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=original.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            new_conv.weight.data = original.weight.data.mean(dim=1, keepdim=True)
            if original.bias is not None:
                new_conv.bias.data = original.bias.data.clone()
    return new_conv


def convnext_tiny_grayscale_features(pretrained: bool) -> nn.Sequential:
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.convnext_tiny(weights=weights)
    backbone.features[0][0] = grayscale_conv(backbone.features[0][0], pretrained)
    return backbone.features  # (B, 768, 16, 16) at 512x512 input


class ConvNeXtTinyCBAMMSAM(nn.Module):
    """Proposed model: ConvNeXt-Tiny (pretrained) + CBAM + MSAM."""

    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.features = convnext_tiny_grayscale_features(pretrained)
        self.cbam = CBAM(channels=CONVNEXT_TINY_FEATURES, reduction=16, kernel_size=7)
        self.msam = MultiScaleAttention(kernel_size=7, dilations=(1, 2, 4))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(CONVNEXT_TINY_FEATURES, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        x = self.msam(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.classifier(x)


class ConvNeXtTinyCBAM(nn.Module):
    """Ablation midpoint: ConvNeXt-Tiny + CBAM, no MSAM."""

    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.features = convnext_tiny_grayscale_features(pretrained)
        self.cbam = CBAM(channels=CONVNEXT_TINY_FEATURES, reduction=16, kernel_size=7)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(CONVNEXT_TINY_FEATURES, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.classifier(x)


class ConvNeXtTinyPlain(nn.Module):
    """Plain ConvNeXt-Tiny, no attention."""

    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.features = convnext_tiny_grayscale_features(pretrained)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(CONVNEXT_TINY_FEATURES, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.classifier(x)
