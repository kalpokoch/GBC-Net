"""CBAM and multi-scale spatial attention (MSAM)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module from CBAM (Woo et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Single-scale spatial attention module from CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class MultiScaleAttention(nn.Module):
    """MSAM: spatial attention at several dilated receptive fields.

    Dilations 1/2/4 on a shared kernel_size=7 give 7x7, 13x13 and 25x25
    receptive fields, computed from the same avg+max channel-pooled summary
    CBAM's spatial branch uses and fused by a learned 1x1 conv.
    `last_gate` keeps the most recent attention map for inspection.
    """

    def __init__(self, kernel_size: int = 7, dilations=(1, 2, 4)):
        super().__init__()
        self.scale_convs = nn.ModuleList(
            [
                nn.Conv2d(2, 1, kernel_size, padding=((kernel_size - 1) * d) // 2, dilation=d, bias=False)
                for d in dilations
            ]
        )
        self.fuse = nn.Conv2d(len(dilations), 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.last_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        scale_maps = [conv(pooled) for conv in self.scale_convs]
        gate = self.sigmoid(self.fuse(torch.cat(scale_maps, dim=1)))
        self.last_gate = gate
        return x * gate
