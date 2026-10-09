"""Losses and MixUp."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on logits (alpha weights every element; no pos_weight)."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MaskedAsymmetricLossMultiLabel(nn.Module):
    """Asymmetric loss (Ben-Baruch et al., 2021) with an element mask.

    Masked elements are excluded from both the sum and the normalizing
    denominator instead of being scored as negatives.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, targets = logits.float(), targets.float()
        probs_pos = torch.sigmoid(logits)
        probs_neg = 1 - probs_pos
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        focus_pos = torch.pow(1 - probs_pos * targets, self.gamma_pos)
        focus_neg = torch.pow(1 - probs_neg * (1 - targets), self.gamma_neg)
        loss = loss_pos * focus_pos + loss_neg * focus_neg  # (B, C)

        if mask is None:
            return -loss.sum(dim=1).mean()
        mask = mask.float()
        return -(loss * mask).sum() / mask.sum().clamp(min=1.0)
