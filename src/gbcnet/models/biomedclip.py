"""BiomedCLIP vision encoder + MLP head for multi-label attribute prediction."""

from __future__ import annotations

import gc

import torch
import torch.nn as nn


def resample_encoder_pos_embed(encoder: nn.Module, image_size: int, patch: int = 16) -> None:
    """Resample the ViT absolute position embedding to a new input size (e.g. 336)."""
    import timm

    trunk = getattr(encoder, "trunk", None)
    if trunk is None or not hasattr(trunk, "pos_embed"):
        raise RuntimeError("Expected encoder.trunk.pos_embed; cannot resample.")
    grid = image_size // patch
    trunk.patch_embed.img_size = (image_size, image_size)
    trunk.patch_embed.grid_size = (grid, grid)
    trunk.patch_embed.num_patches = grid * grid
    trunk.patch_embed.strict_img_size = False
    new_pe = timm.layers.resample_abs_pos_embed(
        trunk.pos_embed.data, [grid, grid], num_prefix_tokens=trunk.num_prefix_tokens
    )
    trunk.pos_embed = nn.Parameter(new_pe)


class BiomedCLIPClassifier(nn.Module):
    """BiomedCLIP ViT-B/16 encoder; dual-scale mode encodes full slice and RUQ crop
    (stacked as 6 channels) with the shared encoder and concatenates features."""

    def __init__(self, model_name: str, num_classes: int, hidden_dim: int, dropout: float, image_size: int, dual_scale: bool):
        super().__init__()
        import open_clip

        self.dual_scale = dual_scale
        base_model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.vision_encoder = base_model.visual
        del base_model
        gc.collect()

        if image_size != 224:
            resample_encoder_pos_embed(self.vision_encoder, image_size)

        with torch.no_grad():
            feat_dim = self._encode(torch.zeros(1, 3, image_size, image_size)).shape[-1]

        head_in = feat_dim * (2 if dual_scale else 1)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.vision_encoder(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dual_scale:
            feat = torch.cat([self._encode(x[:, :3]), self._encode(x[:, 3:])], dim=1)
        else:
            feat = self._encode(x)
        return self.classifier(feat)


def encoder_blocks(model: BiomedCLIPClassifier):
    encoder = model.vision_encoder
    if hasattr(encoder, "trunk") and hasattr(encoder.trunk, "blocks"):
        return encoder.trunk.blocks
    raise RuntimeError("Cannot locate transformer blocks in vision_encoder.")


def set_encoder_trainable(model: BiomedCLIPClassifier, mode: str, unfreeze_last_n: int = 0) -> int:
    """mode='frozen': head only. mode='partial': also the last N encoder blocks.
    Returns the number of unfrozen encoder blocks."""
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    n_unfrozen = 0
    if mode == "partial":
        blocks = encoder_blocks(model)
        n_unfrozen = min(unfreeze_last_n, len(blocks))
        for i in range(len(blocks) - n_unfrozen, len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    return n_unfrozen
