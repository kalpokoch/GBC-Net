"""Binary (cancer vs. normal) dataset, transforms and loaders.

Images are read as single-channel grayscale (`dataset_masked/` PNG/JPG),
transformed with albumentations, scaled to [0, 1] and standardized with a
global mean/std.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ALBUMENTATIONS_V2 = int(A.__version__.split(".")[0]) >= 2


class BinaryCTDataset(Dataset):
    """Returns (image[1,H,W] float tensor, label float tensor, filepath)."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
        global_mean: float = 0.456,
        global_std: float = 0.224,
        cache_images: bool = True,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.global_mean = global_mean
        self.global_std = global_std
        self.image_cache = [self._read(fp) for fp in self.dataframe["filepath"]] if cache_images else None

    @staticmethod
    def _read(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        label = self.dataframe.loc[idx, "label"]
        img_path = self.dataframe.loc[idx, "filepath"]
        image = self.image_cache[idx] if self.image_cache is not None else self._read(img_path)
        if self.transform:
            image = self.transform(image=image)["image"]
        image = image.astype(np.float32) / 255.0
        image = (image - self.global_mean) / self.global_std
        image = torch.from_numpy(image).unsqueeze(0)
        return image, torch.tensor(label, dtype=torch.float32), str(img_path)


def _gauss_noise(p: float):
    # var_limit=(10, 50) in uint8 pixel units -> std 3.2-7.1 px.
    if ALBUMENTATIONS_V2:
        return A.GaussNoise(std_range=(np.sqrt(10.0) / 255.0, np.sqrt(50.0) / 255.0), p=p)
    return A.GaussNoise(var_limit=(10.0, 50.0), p=p)


def _elastic(p: float):
    # alpha_affine was removed in albumentations 1.4+/2.x; it is ignored there.
    if ALBUMENTATIONS_V2:
        return A.ElasticTransform(alpha=1, sigma=50, p=p)
    return A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=p)


def _coarse_dropout(p: float):
    if ALBUMENTATIONS_V2:
        return A.CoarseDropout(num_holes_range=(3, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=p)
    return A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=3, min_height=16, min_width=16, p=p)


def build_binary_transforms(image_size: int = 512, center_crop: int = 350) -> Tuple[A.Compose, A.Compose]:
    """Train and val/test transforms for the binary task.

    Several transforms changed argument names in albumentations 2.x (1.x-style
    arguments are silently ignored there), so they are built version-aware.
    """
    train_transform = A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.8),
            A.CenterCrop(height=center_crop, width=center_crop, p=1.0),
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=20, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            _gauss_noise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            _elastic(p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            _coarse_dropout(p=0.2),
        ]
    )
    eval_transform = A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.CenterCrop(height=center_crop, width=center_crop),
            A.Resize(height=image_size, width=image_size),
        ]
    )
    return train_transform, eval_transform


def load_binary_manifest(manifest_path: str | Path, image_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read `relative_path,label,split` and return (train_val_df, test_df)."""
    df = pd.read_csv(manifest_path)
    missing_cols = {"relative_path", "label", "split"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest {manifest_path} is missing columns: {sorted(missing_cols)}")
    image_dir = Path(image_dir)
    df["filepath"] = df["relative_path"].apply(lambda p: str(image_dir / p))
    df["class"] = df["label"].map({1: "Cancer", 0: "Normal"})
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_val_df = df[df["split"] == "train_val"].reset_index(drop=True)
    if len(test_df) + len(train_val_df) != len(df):
        raise ValueError("Manifest split column must contain only 'train_val' and 'test'")
    return train_val_df, test_df


def balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Inverse-class-frequency WeightedRandomSampler (with replacement)."""
    labels = np.asarray(labels).astype(int)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    sampler=None,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    kwargs: Dict = {}
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        **kwargs,
    )


def dataset_from_cfg(df: pd.DataFrame, transform, cfg: Dict, cache: Optional[bool] = None) -> BinaryCTDataset:
    data_cfg = cfg["data"]
    return BinaryCTDataset(
        df,
        transform=transform,
        global_mean=data_cfg.get("global_mean", 0.456),
        global_std=data_cfg.get("global_std", 0.224),
        cache_images=data_cfg.get("cache_images", True) if cache is None else cache,
    )
