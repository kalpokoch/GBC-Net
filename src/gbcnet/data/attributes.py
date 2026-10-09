"""Multi-label attribute dataset (cancer images only), transforms and label loading."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PercentileNormalize:
    """Rescale the foreground (pixels > 0) intensity range [p_low, p_high] to [0, 255]."""

    def __init__(self, low_pct: float = 1.0, high_pct: float = 99.0):
        self.low_pct, self.high_pct = low_pct, high_pct

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img).astype(np.float32)
        gray = arr.mean(axis=2) if arr.ndim == 3 else arr
        fg = gray[gray > 0]
        if fg.size == 0:
            return img
        lo, hi = np.percentile(fg, [self.low_pct, self.high_pct])
        if hi <= lo:
            return img
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
        return Image.fromarray(arr.astype(np.uint8))


def body_bbox(img: Image.Image, thresh: int = 5):
    gray = np.asarray(img.convert("L"))
    mask = gray > thresh
    if not mask.any():
        return (0, 0, img.width, img.height)
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def ruq_crop(img: Image.Image, ruq_frac: Sequence[float]) -> Image.Image:
    """Right-upper-quadrant crop relative to the body bounding box."""
    x0, y0, x1, y1 = body_bbox(img)
    w, h = x1 - x0, y1 - y0
    fx0, fx1, fy0, fy1 = ruq_frac
    return img.crop((x0 + int(fx0 * w), y0 + int(fy0 * h), x0 + int(fx1 * w), y0 + int(fy1 * h)))


def _pre(tcfg: Dict) -> List:
    if tcfg.get("use_intensity_norm", True):
        return [PercentileNormalize(tcfg["intensity_pct_low"], tcfg["intensity_pct_high"])]
    return []


def _to_tensor_norm(tcfg: Dict) -> List:
    return [
        T.Resize((tcfg["image_size"], tcfg["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=tcfg["clip_mean"], std=tcfg["clip_std"]),
    ]


def build_attribute_transforms(tcfg: Dict, split: str) -> T.Compose:
    if split == "train":
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
        return T.Compose(_pre(tcfg) + aug + _to_tensor_norm(tcfg))
    return T.Compose(_pre(tcfg) + _to_tensor_norm(tcfg))


def build_tta_transforms(tcfg: Dict) -> List[T.Compose]:
    base = T.Compose(_pre(tcfg) + _to_tensor_norm(tcfg))
    if not tcfg.get("use_tta", True):
        return [base]
    flipped = T.Compose(
        _pre(tcfg)
        + [
            T.Resize((tcfg["image_size"], tcfg["image_size"])),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=tcfg["clip_mean"], std=tcfg["clip_std"]),
        ]
    )
    return [base, flipped]


class AttributeCTDataset(Dataset):
    """Returns (x, labels, mask, relative_path).

    Dual-scale mode stacks the transformed full slice and RUQ crop as 6
    channels. The mask is row-level (`label_valid` broadcast to every label).
    """

    def __init__(self, df: pd.DataFrame, image_dir: str | Path, label_cols: Sequence[str], transform, tcfg: Dict):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.label_cols = list(label_cols)
        self.transform = transform
        self.dual_scale = tcfg.get("use_dual_scale", False)
        self.ruq_frac = tcfg.get("ruq_frac")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = row["relative_path"]
        img = Image.open(self.image_dir / rel_path).convert("RGB")
        if self.dual_scale:
            x = torch.cat([self.transform(img), self.transform(ruq_crop(img, self.ruq_frac))], dim=0)
        else:
            x = self.transform(img)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32), dtype=torch.float32)
        mask = torch.full((len(self.label_cols),), float(row["label_valid"]), dtype=torch.float32)
        return x, labels, mask, str(rel_path)


def make_attribute_loader(cfg: Dict, df: pd.DataFrame, transform, shuffle: bool = False, sampler=None) -> DataLoader:
    ds = AttributeCTDataset(df, cfg["data"]["image_dir"], cfg["data"]["label_cols"], transform, cfg["transforms"])
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def load_attribute_labels(cfg: Dict, log=print) -> pd.DataFrame:
    """Load the attribute CSV, drop rows with missing images and add `label_valid`.

    Rows with zero positives across all attributes are report-parsing
    failures, not true negatives: they are kept but masked (label_valid=False).
    """
    data_cfg = cfg["data"]
    label_cols = data_cfg["label_cols"]
    df = pd.read_csv(data_cfg["labels_csv"])
    log(f"Raw rows  : {len(df)}")
    for col in label_cols:
        df[col] = df[col].astype(np.float32)

    image_dir = Path(data_cfg["image_dir"])
    exists = df["relative_path"].apply(lambda p: (image_dir / p).exists())
    if (~exists).any():
        log(f"WARNING: {int((~exists).sum())} image(s) not found -- removing from dataset")
        df = df[exists].reset_index(drop=True)
    log(f"Valid rows: {len(df)}")

    all_attrs = label_cols + (["capsular_retraction"] if "capsular_retraction" in df.columns else [])
    unparsed = df[all_attrs].astype(float).sum(axis=1) == 0
    log(f"Extraction failures (0 positives across {len(all_attrs)} attributes): "
        f"{int(unparsed.sum())} / {len(df)} ({100 * unparsed.mean():.1f}%)")
    df["label_valid"] = (~unparsed).astype(bool) if data_cfg.get("mask_unparsed_rows", True) else True
    return df.reset_index(drop=True)
