#!/usr/bin/env python
"""Build the binary split manifest (relative_path,label,split) from the image folders.

Labels come from the top-level folder (`cancer/` -> 1, `non-cancer/` -> 0).

Reproduce the split used for the reported results (fixed 100-image test set
saved by the 5-fold notebook) -- every other image becomes train_val:

    python scripts/prepare_binary_split.py --image-dir data/dataset_masked \
        --pinned-test path/to/Output_ConvNeXtTiny_CBAM_MSAM_5Fold/results/test_set.csv \
        --out data/manifests/binary_split_manifest.csv

Or draw a new stratified test set (N images per class):

    python scripts/prepare_binary_split.py --image-dir data/dataset_masked \
        --test-per-class 50 --seed 42 --out data/manifests/binary_split_manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def scan(image_dir: Path, class_dirs: dict) -> pd.DataFrame:
    rows = []
    for folder, label in class_dirs.items():
        root = image_dir / folder
        if not root.is_dir():
            sys.exit(f"Missing class folder: {root}")
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rows.append({"relative_path": p.relative_to(image_dir).as_posix(), "label": label})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Create the binary train_val/test manifest")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cancer-dir", default="cancer")
    parser.add_argument("--normal-dir", default="non-cancer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pinned-test", help="CSV with a relative_path column listing the test images")
    group.add_argument("--test-per-class", type=int, help="Random stratified test set size per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    df = scan(image_dir, {args.cancer_dir: 1, args.normal_dir: 0})
    print(f"Found {len(df)} images: cancer={int((df.label == 1).sum())} normal={int((df.label == 0).sum())}")

    if args.pinned_test:
        pinned = pd.read_csv(args.pinned_test)["relative_path"].str.replace("\\", "/", regex=False)
        missing = sorted(set(pinned) - set(df.relative_path))
        if missing:
            print(f"ERROR: {len(missing)} pinned test images not found under {image_dir}, e.g. {missing[:3]}")
            sys.exit(1)
        test_paths = set(pinned)
    else:
        rng = np.random.default_rng(args.seed)
        test_paths = set()
        for label in (0, 1):
            pool = df.loc[df.label == label, "relative_path"].values
            test_paths |= set(rng.choice(pool, size=args.test_per_class, replace=False))

    df["split"] = np.where(df.relative_path.isin(test_paths), "test", "train_val")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby(["split", "label"]).size().rename("n").to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
