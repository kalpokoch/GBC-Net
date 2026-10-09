#!/usr/bin/env python
"""Audit a binary split manifest for train_val/test leakage.

Checks
  1. the same relative_path in both splits                       (error)
  2. byte-identical files across splits (SHA-256)                 (error)
  3. near-duplicate images across splits (64-bit average hash,
     Hamming distance <= --max-hamming)                           (warning; error with --strict)
  4. source folders shared by test and train_val                  (informational)

Note: there is no patient ID in this dataset, so an image-level split cannot
rule out different slices of the same patient landing in different splits;
check 3 catches only visually near-identical slices.

    python scripts/audit_split_integrity.py --manifest data/manifests/binary_split_manifest.csv \
        --image-dir data/dataset_masked
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def average_hash(path: Path, size: int = 8) -> int:
    img = np.asarray(Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    bits = (img > img.mean()).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def main():
    parser = argparse.ArgumentParser(description="Audit train_val/test leakage")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--max-hamming", type=int, default=4)
    parser.add_argument("--strict", action="store_true", help="Treat near-duplicates as errors")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    image_dir = Path(args.image_dir)
    status = 0

    # 1. path overlap
    overlap = set(df.loc[df.split == "test", "relative_path"]) & set(df.loc[df.split == "train_val", "relative_path"])
    if overlap:
        print(f"[ERROR] {len(overlap)} paths appear in both splits, e.g. {sorted(overlap)[:3]}")
        status = 1
    else:
        print("[OK] no relative_path appears in both splits")

    paths = {row.relative_path: (row.split, image_dir / row.relative_path) for row in df.itertuples()}
    missing = [p for p, (_, full) in paths.items() if not full.exists()]
    if missing:
        print(f"[ERROR] {len(missing)} manifest images missing on disk, e.g. {missing[:3]}")
        raise SystemExit(1)

    # 2. exact duplicates
    by_hash = defaultdict(list)
    for rel, (split, full) in paths.items():
        by_hash[sha256(full)].append((split, rel))
    exact = [items for items in by_hash.values() if len({s for s, _ in items}) > 1]
    if exact:
        print(f"[ERROR] {len(exact)} byte-identical files span both splits:")
        for items in exact[:10]:
            print("   ", items)
        status = 1
    else:
        print("[OK] no byte-identical files across splits")

    # 3. near duplicates (test vs train_val)
    hashes = {rel: average_hash(full) for rel, (_, full) in paths.items()}
    test = [r for r, (s, _) in paths.items() if s == "test"]
    train = [r for r, (s, _) in paths.items() if s == "train_val"]
    train_h = np.array([hashes[r] for r in train], dtype=np.uint64)
    near = []
    for r in test:
        dist = np.array([bin(int(x)).count("1") for x in np.bitwise_xor(train_h, np.uint64(hashes[r]))])
        for j in np.where(dist <= args.max_hamming)[0]:
            near.append((r, train[j], int(dist[j])))
    if near:
        level = "ERROR" if args.strict else "WARN"
        print(f"[{level}] {len(near)} near-duplicate test/train_val pairs (aHash Hamming <= {args.max_hamming}):")
        for pair in near[:15]:
            print("   ", pair)
        status |= int(args.strict)
    else:
        print(f"[OK] no near-duplicate test/train_val pairs (aHash Hamming <= {args.max_hamming})")

    # 4. folder overlap
    folders = df.assign(folder=df.relative_path.apply(os.path.dirname)).groupby("folder")["split"].agg(set)
    shared = folders[folders.apply(len) > 1]
    print(f"[INFO] {len(shared)} of {len(folders)} source folders contribute to both splits")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
