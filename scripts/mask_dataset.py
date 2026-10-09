#!/usr/bin/env python
"""Create `dataset_masked/` (body-masked slices) and `mapping.csv` from the raw data.

Expected raw layout (folder names configurable):
    <data-dir>/GBCA/<contributor>/<batch>/.../*.jpg|png        -> cancer
    <data-dir>/NORMAL GALL BLADDER 851/*.png                   -> non-cancer

Output mirrors that structure under class-named roots with filenames unchanged:
    <out-dir>/cancer/<contributor>/<batch>/...
    <out-dir>/non-cancer/...

    python scripts/mask_dataset.py --data-dir data/raw --out-dir data/dataset_masked \
        --mapping-csv data/mapping.csv --preview 4

Defaults reproduce the masking used for the reported experiments
(threshold 20, 15 px elliptical kernel, largest contour filled).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gbcnet.preprocessing import (  # noqa: E402
    apply_body_mask,
    build_file_mapping,
    create_body_mask,
    mask_dataset,
    masked_intensity_report,
    source_path,
    write_mapping_csv,
)
from gbcnet.utils import save_json  # noqa: E402


def save_preview(entries, class_dirs, args, path: Path) -> None:
    """Original / mask / masked grid for a few images (replaces the notebook's interactive plots)."""
    fig, axes = plt.subplots(len(entries), 3, figsize=(12, 4 * len(entries)), squeeze=False)
    for row, entry in zip(axes, entries):
        img = cv2.imread(str(source_path(entry, class_dirs)), cv2.IMREAD_GRAYSCALE)
        mask = create_body_mask(img, args.intensity_threshold, args.morph_kernel_size, not args.no_fill_holes)
        for ax, im, title in zip(row, (img, mask, apply_body_mask(img, mask)), (entry["class"], "Mask", "Masked")):
            ax.imshow(im, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Body-mask the raw CT slices")
    parser.add_argument("--data-dir", required=True, help="Raw data root containing the class folders")
    parser.add_argument("--out-dir", required=True, help="Output root (dataset_masked)")
    parser.add_argument("--mapping-csv", default=None, help="Default: <out-dir>/../mapping.csv")
    parser.add_argument("--cancer-subdir", default="GBCA")
    parser.add_argument("--normal-subdir", default="NORMAL GALL BLADDER 851")
    parser.add_argument("--intensity-threshold", type=int, default=20, help="Body threshold (lower = more inclusive)")
    parser.add_argument("--morph-kernel-size", type=int, default=15, help="Elliptical kernel for close/open")
    parser.add_argument("--no-fill-holes", action="store_true", help="Keep internal holes instead of filling the largest contour")
    parser.add_argument("--preview", type=int, default=0, help="Save an original/mask/masked grid for N images per class and stop")
    parser.add_argument("--validate-samples", type=int, default=50, help="Images per class for the intensity check; 0 skips it")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    class_dirs = {"cancer": data_dir / args.cancer_subdir, "non-cancer": data_dir / args.normal_subdir}
    mapping = build_file_mapping(class_dirs)
    counts = {c: sum(e["class"] == c for e in mapping) for c in class_dirs}
    print(f"Found {len(mapping)} images: {counts}")

    if args.preview:
        rng = random.Random(args.seed)
        picks = [e for c in class_dirs for e in rng.sample([m for m in mapping if m["class"] == c], min(args.preview, counts[c]))]
        save_preview(picks, class_dirs, args, out_dir.parent / "mask_preview.png")
        return

    mapping_csv = Path(args.mapping_csv) if args.mapping_csv else out_dir.parent / "mapping.csv"
    write_mapping_csv(mapping, mapping_csv)
    print(f"Wrote {mapping_csv}")

    stats = mask_dataset(mapping, class_dirs, out_dir, args.intensity_threshold, args.morph_kernel_size, not args.no_fill_holes)
    report = masked_intensity_report(out_dir, args.validate_samples) if args.validate_samples else None
    save_json({"params": {k: v for k, v in vars(args).items() if k not in ("data_dir", "out_dir", "mapping_csv")},
               "stats": stats, "intensity_report": report}, out_dir.parent / "mask_stats.json")


if __name__ == "__main__":
    main()
