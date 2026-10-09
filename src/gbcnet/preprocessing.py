"""Body masking: produce `dataset_masked/` from the raw CT slice folders.

Each slice is thresholded, cleaned with morphological close/open, reduced to
its largest external contour (filled), and everything outside that body mask
is set to black. The output mirrors the source folder structure under a
class-named root (`cancer/...`, `non-cancer/...`) with filenames unchanged.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def build_file_mapping(class_dirs: Dict[str, Path]) -> List[Dict[str, str]]:
    """List every image under each class directory.

    Returns entries with the original filename, the relative output path
    (`<class>/<path within class dir>`) and the class name. Non-image files
    (e.g. .xlsx reports mixed into the case folders) are skipped.
    """
    mapping = []
    for class_name, class_dir in class_dirs.items():
        class_dir = Path(class_dir)
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        for img_path in sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            mapping.append({
                "filename": img_path.name,
                "relative_path": (Path(class_name) / img_path.relative_to(class_dir)).as_posix(),
                "class": class_name,
            })
    return mapping


def write_mapping_csv(mapping: List[Dict[str, str]], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "relative_path", "class"])
        writer.writeheader()
        writer.writerows(mapping)


def source_path(entry: Dict[str, str], class_dirs: Dict[str, Path]) -> Path:
    """Original location of a mapping entry under its class directory."""
    return Path(class_dirs[entry["class"]]) / Path(entry["relative_path"]).relative_to(entry["class"])


def create_body_mask(
    image: np.ndarray, intensity_threshold: int = 20, morph_kernel_size: int = 15, fill_holes: bool = True
) -> np.ndarray:
    """Binary body mask (255 = body incl. skin/fat, 0 = background) for a grayscale slice."""
    _, binary = cv2.threshold(image, intensity_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)  # close small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # remove small noise
    if fill_holes:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
            mask = filled
    return mask


def apply_body_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Black out everything outside the mask."""
    return cv2.bitwise_and(image, image, mask=mask)


def mask_dataset(
    mapping: List[Dict[str, str]],
    class_dirs: Dict[str, Path],
    output_dir: str | Path,
    intensity_threshold: int = 20,
    morph_kernel_size: int = 15,
    fill_holes: bool = True,
    log=print,
) -> Dict[str, Dict]:
    """Mask every mapped image and write it to `output_dir/<relative_path>`."""
    output_dir = Path(output_dir)
    stats: Dict[str, Dict] = {}
    for class_name in sorted({e["class"] for e in mapping}):
        entries = [e for e in mapping if e["class"] == class_name]
        s = {"processed": 0, "failed": 0, "body_coverage": []}
        log(f"\nProcessing {class_name}: {len(entries)} images")
        for entry in entries:
            img = cv2.imread(str(source_path(entry, class_dirs)), cv2.IMREAD_GRAYSCALE)
            if img is None:
                log(f"  failed to read: {entry['relative_path']}")
                s["failed"] += 1
                continue
            mask = create_body_mask(img, intensity_threshold, morph_kernel_size, fill_holes)
            out_path = output_dir / entry["relative_path"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out_path), apply_body_mask(img, mask)):
                log(f"  failed to write: {entry['relative_path']}")
                s["failed"] += 1
                continue
            s["body_coverage"].append(float((mask > 0).mean()))
            s["processed"] += 1
        coverage = s.pop("body_coverage")
        s["mean_body_coverage"] = float(np.mean(coverage)) if coverage else None
        log(f"  processed={s['processed']} failed={s['failed']} "
            + (f"mean body coverage={100 * s['mean_body_coverage']:.1f}%" if coverage else ""))
        stats[class_name] = s
    return stats


def masked_intensity_report(masked_dir: str | Path, n_samples: int = 50, log=print) -> Dict[str, float]:
    """Compare cancer vs. non-cancer mean gray level, whole image and body-only (pixels > 0).

    A large body-only gap means the classes differ in intensity for reasons
    other than pathology (acquisition/windowing), a potential shortcut.
    """
    masked_dir = Path(masked_dir)
    result = {}
    for class_name in ("cancer", "non-cancer"):
        files = sorted(p for p in (masked_dir / class_name).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)[:n_samples]
        overall, body = [], []
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            overall.append(img.mean())
            nonzero = img[img > 0]
            if nonzero.size:
                body.append(nonzero.mean())
        result[class_name] = {"n": len(overall), "overall_mean": float(np.mean(overall)), "overall_std": float(np.std(overall)),
                              "body_mean": float(np.mean(body)), "body_std": float(np.std(body))}
    c, n = result["cancer"], result["non-cancer"]
    result["overall_difference"] = abs(c["overall_mean"] - n["overall_mean"])
    result["body_difference"] = abs(c["body_mean"] - n["body_mean"])
    log(f"\nIntensity check ({c['n']} cancer, {n['n']} normal; 8-bit gray levels)")
    log(f"  whole image : cancer {c['overall_mean']:.2f} +/- {c['overall_std']:.2f} | normal {n['overall_mean']:.2f} "
        f"+/- {n['overall_std']:.2f} | diff {result['overall_difference']:.2f}")
    log(f"  body only   : cancer {c['body_mean']:.2f} +/- {c['body_std']:.2f} | normal {n['body_mean']:.2f} "
        f"+/- {n['body_std']:.2f} | diff {result['body_difference']:.2f} "
        f"({'PASS' if result['body_difference'] < 1 else 'FAIL'} vs. target < 1)")
    return result
