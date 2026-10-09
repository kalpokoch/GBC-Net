"""Predict cancer probability for one image or a folder with one or more binary checkpoints.

    python -m gbcnet.predict --image path/to/slice.png \
        --checkpoints outputs/binary_convnext_cbam_msam/trained_models/*_best.pth --threshold 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .data.binary import build_binary_transforms
from .models.registry import MODEL_REGISTRY, build_model
from .utils import get_device

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Binary GBC prediction (fold-ensemble averaging)")
    parser.add_argument("--image", required=True, help="Image file or folder")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--model", default="ConvNeXtTiny_CBAM_MSAM", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--center-crop", type=int, default=350)
    parser.add_argument("--global-mean", type=float, default=0.456)
    parser.add_argument("--global-std", type=float, default=0.224)
    args = parser.parse_args()

    device = get_device()
    src = Path(args.image)
    paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    _, eval_tf = build_binary_transforms(args.image_size, args.center_crop)

    models = []
    for ckpt in args.checkpoints:
        m = build_model(args.model, pretrained=False).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models.append(m)

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"{path}: unreadable, skipped")
            continue
        x = eval_tf(image=image)["image"].astype(np.float32) / 255.0
        x = torch.from_numpy((x - args.global_mean) / args.global_std)[None, None].to(device)
        with torch.no_grad():
            prob = float(np.mean([torch.sigmoid(m(x)).item() for m in models]))
        print(f"{path}\tp(cancer)={prob:.4f}\t{'Cancer' if prob >= args.threshold else 'Normal'}")


if __name__ == "__main__":
    main()
