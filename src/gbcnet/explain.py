"""Grad-CAM++ heatmaps for a binary checkpoint on test images.

    python -m gbcnet.explain --run-dir outputs/binary_convnext_cbam_msam --fold 1 --n 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .config import load_yaml  # noqa: E402
from .data.binary import build_binary_transforms, dataset_from_cfg, load_binary_manifest  # noqa: E402
from .models.registry import build_model, get_spec  # noqa: E402
from .utils import ensure_dir, get_device  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM++ for a binary GBC checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n", type=int, default=16, help="Number of test images (balanced across classes)")
    parser.add_argument("--outdir", default=None, help="Default: <run-dir>/gradcam_fold<k>")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "config_resolved.yaml")
    model_name = cfg["model"]["name"]
    outdir = ensure_dir(args.outdir or run_dir / f"gradcam_fold{args.fold}")
    device = get_device()

    model = build_model(model_name, pretrained=False, dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load(run_dir / "trained_models" / f"{model_name}_fold{args.fold}_best.pth", map_location=device))
    model.eval()

    _, test_df = load_binary_manifest(cfg["data"]["manifest"], cfg["data"]["image_dir"])
    rng = np.random.default_rng(args.seed)
    per_class = max(args.n // 2, 1)
    picked = np.concatenate([
        rng.choice(np.where(test_df["label"].values == c)[0], size=min(per_class, int((test_df["label"] == c).sum())), replace=False)
        for c in (1, 0)
    ])
    _, eval_tf = build_binary_transforms(cfg["transforms"]["image_size"], cfg["transforms"]["center_crop"])
    ds = dataset_from_cfg(test_df.iloc[picked].reset_index(drop=True), eval_tf, cfg, cache=False)

    cam = GradCAMPlusPlus(model=model, target_layers=[get_spec(model_name).cam_layer(model)])
    mean, std = cfg["data"]["global_mean"], cfg["data"]["global_std"]
    cols = 4
    rows = int(np.ceil(len(ds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(len(ds)):
        x, label, _ = ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0]
        img = np.clip(x[0, 0].cpu().numpy() * std + mean, 0, 1)
        ax = axes.ravel()[i]
        ax.imshow(img, cmap="gray")
        ax.imshow(heat, cmap="jet", alpha=0.4)
        ax.set_title(f"{'Cancer' if label.item() == 1 else 'Normal'} | p(cancer)={prob:.2f}", fontsize=10)
    plt.tight_layout()
    out = outdir / "gradcam_plus_plus_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
