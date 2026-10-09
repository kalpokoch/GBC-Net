"""Re-evaluate a finished train_binary run: per-fold metrics and the fold ensemble.

Uses the run's saved checkpoints and a fixed 0.5 decision threshold. Point
--manifest at another manifest to score an external test set (its `test`
rows are used).

    python -m gbcnet.evaluate_binary --run-dir outputs/binary_convnext_cbam_msam
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import plots
from .config import apply_overrides, load_yaml
from .data.binary import build_binary_transforms, dataset_from_cfg, load_binary_manifest, make_loader
from .engine import predict_probs
from .metrics import DECISION_THRESHOLD, binary_metrics, bootstrap_binary_ci
from .models.registry import build_model
from .utils import ensure_dir, get_device, save_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate a train_binary run on a test set")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", default=None, help="Override the manifest (uses its 'test' rows)")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--outdir", default=None, help="Default: <run-dir>/evaluation")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = apply_overrides(load_yaml(run_dir / "config_resolved.yaml"), args.set)
    if args.manifest:
        cfg["data"]["manifest"] = args.manifest
    if args.image_dir:
        cfg["data"]["image_dir"] = args.image_dir
    outdir = ensure_dir(args.outdir or run_dir / "evaluation")
    device = get_device()
    model_name = cfg["model"]["name"]

    _, test_df = load_binary_manifest(cfg["data"]["manifest"], cfg["data"]["image_dir"])
    _, eval_tf = build_binary_transforms(cfg["transforms"]["image_size"], cfg["transforms"]["center_crop"])
    loader = make_loader(dataset_from_cfg(test_df, eval_tf, cfg), args.batch_size, cfg["data"]["num_workers"])

    ckpts = sorted((run_dir / "trained_models").glob(f"{model_name}_fold*_best.pth"))
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {run_dir / 'trained_models'}")

    rows, fold_probs, targets = [], [], None
    for ckpt in ckpts:
        fold = int(ckpt.stem.split("_fold")[1].split("_")[0])
        model = build_model(model_name, pretrained=False, dropout=cfg["model"]["dropout"]).to(device, memory_format=torch.channels_last)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        targets, probs, _ = predict_probs(model, loader, device, cfg["train"])
        fold_probs.append(probs)
        m = binary_metrics(targets, probs, DECISION_THRESHOLD)
        rows.append({"fold": fold, **{k: v for k, v in m.items() if k not in ("predictions", "confusion_matrix")}})
        print(f"fold {fold}: AUC={m['auc']:.4f} F1={m['f1']:.4f}")

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(outdir / "per_fold_metrics.csv", index=False)

    ensemble_probs = np.mean(fold_probs, axis=0)
    ens = binary_metrics(targets, ensemble_probs, DECISION_THRESHOLD)
    ci = bootstrap_binary_ci(targets, ensemble_probs, ens["predictions"], args.bootstrap) if args.bootstrap else {}
    result = {
        "model_name": model_name, "n_folds": len(ckpts), "n_test_images": len(test_df),
        "ensemble": {k: v for k, v in ens.items() if k != "predictions"}, "ensemble_bootstrap_ci_95": ci,
    }
    save_json(result, outdir / "ensemble_metrics.json")
    pd.DataFrame({"relative_path": test_df["relative_path"], "target": targets, "ensemble_prob": ensemble_probs,
                  **{f"prob_fold{r['fold']}": p for r, p in zip(rows, fold_probs)}}).to_csv(outdir / "predictions.csv", index=False)
    plots.plot_confusion_matrix(ens["confusion_matrix"], f"{model_name} - fold ensemble (threshold {DECISION_THRESHOLD})",
                                outdir / "ensemble_confusion_matrix.png")
    plots.plot_roc(targets, ensemble_probs, f"{model_name} - fold ensemble ROC", outdir / "ensemble_roc_curve.png")

    print(f"\nEnsemble of {len(ckpts)} folds: AUC={ens['auc']:.4f} F1={ens['f1']:.4f} "
          f"Sens={ens['sensitivity']:.4f} Spec={ens['specificity']:.4f}")
    for k, v in ci.items():
        print(f"  95% CI {k:<12} {v['lower']:.4f} - {v['upper']:.4f}")
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
