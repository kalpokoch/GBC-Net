"""Backbone comparison sweep and cross-model comparison figures.

Trains every model in `models:` with the shared recipe (via train_binary), then
compares them with the proposed model's saved run (`proposed_run_dir`), read
from its fold_predictions.csv -- no retraining or re-inference.

    python -m gbcnet.ablation --config configs/ablation_backbones.yaml
    python -m gbcnet.ablation --config configs/ablation_backbones.yaml --aggregate-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from . import plots
from .config import add_config_args, config_from_args
from .metrics import DECISION_THRESHOLD
from .train_binary import run_binary_cv
from .utils import Logger, ensure_dir, load_json


def load_run(run_dir: str | Path) -> Dict:
    """Read one train_binary run: per-fold test curves + summary (threshold 0.5)."""
    results = Path(run_dir) / "results"
    preds = pd.read_csv(results / "fold_predictions.csv")
    final = load_json(results / "final_metrics.json")
    curves, cms = [], []
    for fold, g in preds.groupby("fold"):
        t, p = g["target"].values, g["prob"].values
        y = (p >= DECISION_THRESHOLD).astype(int)
        curves.append({"fold": int(fold), "targets": t, "probabilities": p, "auc": roc_auc_score(t, p), "acc": (y == t).mean()})
        cms.append(confusion_matrix(t, y, labels=[0, 1]))
    return {"curves": curves, "pooled_cm": np.sum(cms, axis=0), "summary": final, "n_splits": final["n_splits"]}


def main():
    parser = argparse.ArgumentParser(description="Backbone comparison sweep")
    add_config_args(parser, default_config="configs/ablation_backbones.yaml")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip training; rebuild comparison from saved runs")
    args = parser.parse_args()
    cfg = config_from_args(args)

    out_dir = ensure_dir(cfg["output"]["dir"])
    log = Logger(out_dir / "ablation.log")
    model_names: List[str] = cfg["models"]

    if not args.aggregate_only:
        for name in model_names:
            log(f"\n{'#' * 78}\n# {name}\n{'#' * 78}")
            run_binary_cv(cfg, name, out_dir / name, log)

    runs = {name: load_run(out_dir / name) for name in model_names if (out_dir / name / "results").exists()}
    proposed_label = cfg.get("proposed_label", "Proposed")
    proposed_dir = cfg.get("proposed_run_dir")
    if proposed_dir and (Path(proposed_dir) / "results" / "fold_predictions.csv").exists():
        runs[proposed_label] = load_run(proposed_dir)
    else:
        log(f"Proposed run not found at {proposed_dir!r}; it is excluded from the comparison.")
        proposed_label = None

    if not runs:
        raise SystemExit("No completed runs to compare.")

    rows = []
    for name, run in runs.items():
        s = run["summary"]
        rows.append({
            "model": name, "n_splits": run["n_splits"], "is_proposed": name == proposed_label,
            **{f"{k}_{stat}": s[f"{k}_{stat}"] for k in ["auc", "accuracy", "f1", "sensitivity", "specificity"]
               for stat in ["mean", "std"]},
            "pooled_auc": s["pooled_auc"],
        })
    comparison = pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    comparison.to_csv(out_dir / "all_models_comparison_summary.csv", index=False)
    log(f"\nComparison (threshold {DECISION_THRESHOLD}):\n" + comparison.round(4).to_string(index=False))

    plots.plot_model_bars(comparison, "auc_mean", "auc_std", "Test AUC (mean +/- std)", "Model Comparison: Test AUC",
                          out_dir / "all_models_auc_comparison_bar.png")
    plots.plot_model_bars(comparison, "accuracy_mean", "accuracy_std", "Test Accuracy (mean +/- std)",
                          "Model Comparison: Test Accuracy", out_dir / "all_models_accuracy_comparison_bar.png")
    plots.plot_model_mean_rocs({n: r["curves"] for n, r in runs.items()}, proposed_label,
                               out_dir / "all_models_combined_roc_curves.png")
    plots.plot_confusion_grid({n: r["pooled_cm"] for n, r in runs.items()}, out_dir / "all_models_pooled_confusion_matrices.png")
    log(f"\nSaved comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
