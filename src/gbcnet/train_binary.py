"""K-fold CV training for the binary (cancer vs. normal) task.

Folds are carved from the manifest's `train_val` rows only; every fold's best
checkpoint is evaluated on the same fixed `test` rows. Predictions use a fixed
decision threshold of 0.5 (no threshold tuning).

    python -m gbcnet.train_binary --config configs/binary_convnext_cbam_msam.yaml
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from . import plots
from .config import add_config_args, config_from_args, save_yaml
from .data.binary import balanced_sampler, build_binary_transforms, dataset_from_cfg, load_binary_manifest, make_loader
from .engine import autotune_batch_size, predict_probs, train_two_phase
from .losses import FocalLoss
from .metrics import DECISION_THRESHOLD, binary_metrics, bootstrap_binary_ci, summarize_folds
from .models.registry import get_spec
from .splits import stratified_kfold
from .utils import Logger, ensure_dir, environment_info, get_device, save_json, set_seed


def run_binary_cv(cfg: Dict, model_name: str, out_dir: str | Path, log: Callable[[str], None] = print) -> Dict:
    """Train `model_name` with k-fold CV and evaluate on the fixed test set."""
    out_dir = ensure_dir(out_dir)
    models_dir = ensure_dir(out_dir / "trained_models")
    results_dir = ensure_dir(out_dir / "results")
    save_yaml({**cfg, "model": {**cfg["model"], "name": model_name}}, out_dir / "config_resolved.yaml")
    save_json(environment_info(), out_dir / "environment.json")

    set_seed(cfg["seed"], cfg.get("deterministic", False))
    device = get_device(cfg.get("device", "auto"))
    tcfg, ecfg = cfg["train"], cfg["eval"]
    spec = get_spec(model_name)

    train_val_df, test_df = load_binary_manifest(cfg["data"]["manifest"], cfg["data"]["image_dir"])
    log(f"Model: {model_name} | device: {device}")
    log(f"Test (fixed): {len(test_df)} images - Cancer {(test_df.label == 1).sum()} / Normal {(test_df.label == 0).sum()}")
    log(f"Train+Val pool: {len(train_val_df)} images - Cancer {(train_val_df.label == 1).sum()} / Normal {(train_val_df.label == 0).sum()}")
    test_df[["relative_path", "label"]].to_csv(results_dir / "test_set.csv", index=False)

    batch_size, accumulation_steps = autotune_batch_size(spec, cfg, device, log)
    log(f"batch_size={batch_size}, accumulation_steps={accumulation_steps} (effective {batch_size * accumulation_steps})")

    train_tf, eval_tf = build_binary_transforms(cfg["transforms"]["image_size"], cfg["transforms"]["center_crop"])
    num_workers = cfg["data"]["num_workers"]
    test_loader = make_loader(dataset_from_cfg(test_df, eval_tf, cfg), batch_size, num_workers)

    folds = stratified_kfold(train_val_df["label"].values, cfg["cv"]["n_splits"], cfg["seed"])
    fold_rows, prediction_rows, histories, fold_metrics = [], [], [], []
    fold_curves = []  # for ROC figures: test probabilities per fold
    t_start = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_train_df = train_val_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = train_val_df.iloc[va_idx].reset_index(drop=True)
        tag = f"{model_name}_fold{fold_idx}"
        log(f"\n{'=' * 78}\nFOLD {fold_idx}/{len(folds)}: train={len(fold_train_df)} val={len(fold_val_df)} "
            f"[{(time.time() - t_start) / 60:.1f} min elapsed]\n{'=' * 78}")

        sampler = balanced_sampler(fold_train_df["label"].values) if tcfg.get("weighted_sampler", True) else None
        train_loader = make_loader(dataset_from_cfg(fold_train_df, train_tf, cfg), batch_size, num_workers,
                                   sampler=sampler, shuffle=sampler is None, drop_last=True)
        val_loader = make_loader(dataset_from_cfg(fold_val_df, eval_tf, cfg), batch_size, num_workers)

        model = spec.factory(pretrained=cfg["model"]["pretrained"], dropout=cfg["model"]["dropout"])
        model = model.to(device, memory_format=torch.channels_last)
        criterion = FocalLoss(tcfg["focal_alpha"], tcfg["focal_gamma"])
        ckpt = models_dir / f"{tag}_best.pth"

        history, best_val_f1 = train_two_phase(model, spec, train_loader, val_loader, criterion, tcfg,
                                               accumulation_steps, ckpt, device, log)
        histories.append(history)
        save_json(history, results_dir / f"{tag}_history.json")
        plots.plot_training_history(history, f"{model_name} fold {fold_idx}", results_dir / f"{tag}_training_history.png")

        model.load_state_dict(torch.load(ckpt, map_location=device))
        test_t, test_p, test_paths = predict_probs(model, test_loader, device, tcfg)

        m = binary_metrics(test_t, test_p, DECISION_THRESHOLD)
        fold_metrics.append({**m, "fold": fold_idx})
        fold_rows.append({
            "fold": fold_idx, "epochs_trained": len(history["val_f1"]), "best_val_f1": best_val_f1,
            **{k: m[k] for k in ["auc", "f1", "sensitivity", "specificity", "precision", "accuracy"]},
        })
        plots.plot_confusion_matrix(m["confusion_matrix"], f"{model_name} fold {fold_idx} (threshold {DECISION_THRESHOLD})",
                                    results_dir / f"{tag}_confusion_matrix.png")
        plots.plot_roc(test_t, test_p, f"{model_name} fold {fold_idx} - ROC", results_dir / f"{tag}_roc_curve.png")
        fold_curves.append({"fold": fold_idx, "targets": test_t, "probabilities": test_p, "auc": m["auc"]})
        assert test_paths == test_df["filepath"].tolist()  # unshuffled loader preserves manifest order
        prediction_rows += [
            {"fold": fold_idx, "relative_path": rel, "target": int(t), "prob": float(pr), "pred": int(y)}
            for rel, t, pr, y in zip(test_df["relative_path"], test_t, test_p, m["predictions"])
        ]
        log(f"Fold {fold_idx}: AUC={m['auc']:.4f} F1={m['f1']:.4f} Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f}")

        del model, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(fold_rows).to_csv(results_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(results_dir / "fold_predictions.csv", index=False)

    targets = np.concatenate([c["targets"] for c in fold_curves])
    probs = np.concatenate([c["probabilities"] for c in fold_curves])
    preds = np.concatenate([m["predictions"] for m in fold_metrics])
    tn, fp, fn, tp = confusion_matrix(targets.astype(int), preds, labels=[0, 1]).ravel()
    final = {
        "model_name": model_name, "n_splits": len(folds), "n_test_images": len(test_df), "threshold": DECISION_THRESHOLD,
        **summarize_folds(fold_metrics),
        "pooled_n_predictions": int(len(targets)),
        "pooled_auc": float(roc_auc_score(targets, probs)),
        "pooled_sensitivity": tp / (tp + fn),
        "pooled_specificity": tn / (tn + fp),
        "pooled_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "pooled_accuracy": (tp + tn) / len(targets),
        "pooled_f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "pooled_confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if ecfg.get("bootstrap", 0) > 0:
        final["pooled_bootstrap_ci_95"] = bootstrap_binary_ci(targets, probs, preds, ecfg["bootstrap"], cfg["seed"])

    log(f"\n{'=' * 78}\n{model_name} {len(folds)}-fold summary (fixed {len(test_df)}-image test set, "
        f"threshold {DECISION_THRESHOLD})\n{'=' * 78}")
    for key in ["auc", "f1", "sensitivity", "specificity", "precision", "accuracy"]:
        log(f"  {key:<12} {final[f'{key}_mean']:.4f} +/- {final[f'{key}_std']:.4f}")
    log(f"  pooled AUC   {final['pooled_auc']:.4f} over {final['pooled_n_predictions']} predictions")
    for k, v in final.get("pooled_bootstrap_ci_95", {}).items():
        log(f"  95% CI {k:<12} {v['lower']:.4f} - {v['upper']:.4f}")

    save_json(final, results_dir / "final_metrics.json")
    pd.DataFrame([{k: v for k, v in final.items() if not isinstance(v, (dict, list))}]).to_csv(
        results_dir / "final_metrics.csv", index=False)

    plots.plot_confusion_matrix(final["pooled_confusion_matrix"],
                                f"{model_name} - pooled ({len(folds)} folds x {len(test_df)} test images)",
                                results_dir / "pooled_confusion_matrix.png")
    plots.plot_roc(targets, probs, f"{model_name} - pooled ROC", results_dir / "pooled_roc_curve.png", label="Pooled ROC")
    plots.plot_fold_rocs(fold_curves, f"{model_name} - ROC across {len(folds)} folds", results_dir / "combined_roc_curves.png")
    plots.plot_fold_training_curves(histories, model_name, results_dir / "combined_training_curves.png")
    log(f"\nSaved outputs to {out_dir} [{(time.time() - t_start) / 60:.1f} min]")
    return {"final": final, "fold_curves": fold_curves, "fold_metrics": fold_metrics}


def main():
    parser = argparse.ArgumentParser(description="K-fold CV training for the binary GBC task")
    add_config_args(parser, default_config="configs/binary_convnext_cbam_msam.yaml")
    args = parser.parse_args()
    cfg = config_from_args(args)
    out_dir = Path(cfg["output"]["dir"])
    run_binary_cv(cfg, cfg["model"]["name"], out_dir, Logger(out_dir / "train.log"))


if __name__ == "__main__":
    main()
