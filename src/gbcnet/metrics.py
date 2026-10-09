"""Binary and multi-label evaluation metrics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------


# Fixed decision threshold for both tasks (no threshold tuning).
DECISION_THRESHOLD = 0.5


def binary_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float = DECISION_THRESHOLD) -> Dict:
    targets = np.asarray(targets).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else float("nan"),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "accuracy": float((tp + tn) / cm.sum()),
        "confusion_matrix": cm.tolist(),
        "predictions": preds,
    }


SUMMARY_KEYS = ["auc", "f1", "sensitivity", "specificity", "precision", "accuracy"]


def summarize_folds(fold_metrics: List[Dict]) -> Dict[str, float]:
    """Mean and (population) std across folds."""
    out = {}
    for key in SUMMARY_KEYS:
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        out[f"{key}_mean"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))
    return out


def bootstrap_binary_ci(
    targets: np.ndarray, probs: np.ndarray, preds: np.ndarray, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs over images for AUC, F1, sensitivity, specificity, accuracy."""
    rng = np.random.default_rng(seed)
    targets, probs, preds = map(np.asarray, (targets, probs, preds))
    n = len(targets)
    values = {k: [] for k in ["auc", "f1", "sensitivity", "specificity", "accuracy"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t, p, y = targets[idx], probs[idx], preds[idx]
        if len(np.unique(t)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(t, y, labels=[0, 1]).ravel()
        values["auc"].append(roc_auc_score(t, p))
        values["f1"].append(f1_score(t, y, zero_division=0))
        values["sensitivity"].append(tp / (tp + fn))
        values["specificity"].append(tn / (tn + fp))
        values["accuracy"].append((tp + tn) / n)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: {"mean": float(np.mean(v)), "lower": float(np.percentile(v, lo)), "upper": float(np.percentile(v, hi))}
        for k, v in values.items()
        if v
    }


# ---------------------------------------------------------------------------
# Multi-label (masked)
# ---------------------------------------------------------------------------


def masked_mean_auc(ytrue: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> float:
    aucs = []
    for i in range(ytrue.shape[1]):
        valid = mask[:, i].astype(bool)
        if valid.sum() == 0 or len(np.unique(ytrue[valid, i])) < 2:
            continue
        aucs.append(roc_auc_score(ytrue[valid, i], probs[valid, i]))
    return float(np.mean(aucs)) if aucs else 0.0


def multilabel_metrics(y_true, y_probs, label_cols, mask, threshold: float = DECISION_THRESHOLD) -> pd.DataFrame:
    """Per-label metrics on valid rows plus a MEAN row."""
    rows = []
    for i, col in enumerate(label_cols):
        valid = mask[:, i].astype(bool)
        yt, yp = y_true[valid, i].astype(int), y_probs[valid, i]
        ypred = (yp >= threshold).astype(int)
        two_classes = len(np.unique(yt)) > 1
        tn, fp, fn, tp = confusion_matrix(yt, ypred, labels=[0, 1]).ravel()
        rows.append(
            {
                "Label": col,
                "Threshold": threshold,
                "N_valid": int(valid.sum()),
                "Accuracy": accuracy_score(yt, ypred),
                "Balanced_Accuracy": balanced_accuracy_score(yt, ypred) if two_classes else np.nan,
                "AUC_ROC": roc_auc_score(yt, yp) if two_classes else np.nan,
                "AP": average_precision_score(yt, yp) if two_classes else np.nan,
                "F1": f1_score(yt, ypred, zero_division=0),
                "Sensitivity": recall_score(yt, ypred, zero_division=0),
                "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }
        )
    mdf = pd.DataFrame(rows)
    mean_cols = ["Accuracy", "Balanced_Accuracy", "AUC_ROC", "AP", "F1", "Sensitivity", "Specificity"]
    mean_row = {"Label": "MEAN", **{c: mdf[c].mean() for c in mean_cols}}
    return pd.concat([mdf, pd.DataFrame([mean_row])], ignore_index=True)
