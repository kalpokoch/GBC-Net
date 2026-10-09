"""Result figures (confusion matrices, ROC curves, training curves, model comparison)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import auc, roc_auc_score, roc_curve  # noqa: E402

CLASS_NAMES = ["Normal", "Cancer"]


def _save(fig, path: str | Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, title: str, path: str | Path, figsize=(8, 6)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    _save(fig, path)


def plot_roc(targets, probs, title: str, path: str | Path, label: str = "ROC curve") -> None:
    fpr, tpr, _ = roc_curve(targets, probs)
    score = roc_auc_score(targets, probs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"{label} (AUC = {score:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_training_history(history: Dict[str, List[float]], title: str, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title(f"{title} - Loss")
    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val Accuracy", linewidth=2)
    axes[1].set_title(f"{title} - Accuracy")
    axes[2].plot(history["val_f1"], label="Val F1", linewidth=2, color="green")
    axes[2].set_title(f"{title} - Val F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, path)


def mean_roc(fold_curves: Sequence[Dict]):
    """Interpolated mean ROC over folds; each item has 'targets' and 'probabilities'."""
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    for r in fold_curves:
        fpr, tpr, _ = roc_curve(np.asarray(r["targets"]), np.asarray(r["probabilities"]))
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr, np.std(tprs, axis=0), auc(mean_fpr, mean_tpr)


def plot_fold_rocs(fold_curves: Sequence[Dict], title: str, path: str | Path) -> None:
    """Per-fold ROC curves plus mean curve with a +/-1 std band."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fold_curves), 1)))
    for i, r in enumerate(fold_curves):
        fpr, tpr, _ = roc_curve(np.asarray(r["targets"]), np.asarray(r["probabilities"]))
        ax.plot(fpr, tpr, lw=1.5, alpha=0.6, color=colors[i], label=f"Fold {r['fold']} (AUC = {r['auc']:.4f})")
    mean_fpr, mean_tpr, std_tpr, mean_auc = mean_roc(fold_curves)
    std_auc = np.std([r["auc"] for r in fold_curves])
    ax.plot(mean_fpr, mean_tpr, color="black", lw=3, label=f"Mean ROC (AUC = {mean_auc:.4f} +/- {std_auc:.4f})")
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                    color="grey", alpha=0.25, label="+/- 1 std. dev.")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, path)


def plot_fold_training_curves(histories: Sequence[Dict], title_prefix: str, path: str | Path) -> None:
    """Per-fold validation curves plus NaN-padded mean."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(histories), 1)))
    for ax, (key, label) in zip(axes, [("val_loss", "Validation Loss"), ("val_acc", "Validation Accuracy"), ("val_f1", "Validation F1")]):
        max_len = max(len(h[key]) for h in histories)
        padded = []
        for i, h in enumerate(histories):
            ax.plot(h[key], alpha=0.5, lw=1.3, color=colors[i], label=f"Fold {i + 1}")
            padded.append(list(h[key]) + [np.nan] * (max_len - len(h[key])))
        ax.plot(np.nanmean(np.array(padded), axis=0), color="black", lw=2.5, label="Mean")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{title_prefix}: {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, path)


def plot_confusion_grid(cms: Dict[str, np.ndarray], path: str | Path) -> None:
    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4.5), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES, ax=ax, cbar=False)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    _save(fig, path)


def plot_model_bars(df, mean_col: str, std_col: str, ylabel: str, title: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:orange" if p else "tab:blue" for p in df["is_proposed"]]
    ax.bar(df["model"], df[mean_col], yerr=df[std_col], capsize=5, color=colors, alpha=0.85)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(handles=[Patch(color="tab:orange", label="Proposed model"), Patch(color="tab:blue", label="Comparison model")])
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, path)


def plot_model_mean_rocs(curves_by_model: Dict[str, Sequence[Dict]], proposed: str | None, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves_by_model), 1)))
    for i, (name, curves) in enumerate(curves_by_model.items()):
        mean_fpr, mean_tpr, _, mean_auc = mean_roc(curves)
        if name == proposed:
            ax.plot(mean_fpr, mean_tpr, lw=3.5, color="black", label=f"{name} (AUC={mean_auc:.3f})")
        else:
            ax.plot(mean_fpr, mean_tpr, lw=2, color=colors[i], label=f"{name} (AUC={mean_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Mean ROC Curves -- All Models")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    _save(fig, path)
