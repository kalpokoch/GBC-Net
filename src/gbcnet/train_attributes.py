"""Multi-label attribute classifier: multi-seed x grouped k-fold CV with OOF evaluation.

Pipeline (from train_v3_multiseed_maskedloss.ipynb):
  1. load LLM- or rule-extracted attribute labels; rows with zero positives are
     report-parsing failures -> kept but masked from loss, metrics and sampling
  2. grouped MultilabelStratifiedKFold: same-report sibling slices never span folds
  3. for each seed x fold: BiomedCLIP @336, dual-scale, head-only warmup then
     partial encoder unfreeze, masked asymmetric loss, early stopping on val AUC,
     TTA predictions on the held-out fold -> OOF probabilities
  4. seed-averaged OOF -> metrics at a fixed 0.5 threshold for every label
  5. optional refit on the full pool for mean(best_epoch) epochs

    python -m gbcnet.train_attributes --config configs/attributes_biomedclip.yaml
"""

from __future__ import annotations

import argparse
import copy
import gc
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import WeightedRandomSampler

from .config import add_config_args, config_from_args, save_yaml
from .data.attributes import build_attribute_transforms, build_tta_transforms, load_attribute_labels, make_attribute_loader
from .losses import MaskedAsymmetricLossMultiLabel
from .metrics import DECISION_THRESHOLD, masked_mean_auc, multilabel_metrics
from .models.biomedclip import BiomedCLIPClassifier, set_encoder_trainable
from .splits import grouped_multilabel_kfold
from .utils import Logger, ensure_dir, environment_info, get_device, save_json, set_seed

DEVICE = get_device()
USE_AMP = DEVICE.type == "cuda"


def _autocast():
    return torch.autocast(device_type="cuda", enabled=USE_AMP) if USE_AMP else torch.autocast("cpu", enabled=False)


def _scaler():
    return torch.amp.GradScaler("cuda", enabled=USE_AMP)


def head_only_optimizer(model, tcfg: Dict, epochs: int):
    optimizer = torch.optim.AdamW([{"params": model.classifier.parameters(), "lr": tcfg["lr_head"]}],
                                  weight_decay=tcfg["weight_decay"])
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=tcfg["warmup_epochs"])
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - tcfg["warmup_epochs"], 1), eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[tcfg["warmup_epochs"]])
    return optimizer, scheduler, _scaler()


def partial_unfreeze_optimizer(model, tcfg: Dict, epochs: int, elapsed: int):
    """Encoder+head optimizer whose schedule continues from epoch `elapsed`."""
    enc_params = [p for p in model.vision_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": tcfg["lr_encoder"]}, {"params": model.classifier.parameters(), "lr": tcfg["lr_head"]}],
        weight_decay=tcfg["weight_decay"],
    )
    remaining = epochs - elapsed
    warmup_remaining = max(0, tcfg["warmup_epochs"] - elapsed)
    cosine_steps = remaining - warmup_remaining
    if warmup_remaining > 0:
        start = 0.1 + 0.9 * elapsed / tcfg["warmup_epochs"]
        warmup = LinearLR(optimizer, start_factor=start, end_factor=1.0, total_iters=warmup_remaining)
        if cosine_steps > 0:
            cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-7)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_remaining])
        else:
            scheduler = warmup
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining, 1), eta_min=1e-7)
    return optimizer, scheduler, _scaler()


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, save_path: Optional[str] = None):
        self.patience, self.min_delta, self.save_path = patience, min_delta, save_path
        self.best_auc, self.best_val_loss, self.best_epoch = -float("inf"), float("inf"), 0
        self.counter, self.early_stop, self.best_state = 0, False, None

    def __call__(self, val_auc: float, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc, self.best_val_loss, self.best_epoch, self.counter = val_auc, val_loss, epoch, 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
            else:
                self.best_state = copy.deepcopy(model.state_dict())
            return True
        self.counter += 1
        self.early_stop = self.counter >= self.patience
        return False


def make_sample_weights(df_train: pd.DataFrame, label_cols) -> torch.Tensor:
    """Rare-label-aware row weights (max inverse label frequency); masked rows get 0."""
    y = df_train[label_cols].values.astype(np.float32)
    inv_freq = 1.0 / y.mean(axis=0).clip(min=1e-3)
    row_weight = np.where(y.sum(axis=1) > 0, (y * inv_freq).max(axis=1), 1.0)
    row_weight = np.where(df_train["label_valid"].values.astype(bool), row_weight, 0.0)
    return torch.tensor(row_weight, dtype=torch.double)


def train_epoch(model, loader, criterion, optimizer, scaler, grad_clip: float) -> float:
    model.train()
    total, n = 0.0, 0
    for imgs, labels, mask, _ in loader:
        imgs, labels, mask = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            loss = criterion(model(imgs), labels, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total = 0.0
    probs, ytrue, masks, ids = [], [], [], []
    for imgs, labels, mask, batch_ids in loader:
        with _autocast():
            logits = model(imgs.to(DEVICE, non_blocking=True))
            loss = criterion(logits, labels.to(DEVICE), mask.to(DEVICE))
        total += loss.item()
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ytrue.append(labels.numpy())
        masks.append(mask.numpy())
        ids.extend(batch_ids)
    probs, ytrue, masks = np.vstack(probs), np.vstack(ytrue), np.vstack(masks)
    return total / max(len(loader), 1), masked_mean_auc(ytrue, probs, masks), probs, ytrue, masks, ids


@torch.no_grad()
def eval_tta(model, df_eval: pd.DataFrame, cfg: Dict, criterion):
    variant_probs, total_loss = [], 0.0
    for tf in build_tta_transforms(cfg["transforms"]):
        loss, _, probs, ytrue, masks, ids = eval_epoch(model, make_attribute_loader(cfg, df_eval, tf), criterion)
        variant_probs.append(probs)
        total_loss += loss
    probs = np.mean(variant_probs, axis=0)
    n = len(variant_probs)
    return total_loss / n, masked_mean_auc(ytrue, probs, masks), probs, ytrue, masks, ids


def build_model(cfg: Dict) -> BiomedCLIPClassifier:
    mcfg, tcfg = cfg["model"], cfg["transforms"]
    return BiomedCLIPClassifier(mcfg["name"], len(cfg["data"]["label_cols"]), mcfg["hidden_dim"], mcfg["dropout"],
                                tcfg["image_size"], tcfg["use_dual_scale"]).to(DEVICE)


def criterion_from_cfg(tcfg: Dict) -> MaskedAsymmetricLossMultiLabel:
    return MaskedAsymmetricLossMultiLabel(tcfg["asl_gamma_neg"], tcfg["asl_gamma_pos"], tcfg["asl_clip"])


def train_sampler(cfg: Dict, df_train: pd.DataFrame):
    if not cfg["train"].get("use_weighted_sampler", True):
        return None
    w = make_sample_weights(df_train, cfg["data"]["label_cols"])
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


def run_training(cfg: Dict, seed: int, df_train, df_val, save_path: str, log: Callable[[str], None]):
    tcfg = cfg["train"]
    set_seed(seed, cfg.get("deterministic", True))
    train_loader = make_attribute_loader(cfg, df_train, build_attribute_transforms(cfg["transforms"], "train"),
                                         shuffle=True, sampler=train_sampler(cfg, df_train))
    val_loader = make_attribute_loader(cfg, df_val, build_attribute_transforms(cfg["transforms"], "val"))

    model = build_model(cfg)
    criterion = criterion_from_cfg(tcfg)
    set_encoder_trainable(model, "frozen")
    epochs = tcfg["epochs"]
    optimizer, scheduler, scaler = head_only_optimizer(model, tcfg, epochs)
    stopper = EarlyStopping(patience=tcfg["patience"], save_path=save_path)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    for epoch in range(1, epochs + 1):
        if epoch == tcfg["freeze_warmup_epochs"] + 1:
            n = set_encoder_trainable(model, "partial", tcfg["unfreeze_last_n_blocks"])
            log(f"  [epoch {epoch}] unfreezing last {n} encoder blocks")
            optimizer, scheduler, scaler = partial_unfreeze_optimizer(model, tcfg, epochs, tcfg["freeze_warmup_epochs"])
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, tcfg["grad_clip"])
        vl_loss, vl_auc, *_ = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        for k, v in zip(history, (tr_loss, vl_loss, vl_auc)):
            history[k].append(v)
        improved = stopper(vl_auc, vl_loss, model, epoch)
        log(f"  ep {epoch:>3}  train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  val_auc={vl_auc:.4f}  "
            + ("SAVED" if improved else f"({stopper.counter}/{tcfg['patience']})"))
        if stopper.early_stop:
            log(f"  early stop at epoch {epoch} (best={stopper.best_epoch}, auc={stopper.best_auc:.4f})")
            break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, stopper, history


def main():
    parser = argparse.ArgumentParser(description="Multi-seed grouped-CV attribute classifier")
    add_config_args(parser, default_config="configs/attributes_biomedclip.yaml")
    args = parser.parse_args()
    cfg = config_from_args(args)

    out_dir = ensure_dir(cfg["output"]["dir"])
    cv_dir = ensure_dir(out_dir / "cv_folds")
    log = Logger(out_dir / "train.log")
    save_yaml(cfg, out_dir / "config_resolved.yaml")
    save_json(environment_info(), out_dir / "environment.json")
    log(f"Output dir: {out_dir.resolve()} | device: {DEVICE}")

    label_cols = cfg["data"]["label_cols"]
    df_pool = load_attribute_labels(cfg, log)
    valid_df = df_pool[df_pool["label_valid"]]
    prevalence = valid_df[label_cols].sum().rename("positive_count").to_frame()
    prevalence["pct"] = (valid_df[label_cols].mean() * 100).values
    log("\nLabel positives (masked rows excluded):\n" + prevalence.to_string())
    prevalence.to_csv(out_dir / "label_prevalence.csv")
    low_prevalence = prevalence.index[prevalence["positive_count"] < cfg["data"]["low_prevalence_threshold"]].tolist()

    splits, group_ids = grouped_multilabel_kfold(df_pool, label_cols, cfg["cv"]["n_folds"], cfg["seed"],
                                                 cfg["cv"].get("group_min_chars", 25), log)
    df_pool["group_id"] = group_ids.values
    df_pool.to_csv(out_dir / "df_pool_with_groups.csv", index=False)

    seeds = cfg["seeds"]
    n_labels = len(label_cols)
    oof_per_seed = np.zeros((len(seeds), len(df_pool), n_labels), dtype=np.float32)
    filled = np.zeros(len(df_pool), dtype=bool)
    label_mask = np.repeat(df_pool["label_valid"].values.astype(bool)[:, None], n_labels, axis=1)
    criterion = criterion_from_cfg(cfg["train"])

    cv_rows = []
    t_start = time.time()
    for si, seed in enumerate(seeds):
        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            df_tr, df_va = df_pool.iloc[tr_idx].reset_index(drop=True), df_pool.iloc[va_idx].reset_index(drop=True)
            log(f"\n{'=' * 70}\nSEED {seed} FOLD {fold}/{len(splits)} (train={len(df_tr)}, val={len(df_va)}) "
                f"[{(time.time() - t_start) / 60:.1f} min elapsed]\n{'=' * 70}")
            ckpt = str(cv_dir / f"seed{seed}_fold{fold}_best.pth")
            model, stopper, history = run_training(cfg, seed, df_tr, df_va, ckpt, log)
            save_json(history, cv_dir / f"seed{seed}_fold{fold}_history.json")

            _, _, val_probs, val_labels, val_mask, _ = eval_tta(model, df_va, cfg, criterion)
            oof_per_seed[si, va_idx] = val_probs
            filled[va_idx] = True

            per_label_auc = {}
            for i, col in enumerate(label_cols):
                valid = val_mask[:, i].astype(bool)
                ok = valid.sum() > 0 and len(np.unique(val_labels[valid, i])) > 1
                per_label_auc[f"auc_{col}"] = roc_auc_score(val_labels[valid, i], val_probs[valid, i]) if ok else np.nan
            cv_rows.append({"seed": seed, "fold": fold, "best_epoch": stopper.best_epoch, "best_val_auc": stopper.best_auc,
                            "best_val_loss": stopper.best_val_loss, **per_label_auc})

            del model
            gc.collect()
            if USE_AMP:
                torch.cuda.empty_cache()
            pd.DataFrame(cv_rows).to_csv(out_dir / "cv_fold_results.csv", index=False)
            np.save(out_dir / "oof_probs_per_seed.npy", oof_per_seed)

    assert filled.all(), "Every pool image should receive exactly one OOF prediction per seed"
    oof = oof_per_seed.mean(axis=0)
    cv_df = pd.DataFrame(cv_rows)
    np.save(out_dir / "oof_probs.npy", oof)
    np.save(out_dir / "label_mask.npy", label_mask)
    per_seed_val = cv_df.groupby("seed")["best_val_auc"].mean()
    log("\nCV summary (per seed x fold):\n" + cv_df.round(4).to_string(index=False))
    log(f"Mean val AUC over {len(cv_df)} runs: {cv_df['best_val_auc'].mean():.4f} +/- {cv_df['best_val_auc'].std():.4f}")

    y_pool = df_pool[label_cols].values.astype(np.float32)
    metrics = multilabel_metrics(y_pool, oof, label_cols, label_mask, threshold=DECISION_THRESHOLD)
    metrics.to_csv(out_dir / "oof_metrics.csv", index=False)
    log(f"\nOOF metrics @ threshold {DECISION_THRESHOLD}:\n" + metrics.round(4).to_string(index=False))

    per_seed_oof_auc = [
        float(multilabel_metrics(y_pool, oof_per_seed[si], label_cols, label_mask).query("Label == 'MEAN'")["AUC_ROC"].iloc[0])
        for si in range(len(seeds))
    ]

    final_epochs, final_ckpt = None, None
    if cfg.get("final_refit", True):
        final_epochs = max(1, int(round(cv_df["best_epoch"].mean())))
        log(f"\nRefitting on all {len(df_pool)} images for {final_epochs} epochs (no early stopping)")
        tcfg = cfg["train"]
        set_seed(cfg["seed"], cfg.get("deterministic", True))
        loader = make_attribute_loader(cfg, df_pool, build_attribute_transforms(cfg["transforms"], "train"),
                                       shuffle=True, sampler=train_sampler(cfg, df_pool))
        model = build_model(cfg)
        set_encoder_trainable(model, "frozen")
        optimizer, scheduler, scaler = head_only_optimizer(model, tcfg, final_epochs)
        for epoch in range(1, final_epochs + 1):
            if epoch == tcfg["freeze_warmup_epochs"] + 1:
                set_encoder_trainable(model, "partial", tcfg["unfreeze_last_n_blocks"])
                optimizer, scheduler, scaler = partial_unfreeze_optimizer(model, tcfg, final_epochs, tcfg["freeze_warmup_epochs"])
            loss = train_epoch(model, loader, criterion, optimizer, scaler, tcfg["grad_clip"])
            scheduler.step()
            if epoch % 5 == 0 or epoch == final_epochs:
                log(f"  ep {epoch:>3}/{final_epochs}  train_loss={loss:.4f}")
        final_ckpt = out_dir / "final_model_full_data.pth"
        torch.save(model.state_dict(), final_ckpt)
        log(f"Saved final model -> {final_ckpt}")

    mean_row = metrics.query("Label == 'MEAN'").iloc[0]
    group_counts = group_ids.value_counts()
    summary = {
        "label_source": cfg["data"]["labels_csv"],
        "labels_modeled": label_cols,
        "dataset_size": len(df_pool),
        "n_label_valid": int(label_mask[:, 0].sum()),
        "n_masked_unparsed": int((~label_mask[:, 0]).sum()),
        "evaluation_protocol": f"{len(splits)}_fold_grouped_oof_{len(seeds)}_seed_ensemble_no_held_out_test",
        "n_multi_image_groups": int((group_counts > 1).sum()),
        "n_images_in_multi_groups": int(group_counts[group_counts > 1].sum()),
        "seeds": seeds,
        "split_seed": cfg["seed"],
        "cv_mean_val_auc_all_runs": float(cv_df["best_val_auc"].mean()),
        "cv_std_val_auc_all_runs": float(cv_df["best_val_auc"].std()),
        "per_seed_mean_val_auc": {int(k): float(v) for k, v in per_seed_val.items()},
        "per_seed_oof_mean_auc": per_seed_oof_auc,
        "std_across_seeds_oof_auc": float(np.std(per_seed_oof_auc)),
        "oof_mean_auc": float(mean_row["AUC_ROC"]),
        "oof_mean_ap": float(mean_row["AP"]),
        "threshold": DECISION_THRESHOLD,
        "oof_mean_accuracy": float(mean_row["Accuracy"]),
        "oof_mean_f1": float(mean_row["F1"]),
        "oof_mean_balanced_accuracy": float(mean_row["Balanced_Accuracy"]),
        "oof_mean_sensitivity": float(mean_row["Sensitivity"]),
        "oof_mean_specificity": float(mean_row["Specificity"]),
        "low_prevalence_labels": low_prevalence,
        "final_model_epochs": final_epochs,
        "final_model_path": final_ckpt,
    }
    save_json(summary, out_dir / "final_summary.json")
    log(f"\nOOF mean AUC {summary['oof_mean_auc']:.4f} | F1 {summary['oof_mean_f1']:.4f} | "
        f"seed std {summary['std_across_seeds_oof_auc']:.4f} | total {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
