"""Training/inference engine for the binary task (shared by every registry model).

Recipe:
  phase 1  backbone + attention frozen, AdamW on attention/classifier,
           CosineAnnealingLR over `warmup_epochs`
  phase 2  everything trainable, per-group AdamW learning rates,
           CosineAnnealingWarmRestarts; early stopping on val F1 after `min_epochs`
MixUp, focal loss, bf16 autocast, channels_last, gradient accumulation and
gradient clipping throughout. The best-val-F1 checkpoint is kept.
"""

from __future__ import annotations

import contextlib
import gc
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .losses import FocalLoss, mixup_criterion, mixup_data
from .models.registry import ModelSpec
from .utils import amp_dtype

History = Dict[str, List[float]]


def autocast(device: torch.device, tcfg: Dict):
    if device.type != "cuda" or not tcfg.get("use_amp", True):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype(tcfg.get("amp_dtype", "bfloat16")))


def _params(modules: List[nn.Module]) -> List[nn.Parameter]:
    return [p for m in modules for p in m.parameters()]


def set_backbone_trainable(model: nn.Module, spec: ModelSpec, trainable: bool) -> None:
    for module in spec.backbone(model) + spec.attention(model):
        for p in module.parameters():
            p.requires_grad = trainable


def build_warmup_optimizer(model: nn.Module, spec: ModelSpec, tcfg: Dict) -> torch.optim.Optimizer:
    lr, wd = tcfg["warmup_lr"], tcfg["warmup_wd"]
    groups = []
    if spec.attention(model):
        groups.append({"params": _params(spec.attention(model)), "lr": lr["attention"], "weight_decay": wd["attention"]})
    groups.append({"params": _params(spec.classifier(model)), "lr": lr["classifier"], "weight_decay": wd["classifier"]})
    return torch.optim.AdamW(groups)


def build_finetune_optimizer(model: nn.Module, spec: ModelSpec, tcfg: Dict) -> torch.optim.Optimizer:
    lr, wd = tcfg["finetune_lr"], tcfg["finetune_wd"]
    groups = [{"params": _params(spec.backbone(model)), "lr": lr["backbone"], "weight_decay": wd["backbone"]}]
    if spec.attention(model):
        groups.append({"params": _params(spec.attention(model)), "lr": lr["attention"], "weight_decay": wd["attention"]})
    groups.append({"params": _params(spec.classifier(model)), "lr": lr["classifier"], "weight_decay": wd["classifier"]})
    return torch.optim.AdamW(groups, weight_decay=wd["backbone"])


def train_epoch(model, train_loader, val_loader, criterion, optimizer, accumulation_steps, device, tcfg, desc="") -> Dict[str, float]:
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    optimizer.zero_grad()
    grad_clip = tcfg.get("grad_clip", 1.0)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{desc} train", leave=False)

    i = -1
    for i, (images, labels, _) in pbar:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        inputs, labels_a, labels_b, lam = mixup_data(images, labels, alpha=tcfg.get("mixup_alpha", 0.2))

        with autocast(device, tcfg):
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps * images.size(0)
        preds = (torch.sigmoid(outputs.float()) > 0.5).float()
        primary_labels = labels_a if lam > 0.5 else labels_b
        train_correct += (preds == primary_labels).sum().item()
        train_total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}", acc=f"{train_correct / train_total:.4f}")

    if i >= 0 and (i + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with autocast(device, tcfg):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs.float()) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_preds.extend(preds.cpu().numpy().ravel())
            val_targets.extend(labels.cpu().numpy().ravel())

    return {
        "train_loss": train_loss / max(train_total, 1),
        "train_acc": train_correct / max(train_total, 1),
        "val_loss": val_loss / max(val_total, 1),
        "val_acc": val_correct / max(val_total, 1),
        "val_f1": float(f1_score(val_targets, val_preds, zero_division=0)),
    }


@torch.no_grad()
def predict_probs(model, loader, device, tcfg) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    targets, probs, paths = [], [], []
    for images, labels, batch_paths in tqdm(loader, desc="predict", leave=False):
        images = images.to(device, memory_format=torch.channels_last)
        with autocast(device, tcfg):
            outputs = model(images)
        probs.extend(np.atleast_1d(torch.sigmoid(outputs.float()).squeeze(1).cpu().numpy()))
        targets.extend(labels.numpy())
        paths.extend(batch_paths)
    return np.asarray(targets), np.asarray(probs), paths


def train_two_phase(
    model: nn.Module,
    spec: ModelSpec,
    train_loader,
    val_loader,
    criterion,
    tcfg: Dict,
    accumulation_steps: int,
    checkpoint_path: str | Path,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> Tuple[History, float]:
    history: History = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    best_val_f1 = -1.0  # guarantees a checkpoint after epoch 1 even if val F1 is 0
    patience_counter = 0
    epochs, warmup_epochs = tcfg["epochs"], tcfg["warmup_epochs"]

    def record(epoch: int, total: int, eh: Dict[str, float], phase: str) -> bool:
        nonlocal best_val_f1
        for k in history:
            history[k].append(eh[k])
        improved = eh["val_f1"] > best_val_f1
        if improved:
            best_val_f1 = eh["val_f1"]
            torch.save(model.state_dict(), checkpoint_path)
        log(f"  [{phase}] epoch {epoch + 1:>3}/{total}  train_loss={eh['train_loss']:.4f} train_acc={eh['train_acc']:.4f}  "
            f"val_loss={eh['val_loss']:.4f} val_acc={eh['val_acc']:.4f} val_f1={eh['val_f1']:.4f}"
            + ("  * saved" if improved else ""))
        return improved

    # Phase 1: warmup with backbone (and attention) frozen
    set_backbone_trainable(model, spec, False)
    optimizer = build_warmup_optimizer(model, spec, tcfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(warmup_epochs, 1), eta_min=tcfg["warmup_eta_min"])
    for epoch in range(warmup_epochs):
        eh = train_epoch(model, train_loader, val_loader, criterion, optimizer, accumulation_steps, device, tcfg, desc=f"ep{epoch + 1}")
        scheduler.step()
        record(epoch, epochs, eh, "warmup")

    # Phase 2: full fine-tuning
    set_backbone_trainable(model, spec, True)
    optimizer = build_finetune_optimizer(model, spec, tcfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=tcfg["restart_t0"], T_mult=tcfg["restart_t_mult"], eta_min=tcfg["finetune_eta_min"]
    )
    for epoch in range(warmup_epochs, epochs):
        eh = train_epoch(model, train_loader, val_loader, criterion, optimizer, accumulation_steps, device, tcfg, desc=f"ep{epoch + 1}")
        scheduler.step()
        patience_counter = 0 if record(epoch, epochs, eh, "finetune") else patience_counter + 1
        if epoch + 1 >= tcfg["min_epochs"] and patience_counter >= tcfg["early_stopping_patience"]:
            log(f"  early stopping after {epoch + 1} epochs")
            break

    log(f"  done: {len(history['val_f1'])} epochs, best val F1 = {best_val_f1:.4f}")
    return history, best_val_f1


def autotune_batch_size(spec: ModelSpec, cfg: Dict, device: torch.device, log: Callable[[str], None] = print) -> Tuple[int, int]:
    """Largest per-step batch (64/32/16/8, divisors of the effective batch) whose
    worst-case memory (full fine-tune, AdamW state allocated) fits within 85% of
    GPU memory. accumulation_steps keeps batch*accumulation = effective batch."""
    tcfg = cfg["train"]
    effective = tcfg["effective_batch_size"]
    fallback = tcfg["batch_size"]
    if device.type != "cuda" or not tcfg.get("autotune_batch_size", True):
        return fallback, max(effective // fallback, 1)

    image_size = cfg["transforms"]["image_size"]
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9

    def try_size(batch_size: int):
        model = optimizer = None
        try:
            model = spec.factory(pretrained=False, dropout=0.5).to(device, memory_format=torch.channels_last)
            model.train()
            optimizer = build_finetune_optimizer(model, spec, tcfg)
            criterion = FocalLoss(tcfg["focal_alpha"], tcfg["focal_gamma"])
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):  # AdamW moments allocate on the first step
                x = torch.randn(batch_size, 1, image_size, image_size, device=device).to(memory_format=torch.channels_last)
                y = torch.randint(0, 2, (batch_size, 1), device=device).float()
                inputs, ya, yb, lam = mixup_data(x, y, alpha=0.2)
                optimizer.zero_grad()
                with autocast(device, tcfg):
                    loss = mixup_criterion(criterion, model(inputs), ya, yb, lam)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            return True, torch.cuda.max_memory_allocated() / 1e9
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            if isinstance(e, torch.cuda.OutOfMemoryError) or any(s in msg for s in ("out of memory", "nvml", "cudacachingallocator")):
                return False, None
            raise
        finally:
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()

    log(f"Auto-tuning batch size (GPU memory {total_gb:.1f} GB, 85% budget)")
    for bs in [b for b in (64, 32, 16, 8) if effective % b == 0]:
        ok, peak = try_size(bs)
        if ok and peak <= total_gb * 0.85:
            log(f"  batch_size={bs}: OK (peak {peak:.2f} GB)")
            return bs, effective // bs
        log(f"  batch_size={bs}: " + (f"{peak:.2f} GB exceeds budget" if ok else "OOM"))
    log(f"  falling back to batch_size={fallback}")
    return fallback, max(effective // fallback, 1)
