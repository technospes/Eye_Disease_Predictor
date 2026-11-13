#!/usr/bin/env python3
"""
train.py - Milestone 3: training + evaluation engine for the ARTSS DR classifier.

Changes vs the previous train.py / Colab variant
  * Stable mixed precision: bf16 on Ampere+, fp16 + GradScaler (with unscale-before-clip) otherwise.
    Non-finite loss/gradients skip the step instead of poisoning the weights.
  * Gradient clipping, per-step linear warmup + cosine decay, EMA of weights (fixes the erratic val loss).
  * Regularisation knobs (dropout, stochastic depth) to reduce the train/val gap.
  * Reproducible (seeded), argparse config, no hardcoded Colab paths.
  * Crash/disconnect safe: atomic checkpoints, full resume state (--resume), optional Drive backup dir.
  * Early stopping on val AUROC; history.csv written every epoch.
  * Final evaluation of the best checkpoint on val / test_internal / test_external:
    AUROC (+bootstrap CI), AUPRC, BCE, and sensitivity/specificity at a threshold chosen on VAL only.
    Logits are saved so calibration / thresholding / Grad-CAM can reuse them.

Usage (Colab):
    python train.py --root /content --backup-dir /content/drive/MyDrive/Eye_Disease_Predictor
    python train.py --root /content --backup-dir ... --resume        # after a disconnect
    python train.py --image-size 224 --batch-size 32                 # reproduce the old setup
Run it from (or next to) the folder containing dataset.py so `from dataset import ...` resolves.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from torch.amp import GradScaler, autocast

from dataset import get_dataloaders

log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Model (state-dict keys unchanged, so older checkpoints still load)
# --------------------------------------------------------------------------- #
class FundusClassifier(nn.Module):
    def __init__(self, backbone_name="efficientnet_b2", pretrained=True, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=1,
                                          drop_rate=drop_rate, drop_path_rate=drop_path_rate)

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # (B,) logits


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fixed input size -> faster; not bit-exact deterministic


def resolve_amp(choice: str, device: torch.device):
    """Returns the autocast dtype, or None for full fp32."""
    if device.type != "cuda" or choice == "off":
        return None
    if choice == "auto":
        choice = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    return torch.bfloat16 if choice == "bf16" else torch.float16


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def backup_file(path: Path, backup_dir: Path | None) -> None:
    """Best-effort copy (e.g. to Google Drive); never crashes training."""
    if backup_dir is None:
        return
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        tmp = backup_dir / (path.name + ".tmp")
        shutil.copy2(path, tmp)
        os.replace(tmp, backup_dir / path.name)
    except OSError as exc:
        log.warning("Backup of %s to %s failed: %s", path.name, backup_dir, exc)


class EMA:
    """Exponential moving average of weights and BN buffers, with warmup of the decay."""

    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                e.copy_(m)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_one_epoch(model, ema, loader, optimizer, scheduler, criterion, scaler, amp_dtype, device, clip, max_skips):
    model.train()
    losses, all_logits, all_labels = [], [], []
    skipped = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
        loss = criterion(logits.float(), labels)  # loss always in fp32

        if not torch.isfinite(loss):
            skipped += 1
            if skipped > max_skips:
                raise RuntimeError(f"Aborting: {skipped} non-finite losses in one epoch.")
            continue

        if scaler is not None:  # fp16 path
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # must unscale BEFORE clipping
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
            if not torch.isfinite(grad_norm):
                skipped += 1
                if skipped > max_skips:
                    raise RuntimeError(f"Aborting: {skipped} non-finite gradients in one epoch.")
                continue
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        losses.append(loss.item())
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.cpu())

    if not losses:
        raise RuntimeError("No successful training steps in this epoch.")
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return float(np.mean(losses)), roc_auc_score(labels, logits), skipped


@torch.no_grad()
def predict(model, loader, amp_dtype, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def compute_metrics(logits: np.ndarray, labels: np.ndarray, pos_weight: float) -> dict:
    t, y = torch.from_numpy(logits), torch.from_numpy(labels).float()
    return {
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(average_precision_score(labels, logits)),
        "bce": float(F.binary_cross_entropy_with_logits(t, y)),  # unweighted: comparable across runs/splits
        "weighted_bce": float(F.binary_cross_entropy_with_logits(t, y, pos_weight=torch.tensor([pos_weight]))),
    }


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(thr[np.argmax(tpr - fpr)])


def threshold_report(labels: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, probs >= thr, labels=[0, 1]).ravel()
    safe = lambda a, b: float(a / b) if b else None
    return {"threshold": thr, "sensitivity": safe(tp, tp + fn), "specificity": safe(tn, tn + fp),
            "ppv": safe(tp, tp + fp), "npv": safe(tn, tn + fn), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def bootstrap_auroc_ci(labels: np.ndarray, scores: np.ndarray, n: int, seed: int):
    if n <= 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        vals.append(roc_auc_score(labels[idx], scores[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def final_evaluation(model, loaders, amp_dtype, device, pos_weight, args, out_dir: Path, tag: str) -> dict:
    results, cache = {}, {}
    for name in ("val", "test_internal", "test_external"):
        logits, labels = predict(model, getattr(loaders, name), amp_dtype, device)
        cache[name] = (logits, labels)
        np.savez_compressed(out_dir / f"{tag}_logits_{name}.npz", logits=logits, labels=labels)
        results[name] = compute_metrics(logits, labels, pos_weight)
        results[name]["auroc_ci95"] = bootstrap_auroc_ci(labels, logits, args.bootstrap, args.seed)
        results[name]["n"] = int(len(labels))

    # Operating point is chosen on VAL ONLY, then applied unchanged to the test sets.
    v_logits, v_labels = cache["val"]
    thr = youden_threshold(v_labels, 1 / (1 + np.exp(-v_logits)))
    for name, (logits, labels) in cache.items():
        results[name]["at_val_youden_threshold"] = threshold_report(labels, 1 / (1 + np.exp(-logits)), thr)
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.getcwd(), help="Folder containing ARTSS/ (default: cwd)")
    p.add_argument("--processed-dir", default=None, help="Default: <root>/ARTSS/data/processed")
    p.add_argument("--models-dir", default=None, help="Default: <root>/ARTSS/models")
    p.add_argument("--backup-dir", default=None, help="Optional (e.g. Google Drive) folder to mirror checkpoints to")
    p.add_argument("--tag", default=None, help="Checkpoint name prefix (default: dr_<backbone>_<image-size>)")
    p.add_argument("--backbone", default="efficientnet_b2")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--min-lr-ratio", type=float, default=0.01, help="Final LR as a fraction of --lr")
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--clip", type=float, default=1.0, help="Max gradient norm")
    p.add_argument("--drop-rate", type=float, default=0.3)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.998, help="0 disables EMA")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    p.add_argument("--balanced-sampler", action="store_true", help="Use a balanced sampler (then consider pos_weight=1)")
    p.add_argument("--no-pos-weight", action="store_true", help="Use plain BCE (use with --balanced-sampler)")
    p.add_argument("--patience", type=int, default=6, help="Early-stop after this many epochs without val AUROC gain")
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--max-skips", type=int, default=20, help="Max non-finite steps tolerated per epoch")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for AUROC CI (0 disables)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Resume from <tag>_last.pt (models-dir, else backup-dir)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    set_seed(args.seed)

    root = Path(args.root).resolve()
    processed = Path(args.processed_dir).resolve() if args.processed_dir else root / "ARTSS" / "data" / "processed"
    models_dir = Path(args.models_dir).resolve() if args.models_dir else root / "ARTSS" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else None
    tag = args.tag or f"dr_{args.backbone}_{args.image_size}"
    best_path, last_path = models_dir / f"{tag}_best.pth", models_dir / f"{tag}_last.pt"
    history_path = models_dir / f"{tag}_history.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp(args.amp, device)
    log.info("Device: %s | autocast: %s | tag: %s", device, amp_dtype, tag)

    loaders = get_dataloaders(processed, image_size=args.image_size, batch_size=args.batch_size,
                              num_workers=args.num_workers, seed=args.seed, balanced_sampler=args.balanced_sampler)
    info = loaders.info
    log.info("Counts %s | positive rate %s", info["counts"], info["positive_rate"])
    pos_weight_val = 1.0 if args.no_pos_weight else float(info["pos_weight"])
    log.info("pos_weight: %.3f", pos_weight_val)

    model = FundusClassifier(args.backbone, True, args.drop_rate, args.drop_path_rate).to(device)
    model = model.to(memory_format=torch.channels_last)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    eval_model = ema.module if ema is not None else model

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loaders.train)
    scheduler = make_scheduler(optimizer, args.epochs * steps_per_epoch,
                               int(args.warmup_epochs * steps_per_epoch), args.min_lr_ratio)
    scaler = GradScaler(device.type) if amp_dtype == torch.float16 else None

    start_epoch, best_auroc, best_epoch, bad_epochs, history = 0, -1.0, -1, 0, []
    if args.resume:
        src = next((p for p in (last_path, (backup_dir / last_path.name) if backup_dir else None)
                    if p is not None and p.is_file()), None)
        if src is None:
            log.warning("--resume given but no %s found; starting fresh.", last_path.name)
        else:
            ckpt = torch.load(src, map_location=device, weights_only=False)
            if ckpt["config"].get("epochs") != args.epochs:
                log.warning("Resuming with --epochs %d but checkpoint was created with %s; LR schedule will differ.",
                            args.epochs, ckpt["config"].get("epochs"))
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if ema is not None and ckpt.get("ema") is not None:
                ema.module.load_state_dict(ckpt["ema"])
                ema.updates = ckpt["ema_updates"]
            if scaler is not None and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch, best_auroc, best_epoch = ckpt["epoch"] + 1, ckpt["best_auroc"], ckpt["best_epoch"]
            bad_epochs, history = ckpt["bad_epochs"], ckpt["history"]
            if best_path.is_file() is False and backup_dir and (backup_dir / best_path.name).is_file():
                shutil.copy2(backup_dir / best_path.name, best_path)
            log.info("Resumed from %s at epoch %d (best val AUROC %.4f @ epoch %d)",
                     src, start_epoch + 1, best_auroc, best_epoch + 1)

    config = {**vars(args), "pos_weight": pos_weight_val, "amp_dtype": str(amp_dtype), "data_info": info}

    log.info("Starting training: %d epochs x %d steps", args.epochs, steps_per_epoch)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_auroc, skipped = train_one_epoch(model, ema, loaders.train, optimizer, scheduler, criterion,
                                                           scaler, amp_dtype, device, args.clip, args.max_skips)
        v_logits, v_labels = predict(eval_model, loaders.val, amp_dtype, device)
        val = compute_metrics(v_logits, v_labels, pos_weight_val)

        improved = val["auroc"] > best_auroc + args.min_delta
        if improved:
            best_auroc, best_epoch, bad_epochs = val["auroc"], epoch, 0
            atomic_save(eval_model.state_dict(), best_path)
            backup_file(best_path, backup_dir)
        else:
            bad_epochs += 1

        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
               "train_auroc": train_auroc, "val_auroc": val["auroc"], "val_auprc": val["auprc"],
               "val_bce": val["bce"], "val_weighted_bce": val["weighted_bce"], "skipped_steps": skipped,
               "seconds": time.time() - t0}
        history.append(row)
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)

        log.info("Epoch %02d/%d | %.0fs | lr %.2e | train loss %.4f auroc %.4f | val auroc %.4f auprc %.4f bce %.4f%s%s",
                 epoch + 1, args.epochs, row["seconds"], row["lr"], train_loss, train_auroc, val["auroc"],
                 val["auprc"], val["bce"], f" | skipped {skipped}" if skipped else "",
                 "  * best" if improved else f"  (no gain {bad_epochs}/{args.patience})")

        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                     "ema": ema.module.state_dict() if ema else None, "ema_updates": ema.updates if ema else 0,
                     "epoch": epoch, "best_auroc": best_auroc, "best_epoch": best_epoch,
                     "bad_epochs": bad_epochs, "history": history, "config": config}, last_path)
        backup_file(last_path, backup_dir)
        backup_file(history_path, backup_dir)

        if bad_epochs >= args.patience:
            log.info("Early stopping: no val AUROC improvement for %d epochs.", args.patience)
            break

    if best_epoch < 0:
        raise RuntimeError("No valid best checkpoint was produced.")
    log.info("Training done. Best val AUROC %.4f at epoch %d. Evaluating best checkpoint...", best_auroc, best_epoch + 1)

    best_model = FundusClassifier(args.backbone, False, 0.0, 0.0).to(device).to(memory_format=torch.channels_last)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    results = final_evaluation(best_model, loaders, amp_dtype, device, pos_weight_val, args, models_dir, tag)

    for name, r in results.items():
        t = r["at_val_youden_threshold"]
        ci = r["auroc_ci95"]
        log.info("%-14s n=%5d | AUROC %.4f %s | AUPRC %.4f | sens %.3f spec %.3f @ thr %.3f", name, r["n"], r["auroc"],
                 f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "", r["auprc"], t["sensitivity"] or float("nan"),
                 t["specificity"] or float("nan"), t["threshold"])

    out = {"best_epoch": best_epoch + 1, "best_val_auroc": best_auroc, "results": results, "config": config}
    results_path = models_dir / f"{tag}_results.json"
    results_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    backup_file(results_path, backup_dir)
    for name in ("val", "test_internal", "test_external"):
        backup_file(models_dir / f"{tag}_logits_{name}.npz", backup_dir)
    log.info("Saved %s", results_path)
    return 0


if __name__ == "__main__":  # required on Windows when num_workers > 0
    sys.exit(main())