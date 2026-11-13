#!/usr/bin/env python3
"""
make_splits.py - Persistent, leakage-checked data splits for ARTSS (Milestone 2).

Reads   <processed>/training_labels.csv    (from build_training_labels.py)
Writes  <processed>/splits.csv             training_labels + a `split` column
        <processed>/splits_meta.json       seed, counts, input hash, leakage checks

Design
  * ODIR  -> train / val / test_internal   (grouped by split_group, stratified by dr_label)
  * APTOS -> test_external                 (never used for training, early stopping or calibration)
  * APTOS images that share a split_group with an ODIR image (cross-dataset near-duplicates)
    are REMOVED from the external test set; otherwise "external" results are inflated.
  * Splits are written to disk once, so training, calibration, thresholding and Grad-CAM all
    use exactly the same images. Re-running refuses to overwrite them unless you pass --force.

Usage (from the folder that contains ARTSS/):
    python make_splits.py
    python make_splits.py --seed 7 --folds 7 --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger("splits")

REQUIRED = {"dataset", "patient_id", "split_group", "dr_label", "processed_rel", "source_rel", "near_dup_group"}
SPLIT_ORDER = ["train", "val", "test_internal", "test_external"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_labels(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Not found: {path}\nRun build_training_labels.py first.")
    df = pd.read_csv(path, encoding="utf-8")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name} is missing columns {sorted(missing)}; re-run build_training_labels.py.")
    if df["dr_label"].isna().any() or not df["dr_label"].isin([0, 1]).all():
        raise SystemExit("dr_label must be 0/1 with no missing values.")
    if not df["processed_rel"].is_unique:
        raise SystemExit("training_labels.csv contains the same processed image more than once.")
    unknown = sorted(set(df["dataset"]) - {"ODIR", "APTOS"})
    if unknown:
        raise SystemExit(f"Unexpected dataset names {unknown}; expected ODIR and APTOS.")
    return df


def grouped_stratified_split(df: pd.DataFrame, n_splits: int, seed: int):
    """One fold of a stratified, group-aware K-fold: returns (rest_positions, held_out_positions)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = df["dr_label"].to_numpy()
    groups = df["split_group"].to_numpy()
    return next(iter(sgkf.split(np.zeros(len(df)), labels, groups=groups)))


def build_splits(df: pd.DataFrame, folds: int, seed: int) -> tuple[pd.DataFrame, dict]:
    odir = df[df["dataset"] == "ODIR"].reset_index(drop=True)
    aptos = df[df["dataset"] == "APTOS"].reset_index(drop=True)
    if odir.empty or aptos.empty:
        raise SystemExit("Need both ODIR (train/val/test) and APTOS (external test) rows.")

    # Hold out ~1/folds of ODIR for test_internal, then ~1/folds of the rest for val
    # (folds=7 -> roughly 71% / 14% / 14%).
    rest_pos, test_pos = grouped_stratified_split(odir, folds, seed)
    test_df = odir.iloc[test_pos]
    rest = odir.iloc[rest_pos].reset_index(drop=True)
    train_pos, val_pos = grouped_stratified_split(rest, folds - 1, seed + 1)
    train_df, val_df = rest.iloc[train_pos], rest.iloc[val_pos]

    leaking_groups = set(odir["split_group"]) & set(aptos["split_group"])
    external = aptos[~aptos["split_group"].isin(leaking_groups)]
    removed = int(len(aptos) - len(external))
    if removed:
        log.warning("Removed %d APTOS images from the external test set: they share a near-duplicate/"
                    "patient group with ODIR training-pool images.", removed)

    parts = [train_df.assign(split="train"), val_df.assign(split="val"),
             test_df.assign(split="test_internal"), external.assign(split="test_external")]
    out = pd.concat(parts, ignore_index=True)
    out["split"] = pd.Categorical(out["split"], SPLIT_ORDER, ordered=True)
    out = out.sort_values(["split", "source_rel"]).reset_index(drop=True)
    out["split"] = out["split"].astype(str)
    return out, {"aptos_removed_for_leakage": removed, "leaking_groups": len(leaking_groups)}


def check_leakage(df: pd.DataFrame) -> None:
    """Hard assertions: no patient, group or near-duplicate group may appear in two splits."""
    for col in ("split_group", "patient_id", "near_dup_group"):
        sub = df.dropna(subset=[col])
        spread = sub.groupby(col)["split"].nunique()
        bad = spread[spread > 1]
        if len(bad):
            raise AssertionError(f"LEAKAGE: {len(bad)} values of {col} appear in several splits, e.g. {list(bad.index[:3])}")
    if not df["processed_rel"].is_unique:
        raise AssertionError("The same image appears twice in the splits.")
    if (df.loc[df["split"].isin(["train", "val", "test_internal"]), "dataset"] != "ODIR").any():
        raise AssertionError("Non-ODIR image found in the internal splits.")
    if (df.loc[df["split"] == "test_external", "dataset"] != "APTOS").any():
        raise AssertionError("Non-APTOS image found in the external test split.")


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for name in SPLIT_ORDER:
        sub = df[df["split"] == name]
        out[name] = {
            "images": int(len(sub)),
            "positives": int(sub["dr_label"].sum()),
            "positive_rate": round(float(sub["dr_label"].mean()), 4) if len(sub) else None,
            "split_groups": int(sub["split_group"].nunique()),
            "patients": int(sub["patient_id"].nunique()),
        }
    return out


def write_atomic(write_fn, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.getcwd(), help="Folder containing ARTSS/ (default: cwd)")
    p.add_argument("--processed-dir", default=None, help="Default: <root>/ARTSS/data/processed")
    p.add_argument("--folds", type=int, default=7, help="Held-out fraction is ~1/folds for each of val and test_internal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite existing splits.csv")
    args = p.parse_args(argv)
    if args.folds < 3:
        raise SystemExit("--folds must be at least 3")

    root = Path(args.root).resolve()
    processed = Path(args.processed_dir).resolve() if args.processed_dir else root / "ARTSS" / "data" / "processed"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)

    labels_path = processed / "training_labels.csv"
    splits_path = processed / "splits.csv"
    meta_path = processed / "splits_meta.json"
    if splits_path.exists() and not args.force:
        raise SystemExit(f"{splits_path} already exists. Splits must stay fixed once models are trained on them.\n"
                         "Pass --force only if you deliberately want new splits (and will retrain).")

    df = load_labels(labels_path)
    splits, extra = build_splits(df, args.folds, args.seed)
    check_leakage(splits)
    stats = summarize(splits)

    empty = [n for n, s in stats.items() if s["images"] == 0]
    if empty:
        raise SystemExit(f"Empty split(s): {empty}. Not enough data/groups for --folds {args.folds}.")
    overall = float(splits.loc[splits["split"].isin(["train", "val", "test_internal"]), "dr_label"].mean())
    for name in ("train", "val", "test_internal"):
        rate = stats[name]["positive_rate"]
        if abs(rate - overall) > 0.03:
            log.warning("%s positive rate %.3f differs from ODIR overall %.3f by more than 3 points.", name, rate, overall)

    write_atomic(lambda t: splits.to_csv(t, index=False, encoding="utf-8"), splits_path)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "folds": args.folds,
        "labels_file": labels_path.name, "labels_sha256": sha256_file(labels_path),
        "splits": stats, **extra,
        "leakage_checks": "passed: split_group, patient_id, near_dup_group each live in exactly one split",
    }
    write_atomic(lambda t: Path(t).write_text(json.dumps(meta, indent=2), encoding="utf-8"), meta_path)

    log.info("Splits written to %s", splits_path)
    for name in SPLIT_ORDER:
        s = stats[name]
        log.info("  %-14s %6d images | %5d positive (%.1f%%) | %5d groups", name, s["images"], s["positives"],
                 100 * s["positive_rate"], s["split_groups"])
    log.info("Leakage checks passed. ODIR prevalence %.1f%% vs APTOS %.1f%% - expect prevalence shift on the "
             "external test set.", 100 * overall, 100 * stats["test_external"]["positive_rate"])
    return 0


if __name__ == "__main__":
    sys.exit(main())