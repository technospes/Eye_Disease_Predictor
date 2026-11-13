#!/usr/bin/env python3
"""
eval_idrid.py - Zero-shot external benchmark of the ARTSS DR model on IDRiD.

Usage (from the project root):
    python ARTSS/src/eval_idrid.py --data-dir ARTSS/data/raw/IDRiD
    python ARTSS/src/eval_idrid.py --workers 8 --n-boot 2000
    python ARTSS/src/eval_idrid.py --csv "path/to/Training Labels.csv" --csv "path/to/Testing Labels.csv"

Binary task: DR grade 0 -> negative, grade 1..4 -> positive.
Outputs (next to the model bundle):
    idrid_external_benchmark.json   metrics with 95% CIs
    idrid_predictions.csv           per-image probabilities (for audit / re-analysis)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from tqdm import tqdm

from app import Engine, Settings, load_bundle

log = logging.getLogger("eval_idrid")

IMG_EXTS = {".jpg", ".jpeg", ".png"}
ID_CANDIDATES = ["image name", "id_code", "image_id", "id", "filename", "image"]
LABEL_CANDIDATES = ["retinopathy grade", "diagnosis", "dr_label", "label", "diabetic_retinopathy"]
SPLIT_TOKENS = ("train", "test")  # IDRiD reuses IDRiD_001.. in both splits -> must disambiguate


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def scan_tree(root: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    """Single os.walk pass. Images are keyed by lower-cased stem; values are ALL matches."""
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}. Extract the IDRiD archive there first.")
    csvs: list[Path] = []
    imgs: dict[str, list[Path]] = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            ext = p.suffix.lower()
            if ext == ".csv":
                csvs.append(p)
            elif ext in IMG_EXTS:
                imgs.setdefault(p.stem.lower(), []).append(p)
    if not imgs:
        raise FileNotFoundError(f"No fundus images found inside {root}")
    return sorted(csvs), imgs


def resolve_columns(columns: list[str]) -> tuple[str, str] | None:
    """Case-insensitive column match. Returns None instead of guessing."""
    lower = {c.strip().lower(): c for c in columns}
    id_col = next((lower[c] for c in ID_CANDIDATES if c in lower), None)
    lbl_col = next((lower[c] for c in LABEL_CANDIDATES if c in lower), None)
    return (id_col, lbl_col) if id_col and lbl_col else None


def load_labels(csv_paths: list[Path], root: Path) -> pd.DataFrame:
    frames = []
    for csv in csv_paths:
        df = pd.read_csv(csv, skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]
        cols = resolve_columns(list(df.columns))
        if cols is None:
            log.info("Skipping %s (no recognisable id/grade columns)", csv.name)
            continue
        id_col, lbl_col = cols
        out = pd.DataFrame(
            {
                "image_id": df[id_col].astype(str).str.strip(),
                "grade": pd.to_numeric(df[lbl_col], errors="coerce"),
            }
        )
        bad = int(out["grade"].isna().sum())
        if bad:
            log.warning("%s: dropping %d rows with non-numeric grade", csv.name, bad)
        out = out.dropna(subset=["grade"])
        out = out[out["grade"].between(0, 4)]
        out["grade"] = out["grade"].astype(int)
        out["csv_hint"] = str(csv.relative_to(root)).lower()
        frames.append(out)
        log.info("Labels: %s -> %d rows (cols: %s / %s)", csv.name, len(out), id_col, lbl_col)
    if not frames:
        raise FileNotFoundError("No CSV with a recognisable image-id and DR-grade column was found.")
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["image_id", "csv_hint"])
    if len(df) != before:
        log.warning("Dropped %d duplicate label rows", before - len(df))
    df["dr_binary"] = (df["grade"] > 0).astype(int)
    return df


def pick_image(cands: list[Path], csv_hint: str, root: Path) -> Path | None:
    """Resolve train/test stem collisions using the label-file name vs the image path."""
    if len(cands) == 1:
        return cands[0]
    scored = []
    for p in cands:
        rel = str(p.relative_to(root)).lower()
        s = 0
        for tok in SPLIT_TOKENS:
            if tok in csv_hint:
                s += 1 if tok in rel else -1
        scored.append((s, p))
    scored.sort(key=lambda t: t[0], reverse=True)
    if scored[0][0] == scored[1][0]:
        return None  # genuinely ambiguous -> never guess
    return scored[0][1]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric, n_boot: int, seed: int) -> list[float | None]:
    """Stratified percentile bootstrap (keeps both classes in every resample)."""
    if n_boot <= 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals[i] = metric(y[idx], p[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Zero-shot evaluation on the IDRiD dataset")
    ap.add_argument("--data-dir", default="ARTSS/data/raw/IDRiD", help="Root of the extracted IDRiD archive")
    ap.add_argument("--csv", action="append", default=None, help="Label CSV (repeatable). Default: auto-detect all")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="Inference threads (1 = sequential). Set 1 if your Engine is not thread-safe")
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for AUROC/AUPRC CIs (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-skip-frac", type=float, default=0.02,
                    help="Exit non-zero if more than this fraction of labelled images could not be scored")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    root = Path(args.data_dir).resolve()
    auto_csvs, img_index = scan_tree(root)
    csv_paths = [Path(c).resolve() for c in args.csv] if args.csv else auto_csvs
    if not csv_paths:
        raise FileNotFoundError(f"No .csv label file found inside {root}")
    n_imgs = sum(len(v) for v in img_index.values())
    log.info("Indexed %d images (%d unique stems) and %d CSV files", n_imgs, len(img_index), len(csv_paths))

    labels = load_labels(csv_paths, root)

    # ---- match labels -> image files (never guess on ambiguity) -------------
    skip_reasons: Counter[str] = Counter()
    tasks: list[tuple[int, Path]] = []
    for i, row in enumerate(labels.itertuples(index=False)):
        key = Path(row.image_id).stem.lower()
        cands = img_index.get(key)
        if not cands:
            skip_reasons["image_not_found"] += 1
            tasks.append((i, None))  # type: ignore[arg-type]
            continue
        path = pick_image(cands, row.csv_hint, root)
        if path is None:
            skip_reasons["ambiguous_match"] += 1
        tasks.append((i, path))  # type: ignore[arg-type]

    settings = Settings.from_env()
    bundle = load_bundle(settings)
    engine = Engine(settings, bundle)
    log.info("Model %s on %s | workers=%d", bundle.model_version, engine.device, args.workers)

    def infer_one(task):
        idx, path = task
        if path is None:
            return idx, None
        try:
            return idx, float(engine.infer(path.read_bytes(), want_cam=False)["dr_probability"])
        except Exception as exc:  # noqa: BLE001 - record, don't hide
            skip_reasons[f"infer_error:{type(exc).__name__}"] += 1
            log.debug("Inference failed for %s: %s", path, exc)
            return idx, None

    todo = [t for t in tasks if t[1] is not None]
    probs = np.full(len(labels), np.nan)
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for idx, pr in tqdm(pool.map(infer_one, todo), total=len(todo), desc="infer"):
                if pr is not None:
                    probs[idx] = pr
    else:
        for t in tqdm(todo, desc="infer"):
            idx, pr = infer_one(t)
            if pr is not None:
                probs[idx] = pr

    labels = labels.assign(dr_probability=probs)
    ok = labels["dr_probability"].notna()
    n_skipped = int((~ok).sum())
    scored = labels[ok]
    y_true = scored["dr_binary"].to_numpy()
    y_prob = scored["dr_probability"].to_numpy()

    if len(np.unique(y_true)) < 2:
        log.error("Only one class present among scored images (n=%d); AUROC undefined.", len(y_true))
        return 1

    # ---- metrics -------------------------------------------------------------
    thr = float(bundle.threshold)
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_prob >= thr, labels=[0, 1]).ravel())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    report = {
        "external_cohort": "IDRiD (Indian Diabetic Retinopathy Image Dataset)",
        "model_version": bundle.model_version,
        "evaluated_n": int(len(y_true)),
        "skipped_n": n_skipped,
        "skip_reasons": dict(skip_reasons),
        "positive_rate": round(float(y_true.mean()), 4),
        "auroc": round(auroc, 4),
        "auroc_95ci": bootstrap_ci(y_true, y_prob, roc_auc_score, args.n_boot, args.seed),
        "auprc": round(auprc, 4),
        "auprc_95ci": bootstrap_ci(y_true, y_prob, average_precision_score, args.n_boot, args.seed),
        "threshold_locked": round(thr, 4),
        "sensitivity": round(sens, 4),
        "sensitivity_95ci_wilson": wilson_ci(tp, tp + fn),
        "specificity": round(spec, 4),
        "specificity_95ci_wilson": wilson_ci(tn, tn + fp),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "grade_distribution": {str(k): int(v) for k, v in sorted(scored["grade"].value_counts().items())},
    }

    print("\n" + "=" * 55)
    print("IDRiD ZERO-SHOT EXTERNAL VALIDATION RESULTS")
    print("=" * 55)
    print(json.dumps(report, indent=2))

    # ---- atomic writes -------------------------------------------------------
    out_dir = Path(settings.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "idrid_external_benchmark.json"
    out_csv = out_dir / "idrid_predictions.csv"
    for path, writer in (
        (out_json, lambda p: p.write_text(json.dumps(report, indent=2), encoding="utf-8")),
        (out_csv, lambda p: labels.drop(columns=["csv_hint"]).to_csv(p, index=False)),
    ):
        tmp = path.with_suffix(path.suffix + ".tmp")
        writer(tmp)
        tmp.replace(path)
    log.info("Saved %s and %s", out_json, out_csv.name)

    if n_skipped / len(labels) > args.max_skip_frac:
        log.error("Skipped %d/%d images (> %.1f%% allowed). Reasons: %s",
                  n_skipped, len(labels), args.max_skip_frac * 100, dict(skip_reasons))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())