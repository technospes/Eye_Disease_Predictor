#!/usr/bin/env python3
"""
build_training_labels.py - Join the preprocessing manifest with clinical labels (ARTSS, Phase 1).

Reads   <processed>/manifest.csv   (written by preprocess_fundus.py)
        + the raw label files (ODIR full_df.csv, APTOS *.csv)
Writes  <processed>/training_labels.csv    one row per usable, labelled image
        <processed>/label_audit.json       what matched, what did not, and why
        <processed>/label_conflicts.csv    images excluded because labels disagree

Key points
  * Nothing is dropped silently: unmatched images/labels, ambiguous filenames and
    conflicting labels are all counted in label_audit.json.
  * `split_group` merges patient identity AND near-duplicate groups, so one group
    can safely be assigned to a single train/val/test split (no leakage).
  * ODIR eye-level labels are preferred over the patient-level 'D' column.

Usage (from the folder that contains ARTSS/):
    python build_training_labels.py
    python build_training_labels.py --strict
    python build_training_labels.py --aptos-csv path/train.csv --aptos-csv path/valid.csv
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("labels")

MANIFEST_REQUIRED = {
    "dataset", "source_rel", "source_sha256", "status", "duplicate_of", "processed_rel",
    "retina_area_frac", "brightness", "blur_score", "near_dup_group",
}
LABEL_COLS = ["patient_id", "eye_side", "age", "sex", "dr_label", "dr_grade", "label_source"]
OUT_COLUMNS = [
    "dataset", "patient_id", "eye_side", "age", "sex", "dr_label", "dr_grade", "label_source",
    "source_rel", "processed_rel", "retina_area_frac", "brightness", "blur_score",
    "near_dup_group", "split_group",
]
EXAMPLES = 10  # how many example names to put in the audit file


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def odir_key(name: str) -> str:
    return Path(str(name)).name.strip().lower()


def aptos_key(name: str) -> str:
    return Path(str(name)).stem.strip().lower()


def read_csv_checked(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    try:
        return pd.read_csv(path, **kwargs)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise SystemExit(f"Could not parse {path}: {exc}")


def col_or_na(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def write_json_atomic(payload: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def parse_label_list(value):
    """ODIR stores per-eye labels as a string like "['N']"."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        out = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    return list(out) if isinstance(out, (list, tuple)) else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}\nRun preprocess_fundus.py first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = MANIFEST_REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest is missing columns {sorted(missing)}; re-run preprocess_fundus.py.")
    for col in ("retina_area_frac", "brightness", "blur_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["near_dup_group"] = pd.to_numeric(df["near_dup_group"], errors="coerce").astype("Int64")
    df["uid"] = df["dataset"] + "/" + df["source_rel"]
    return df


def dedupe_labels(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list[str]]:
    """One row per key. Keys whose duplicate rows disagree are dropped and reported."""
    n_labels = df.groupby("key")["dr_label"].nunique(dropna=False)
    bad = sorted(n_labels[n_labels > 1].index)
    if bad:
        log.warning("%s: %d image keys have conflicting labels in the label file(s); dropping them, e.g. %s",
                    name, len(bad), bad[:3])
        df = df[~df["key"].isin(bad)]
    return df.drop_duplicates("key", keep="first").reset_index(drop=True), bad


def load_odir(csv_path: Path, source: str) -> tuple[pd.DataFrame, list[str]]:
    df = read_csv_checked(csv_path, encoding_errors="replace")
    if "filename" not in df.columns:
        raise SystemExit(f"{csv_path} has no 'filename' column (columns: {list(df.columns)[:12]}...)")

    stem = df["filename"].astype(str).map(lambda x: Path(x).stem.strip().lower())
    from_name = stem.str.split("_").str[0]
    if "ID" in df.columns:
        num = pd.to_numeric(df["ID"], errors="coerce")
        pid = num.map(lambda v: str(int(v)) if pd.notna(v) else None).fillna(from_name)
        bad = int((pid != from_name).sum())
        if bad:
            log.warning("ODIR: %d rows where ID differs from the filename prefix; using ID.", bad)
    else:
        pid = from_name

    # Eye-level label from the 'labels' list; 'D' is patient-level in the original ODIR data.
    d_col = pd.to_numeric(df["D"], errors="coerce") if "D" in df.columns else None
    eye_dr = None
    if "labels" in df.columns:
        parsed = df["labels"].map(parse_label_list)
        if parsed.notna().mean() >= 0.95:
            eye_dr = parsed.map(lambda lst: np.nan if lst is None else float("D" in lst))
    if eye_dr is not None and d_col is not None:
        both = eye_dr.notna() & d_col.notna()
        if both.any():
            agree = float((eye_dr[both] == d_col[both]).mean())
            log.info("ODIR: per-eye 'labels' and 'D' column agree on %.1f%% of rows.", 100 * agree)
            if agree < 0.99:
                log.warning("ODIR: 'D' disagrees with the per-eye labels on %.1f%% of rows - it is "
                            "probably patient-level, so a healthy eye can inherit the other eye's disease.",
                            100 * (1 - agree))

    if source == "labels" and eye_dr is None:
        raise SystemExit("--odir-label-source labels requested but no usable 'labels' column found.")
    if source == "D" and d_col is None:
        raise SystemExit("--odir-label-source D requested but no 'D' column found.")
    use_eye = (source == "labels") or (source == "auto" and eye_dr is not None)
    if use_eye:
        dr, label_source = eye_dr, "odir_labels_per_eye"
    elif d_col is not None:
        log.warning("ODIR: using the 'D' column (patient-level, diabetes-related) as the DR label.")
        dr, label_source = d_col, "odir_D_patient_level"
    else:
        raise SystemExit(f"{csv_path}: need a 'labels' or 'D' column to build DR labels.")

    dr = dr.where(dr.isin([0, 1]))  # anything else becomes missing
    if dr.isna().any():
        log.warning("ODIR: %d rows have no usable 0/1 DR label.", int(dr.isna().sum()))

    out = pd.DataFrame({
        "key": df["filename"].map(odir_key),
        "patient_id": "ODIR_" + pid,
        "eye_side": stem.map(lambda s: "Left" if s.endswith("_left") else "Right" if s.endswith("_right") else ""),
        "age": pd.to_numeric(col_or_na(df, "Patient Age"), errors="coerce").astype("Int64"),
        "sex": col_or_na(df, "Patient Sex").fillna("").astype(str).str.strip(),
        "dr_label": dr.astype("Int64"),
        "dr_grade": pd.array([pd.NA] * len(df), dtype="Int64"),
        "label_source": label_source,
    })
    out = out[out["dr_label"].notna()]
    return dedupe_labels(out, "ODIR")


def find_aptos_csvs(folder: Path) -> list[Path]:
    found = []
    for p in sorted(folder.rglob("*.csv")):
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
            continue
        if {"id_code", "diagnosis"} <= cols:
            found.append(p)
    return found


def load_aptos(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    for p in paths:
        df = read_csv_checked(p)
        if not {"id_code", "diagnosis"} <= set(df.columns):
            log.warning("APTOS: %s lacks id_code/diagnosis columns, skipping.", p.name)
            continue
        grade = pd.to_numeric(df["diagnosis"], errors="coerce")
        valid = grade.isin([0, 1, 2, 3, 4])
        if (~valid).any():
            log.warning("APTOS: %s has %d rows with an invalid diagnosis, dropping them.", p.name, int((~valid).sum()))
        df, grade = df[valid].reset_index(drop=True), grade[valid].reset_index(drop=True)
        key = df["id_code"].astype(str).map(aptos_key)
        log.info("APTOS: %s -> %d labelled rows", p.name, len(df))
        frames.append(pd.DataFrame({
            "key": key,
            "patient_id": "APTOS_" + key,  # APTOS has no patient IDs: one image = one "patient"
            "eye_side": "",
            "age": pd.array([pd.NA] * len(df), dtype="Int64"),
            "sex": "",
            "dr_label": (grade > 0).astype("Int64"),  # any DR
            "dr_grade": grade.astype("Int64"),  # 0-4, keeps severity for referable-DR (>=2) later
            "label_source": "aptos_diagnosis",
        }))
    if not frames:
        raise SystemExit("No usable APTOS label CSV (needs id_code and diagnosis columns).")
    return dedupe_labels(pd.concat(frames, ignore_index=True), "APTOS")


# --------------------------------------------------------------------------- #
# Joining + auditing
# --------------------------------------------------------------------------- #
def attach(manifest: pd.DataFrame, labels: pd.DataFrame, key_fn, dataset: str,
           label_conflicts: list[str]) -> tuple[pd.DataFrame, dict]:
    """Left-join labels onto EVERY manifest row of one dataset and audit the result."""
    rows = manifest[manifest["dataset"] == dataset].copy().reset_index(drop=True)
    rows["key"] = rows["source_rel"].map(key_fn)
    merged = rows.merge(labels, on="key", how="left", validate="m:1")

    # Same filename but different bytes (e.g. two folders) -> we cannot tell which label belongs where.
    ambiguous = merged.groupby("key")["source_sha256"].transform("nunique") > 1
    for col in LABEL_COLS:
        merged.loc[ambiguous, col] = pd.NA
    merged["ambiguous_key"] = ambiguous

    usable = merged["status"].isin(["ok", "duplicate_exact"])
    ok = merged["status"] == "ok"
    labelled = merged["dr_label"].notna()
    missing_from_manifest = sorted(set(labels["key"]) - set(merged["key"]))
    audit = {
        "manifest_rows": int(len(merged)),
        "ok_images": int(ok.sum()),
        "ok_labelled": int((ok & labelled).sum()),
        "ok_unlabelled": int((ok & ~labelled & ~ambiguous).sum()),
        "unlabelled_examples": merged.loc[ok & ~labelled & ~ambiguous, "source_rel"].head(EXAMPLES).tolist(),
        "ambiguous_filenames": int(ambiguous.sum()),
        "ambiguous_examples": merged.loc[ambiguous, "source_rel"].head(EXAMPLES).tolist(),
        "label_rows": int(len(labels)),
        "labels_for_unusable_images": int((~usable & labelled).sum()),
        "labels_without_image": len(missing_from_manifest),
        "labels_without_image_examples": missing_from_manifest[:EXAMPLES],
        "conflicting_label_keys_dropped": len(label_conflicts),
    }
    return merged, audit


def find_conflicts(allrows: pd.DataFrame) -> dict[str, str]:
    """uid -> reason, for kept images whose label disagrees with a duplicate of themselves."""
    reasons: dict[str, str] = {}
    labelled_ok = allrows[(allrows["status"] == "ok") & allrows["dr_label"].notna()]
    label_of = dict(zip(labelled_ok["uid"], labelled_ok["dr_label"]))

    dups = allrows[(allrows["status"] == "duplicate_exact") & allrows["dr_label"].notna()]
    for dup_uid, kept_uid, lab in zip(dups["uid"], dups["duplicate_of"], dups["dr_label"]):
        if kept_uid in label_of and label_of[kept_uid] != lab:
            reasons.setdefault(kept_uid, f"byte-identical copy {dup_uid} is labelled {int(lab)}, this one {int(label_of[kept_uid])}")

    near = labelled_ok.dropna(subset=["near_dup_group"])
    n_labels = near.groupby("near_dup_group")["dr_label"].nunique()
    for gid in n_labels[n_labels > 1].index:
        members = near[near["near_dup_group"] == gid]
        for uid in members["uid"]:
            reasons.setdefault(uid, f"near-duplicate group {gid} contains images with different labels")
    return reasons


def assign_split_groups(df: pd.DataFrame) -> pd.Series:
    """Connected components over shared patient_id OR shared near_dup_group (union-find)."""
    parent = list(range(len(df)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first_seen: dict[tuple, int] = {}
    for i, (pid, grp) in enumerate(zip(df["patient_id"], df["near_dup_group"])):
        keys = [("p", pid)] + ([("d", int(grp))] if pd.notna(grp) else [])
        for k in keys:
            if k in first_seen:
                ra, rb = find(i), find(first_seen[k])
                if ra != rb:
                    parent[rb] = ra
            else:
                first_seen[k] = i
    ids: dict[int, str] = {}
    out = []
    for i in range(len(df)):
        root = find(i)
        ids.setdefault(root, f"G{len(ids):06d}")
        out.append(ids[root])
    return pd.Series(out, index=df.index)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.getcwd(), help="Folder containing ARTSS/ (default: cwd)")
    p.add_argument("--processed-dir", default=None, help="Folder with manifest.csv (default: <root>/ARTSS/data/processed)")
    p.add_argument("--raw-dir", default=None, help="Folder with raw datasets (default: <root>/ARTSS/data/raw)")
    p.add_argument("--odir-csv", default=None, help="ODIR full_df.csv (default: searched under raw/ODIR-5k)")
    p.add_argument("--aptos-csv", action="append", default=[],
                   help="APTOS label CSV, repeatable (default: every CSV under raw/APTOS-2019 with id_code+diagnosis)")
    p.add_argument("--odir-label-source", choices=["auto", "labels", "D"], default="auto",
                   help="auto = per-eye 'labels' column if present, else patient-level 'D'")
    p.add_argument("--keep-conflicts", action="store_true", help="Keep images whose labels conflict with a duplicate")
    p.add_argument("--strict", action="store_true", help="Exit non-zero on any unmatched, ambiguous or conflicting item")
    p.add_argument("--allow-partial", action="store_true",
                   help="Write outputs even if a dataset's label file is missing or matched nothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    processed = Path(args.processed_dir).resolve() if args.processed_dir else root / "ARTSS" / "data" / "processed"
    raw = Path(args.raw_dir).resolve() if args.raw_dir else root / "ARTSS" / "data" / "raw"

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    handlers = [logging.StreamHandler(sys.stdout)]
    if processed.is_dir():
        handlers.append(logging.FileHandler(processed / "build_labels.log", encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)

    manifest = load_manifest(processed / "manifest.csv")
    present = [d for d in ("ODIR", "APTOS") if (manifest["dataset"] == d).any()]
    unknown = sorted(set(manifest["dataset"]) - {"ODIR", "APTOS"})
    if unknown:
        log.warning("Manifest has datasets this script does not know how to label: %s", unknown)
    log.info("Manifest: %d rows; datasets with labels logic: %s", len(manifest), present)

    merged_parts, audit_by_ds, problems, fatal = [], {}, [], []
    for ds in present:
        if ds == "ODIR":
            csv_path = Path(args.odir_csv) if args.odir_csv else None
            if csv_path is None:
                hits = sorted((raw / "ODIR-5k").rglob("full_df.csv")) if (raw / "ODIR-5k").is_dir() else []
                csv_path = hits[0] if hits else None
            if csv_path is None:
                log.error("ODIR: could not find full_df.csv under %s; pass --odir-csv.", raw / "ODIR-5k")
                fatal.append("ODIR label file missing")
                continue
            labels, conflicts = load_odir(csv_path, args.odir_label_source)
            key_fn = odir_key
        else:
            paths = [Path(p) for p in args.aptos_csv] or (
                find_aptos_csvs(raw / "APTOS-2019") if (raw / "APTOS-2019").is_dir() else [])
            if not paths:
                log.error("APTOS: no label CSV with id_code+diagnosis found under %s; pass --aptos-csv.", raw / "APTOS-2019")
                fatal.append("APTOS label file missing")
                continue
            labels, conflicts = load_aptos(paths)
            key_fn = aptos_key
        merged, audit = attach(manifest, labels, key_fn, ds, conflicts)
        merged_parts.append(merged)
        audit_by_ds[ds] = audit
        log.info("%s: %d/%d usable images labelled; %d unlabelled; %d labels have no image; %d ambiguous filenames",
                 ds, audit["ok_labelled"], audit["ok_images"], audit["ok_unlabelled"],
                 audit["labels_without_image"], audit["ambiguous_filenames"])
        if audit["ok_images"] and audit["ok_labelled"] == 0:
            fatal.append(f"{ds}: zero images matched a label (check filename/ID formats)")
        if args.strict and (audit["ok_unlabelled"] or audit["labels_without_image"] or audit["ambiguous_filenames"]
                            or audit["conflicting_label_keys_dropped"]):
            problems.append(f"{ds}: unmatched/ambiguous/conflicting items (strict mode)")

    if fatal and not args.allow_partial:
        for msg in fatal:
            log.error("FATAL: %s", msg)
        log.error("Not writing any outputs, so an existing training_labels.csv stays untouched. "
                  "Fix the above, or pass --allow-partial.")
        return 1
    problems = fatal + problems
    if not merged_parts:
        log.error("Nothing to build: %s", problems)
        return 1
    allrows = pd.concat(merged_parts, ignore_index=True)

    # Conflicting labels between duplicates are unreliable ground truth -> exclude by default.
    ok = allrows[(allrows["status"] == "ok") & allrows["dr_label"].notna()].copy()
    reasons = find_conflicts(allrows)
    ok["label_conflict"] = ok["uid"].isin(reasons)
    n_conf = int(ok["label_conflict"].sum())
    if n_conf:
        conf_df = ok[ok["label_conflict"]][["uid", "dr_label"]].assign(reason=lambda d: d["uid"].map(reasons))
        write_csv_atomic(conf_df, processed / "label_conflicts.csv")
        log.warning("%d images have conflicting labels (see label_conflicts.csv); %s.", n_conf,
                    "keeping them (--keep-conflicts)" if args.keep_conflicts else "excluding them")
        if args.strict:
            problems.append(f"{n_conf} label conflicts (strict mode)")
    final = ok if args.keep_conflicts else ok[~ok["label_conflict"]]
    final = final.sort_values(["dataset", "source_rel"]).reset_index(drop=True)
    final["split_group"] = assign_split_groups(final)

    sizes = final["split_group"].value_counts()
    if len(final) and sizes.iloc[0] > max(50, 0.05 * len(final)):
        log.warning("Largest split group has %d images (%.1f%% of data); near-duplicate threshold may be too loose.",
                    sizes.iloc[0], 100 * sizes.iloc[0] / len(final))

    write_csv_atomic(final[OUT_COLUMNS], processed / "training_labels.csv")

    dist = final.groupby(["dataset", "dr_label"]).size()
    audit_payload = {
        "datasets": audit_by_ds,
        "conflicting_images": n_conf,
        "conflicts_excluded": not args.keep_conflicts,
        "final_rows": int(len(final)),
        "split_groups": int(final["split_group"].nunique()),
        "largest_split_group": int(sizes.iloc[0]) if len(sizes) else 0,
        "class_counts": {f"{d}/dr_label={int(l)}": int(n) for (d, l), n in dist.items()},
        "label_sources": sorted(final["label_source"].unique().tolist()),
        "problems": problems,
    }
    write_json_atomic(audit_payload, processed / "label_audit.json")

    log.info("Wrote %d labelled images in %d split groups -> %s", len(final), audit_payload["split_groups"],
             processed / "training_labels.csv")
    log.info("Class counts:\n%s", dist.to_string())
    log.info("Audit: %s", processed / "label_audit.json")
    for prob in problems:
        log.error("PROBLEM: %s", prob)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())