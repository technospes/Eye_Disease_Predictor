#!/usr/bin/env python3
"""
preprocess_fundus.py - Retinal fundus preprocessing for ARTSS (Phase 1).

For every image found under the dataset folders this script:
  1. drops byte-identical files (SHA-256), across ALL datasets together;
  2. crops the black camera border, pads to a square (no aspect distortion),
     resizes, and optionally applies Ben Graham local-colour subtraction;
  3. FLAGS near-duplicates (perceptual hash) as groups instead of deleting them,
     so your train/val/test splitter can keep each group on one side;
  4. writes manifest.csv (one row per source file) and preprocess_config.json.

Usage (from the folder that contains ARTSS/):
    python preprocess_fundus.py
    python preprocess_fundus.py --size 512 --workers 4
    python preprocess_fundus.py --dataset ODIR=path/to/odir --dataset APTOS=path/to/aptos --out path/to/out
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    tqdm = None

log = logging.getLogger("fundus")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_EXCLUDE = {"preprocessed_images", "__macosx"}
MIN_RETINA_FRACTION = 0.05  # below this, we assume no retina was found
PHASH_SIZE = 16  # 16x16 = 256-bit hash (8x8 is too coarse for fundus images)
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

MANIFEST_FIELDS = [
    "dataset", "source_rel", "source_sha256", "status", "duplicate_of", "error",
    "processed_rel", "orig_width", "orig_height",
    "crop_x0", "crop_y0", "crop_x1", "crop_y1",
    "retina_area_frac", "brightness", "blur_score",
    "phash", "near_dup_group", "near_dup_group_size",
]


@dataclass(frozen=True)
class Config:
    """Everything here changes the pixels written to disk."""
    size: int = 512
    border_thresh: int = 10
    sigma_ratio: float = 1 / 30  # Ben Graham: sigma = retina_radius / 30
    mask_ratio: float = 0.9  # keep the inner 90% of the retina radius
    enhance: bool = True

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class Item:
    dataset: str
    src: str
    rel: str


# --------------------------------------------------------------------------- #
# Image processing
# --------------------------------------------------------------------------- #
def read_image(path: str) -> np.ndarray:
    """cv2.imread fails on non-ASCII paths on Windows; imdecode does not."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError("empty file")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("undecodable image")
    return img


def find_retina_bbox(gray: np.ndarray, base_thresh: int):
    """Bounding box of the bright retina disc, robust to JPEG noise in the border.

    Returns (x0, y0, x1, y1, mask) or None if no retina is found.
    """
    # Adapt the threshold for very dark photos so they are not rejected outright.
    thresh = min(base_thresh, max(2, int(np.percentile(gray, 99) * 0.15)))
    mask = (cv2.GaussianBlur(gray, (5, 5), 0) > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if mask.mean() < MIN_RETINA_FRACTION:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1, mask


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def circle_mask(size: int, ratio: float) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), int(size / 2 * ratio), 1, -1)
    return m.astype(bool)


def ben_graham(img: np.ndarray, sigma_ratio: float, inner: np.ndarray) -> np.ndarray:
    """Local colour subtraction; sigma scales with image size, outside-circle set to grey."""
    radius = img.shape[0] / 2
    sigma = max(1.0, radius * sigma_ratio)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, 4, blurred, -4, 128)
    out[~inner] = 128
    return out


def write_png_atomic(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OSError("PNG encoding failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    os.replace(tmp, path)  # never leaves a half-written .png behind


def _init_worker() -> None:
    cv2.setNumThreads(1)  # avoid oversubscription when using a process pool


def process_one(task: tuple) -> dict:
    """Runs in a worker process. Never raises: failures are returned as records."""
    idx, src, out_path, cfg_d = task
    cfg = Config(**cfg_d)
    res: dict = {"idx": idx, "status": "ok", "error": ""}
    try:
        img = read_image(src)
        h, w = img.shape[:2]
        res["orig_width"], res["orig_height"] = w, h
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        det = find_retina_bbox(gray, cfg.border_thresh)
        if det is None:
            res["status"], res["error"] = "no_retina", "no retina disc detected"
            return res
        x0, y0, x1, y1, mask = det
        res.update(crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1,
                   retina_area_frac=round(float(mask.mean()), 4),
                   brightness=round(float(gray[mask > 0].mean()), 2))

        std = resize_square(pad_to_square(img[y0:y1, x0:x1]), cfg.size)
        std_gray = cv2.cvtColor(std, cv2.COLOR_BGR2GRAY)
        inner = circle_mask(cfg.size, cfg.mask_ratio)

        # Sharpness inside the retina, on the standardised image (comparable across files).
        lap = cv2.Laplacian(std_gray, cv2.CV_64F)
        res["blur_score"] = round(float(lap[inner].var()), 2)

        # Hash the un-enhanced standardised image so rescaled copies still match.
        bits = imagehash.phash(Image.fromarray(std_gray), hash_size=PHASH_SIZE).hash.flatten()
        res["phash"] = np.packbits(bits).tobytes().hex()

        out = ben_graham(std, cfg.sigma_ratio, inner) if cfg.enhance else std
        write_png_atomic(Path(out_path), out)
    except Exception as exc:  # noqa: BLE001 - a worker must report, not crash the pool
        res["status"], res["error"] = "error", f"{type(exc).__name__}: {exc}"
    return res


# --------------------------------------------------------------------------- #
# Dataset plumbing
# --------------------------------------------------------------------------- #
def discover(root: Path, exclude: set[str]) -> list[Path]:
    found: list[Path] = []
    for cur, dirs, files in os.walk(root):
        skipped = [d for d in dirs if d.lower() in exclude]
        for d in skipped:
            log.warning("Skipping excluded folder: %s", Path(cur) / d)
        dirs[:] = sorted(d for d in dirs if d.lower() not in exclude and not d.startswith("."))
        for name in sorted(files):
            if not name.startswith(".") and Path(name).suffix.lower() in IMAGE_EXTS:
                found.append(Path(cur) / name)
    return found


def sha256_file(path: str) -> tuple[str | None, str]:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return h.hexdigest(), ""


def new_record(item: Item) -> dict:
    rec = {k: "" for k in MANIFEST_FIELDS}
    rec.update(dataset=item.dataset, source_rel=item.rel)
    return rec


def group_near_duplicates(records: list[dict], max_bits: int) -> dict:
    """Union-find over perceptual hashes; adds near_dup_group(+size) to ok records."""
    ok = [r for r in records if r["status"] == "ok"]
    n = len(ok)
    if n == 0:
        return {"groups_gt1": 0, "largest": 0}
    packed = np.stack([np.frombuffer(bytes.fromhex(r["phash"]), dtype=np.uint8) for r in ok])
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n - 1):
        dist = POPCOUNT[packed[i + 1:] ^ packed[i]].sum(axis=1)
        for j in np.flatnonzero(dist <= max_bits):
            ra, rb = find(i), find(i + 1 + int(j))
            if ra != rb:
                parent[rb] = ra

    ids: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for i in range(n):
        root = find(i)
        ids.setdefault(root, len(ids))
        sizes[root] = sizes.get(root, 0) + 1
    for i, rec in enumerate(ok):
        root = find(i)
        rec["near_dup_group"] = ids[root]
        rec["near_dup_group_size"] = sizes[root]
    big = [s for s in sizes.values() if s > 1]
    return {"groups_gt1": len(big), "largest": max(sizes.values())}


def write_manifest(path: Path, records: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def check_config_guard(out_dir: Path, cfg: Config, force: bool) -> None:
    """Refuse to mix outputs made with different settings in one folder."""
    path = out_dir / "preprocess_config.json"
    if not path.exists() or force:
        return
    try:
        old = json.loads(path.read_text(encoding="utf-8")).get("config_digest")
    except (json.JSONDecodeError, OSError):
        old = None
    if old != cfg.digest():
        raise SystemExit(
            f"{out_dir} already holds outputs made with different settings.\n"
            "Use a new --out folder, or pass --force to overwrite."
        )


def setup_logging(out_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(out_dir / "preprocess.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def progress(iterable, total: int, desc: str):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="img")
    return iterable


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.getcwd(), help="Folder containing ARTSS/ (default: cwd)")
    p.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH",
                   help="Dataset folder, repeatable. Default: ODIR and APTOS under ARTSS/data/raw")
    p.add_argument("--out", default=None, help="Output folder (default: <root>/ARTSS/data/processed)")
    p.add_argument("--size", type=int, default=512, help="Output side length in px (default 512)")
    p.add_argument("--border-thresh", type=int, default=10, help="Gray level below which pixels count as border")
    p.add_argument("--no-enhance", action="store_true", help="Save cropped/resized images without Ben Graham")
    p.add_argument("--near-dup-bits", type=int, default=10,
                   help="Max Hamming distance (of 256 bits) to group near-duplicates")
    p.add_argument("--exclude", action="append", default=[],
                   help="Folder names to skip (default skips preprocessed_images)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--force", action="store_true", help="Overwrite outputs made with different settings")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else root / "ARTSS" / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    if args.dataset:
        datasets = []
        for spec in args.dataset:
            name, _, path = spec.partition("=")
            if not name or not path:
                raise SystemExit(f"--dataset must look like NAME=PATH, got: {spec!r}")
            datasets.append((name, Path(path).resolve()))
    else:
        raw = root / "ARTSS" / "data" / "raw"
        datasets = [("ODIR", raw / "ODIR-5k"), ("APTOS", raw / "APTOS-2019")]

    cfg = Config(size=args.size, border_thresh=args.border_thresh, enhance=not args.no_enhance)
    check_config_guard(out_dir, cfg, args.force)
    exclude = DEFAULT_EXCLUDE | {e.lower() for e in args.exclude}

    config_payload = {
        "config": asdict(cfg), "config_digest": cfg.digest(),
        "near_dup_max_bits": args.near_dup_bits, "phash_size": PHASH_SIZE,
        "datasets": {n: str(p) for n, p in datasets},
        "versions": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                     "numpy": np.__version__, "imagehash": imagehash.__version__},
        "started_utc": datetime.now(timezone.utc).isoformat(), "state": "in_progress",
    }
    write_json_atomic(out_dir / "preprocess_config.json", config_payload)

    # 1. Discover files -------------------------------------------------------
    items: list[Item] = []
    for name, ds_root in datasets:
        if not ds_root.is_dir():
            log.error("Dataset folder not found, skipping %s: %s", name, ds_root)
            continue
        files = discover(ds_root, exclude)
        log.info("%s: found %d images under %s", name, len(files), ds_root)
        items.extend(Item(name, str(f), f.relative_to(ds_root).as_posix()) for f in files)
    if not items:
        log.error("No images found. Check the dataset paths.")
        return 2
    records = [new_record(it) for it in items]

    # 2. Exact de-duplication across ALL datasets ------------------------------
    with ThreadPoolExecutor(max_workers=max(2, args.workers * 2)) as pool:
        hashes = list(progress(pool.map(lambda it: sha256_file(it.src), items), len(items), "hashing"))
    seen: dict[str, str] = {}
    to_process: list[int] = []
    for idx, (sha, err) in enumerate(hashes):
        rec = records[idx]
        if sha is None:
            rec["status"], rec["error"] = "unreadable", err
            continue
        rec["source_sha256"] = sha
        label = f"{items[idx].dataset}/{items[idx].rel}"
        if sha in seen:
            rec["status"], rec["duplicate_of"] = "duplicate_exact", seen[sha]
        else:
            seen[sha] = label
            to_process.append(idx)

    # 3. Preprocess unique files in parallel -----------------------------------
    used_out: set[Path] = set()
    tasks = []
    for idx in to_process:
        it = items[idx]
        out_path = out_dir / it.dataset.lower() / Path(it.rel).with_suffix(".png")
        if out_path in used_out:  # e.g. a.jpg and a.png side by side
            out_path = out_path.with_name(f"{out_path.stem}_{records[idx]['source_sha256'][:8]}.png")
        used_out.add(out_path)
        records[idx]["processed_rel"] = out_path.relative_to(out_dir).as_posix()
        tasks.append((idx, it.src, str(out_path), asdict(cfg)))

    log.info("Processing %d unique images with %d worker(s)", len(tasks), args.workers)
    if args.workers <= 1:
        results = map(process_one, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
        results = pool.map(process_one, tasks, chunksize=8)
    try:
        for res in progress(results, len(tasks), "processing"):
            idx = res.pop("idx")
            records[idx].update(res)
            if res["status"] != "ok":
                records[idx]["processed_rel"] = ""
                log.warning("%s: %s (%s)", res["status"], items[idx].rel, res["error"])
    finally:
        if pool is not None:
            pool.shutdown()

    # 4. Sanity check + near-duplicate groups ----------------------------------
    missing = [r["processed_rel"] for r in records
               if r["status"] == "ok" and not (out_dir / r["processed_rel"]).is_file()]
    if missing:
        log.error("%d processed files are missing on disk, e.g. %s", len(missing), missing[0])
    stats = group_near_duplicates(records, args.near_dup_bits)
    n_ok = sum(r["status"] == "ok" for r in records)
    if n_ok and stats["largest"] > max(20, 0.01 * n_ok):
        log.warning("Largest near-duplicate group has %d images; the threshold may be too loose "
                    "(lower --near-dup-bits).", stats["largest"])

    # 5. Outputs --------------------------------------------------------------
    write_manifest(out_dir / "manifest.csv", records)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    config_payload.update(state="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                          counts=counts, near_dup=stats)
    write_json_atomic(out_dir / "preprocess_config.json", config_payload)

    log.info("Done. Status counts: %s", counts)
    log.info("Near-duplicate groups (size>1): %d, largest: %d", stats["groups_gt1"], stats["largest"])
    log.info("Manifest: %s", out_dir / "manifest.csv")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())