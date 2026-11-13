#!/usr/bin/env python3
"""
dataset.py - PyTorch Dataset and DataLoaders for ARTSS (Milestone 2).

Expects the files produced by the earlier steps in <processed>/:
    splits.csv, splits_meta.json (from make_splits.py), preprocess_config.json (from preprocess_fundus.py)
and the processed images referenced by `processed_rel`.

    from dataset import get_dataloaders
    loaders = get_dataloaders("ARTSS/data/processed", image_size=384, batch_size=32)
    loaders.train, loaders.val, loaders.test_internal, loaders.test_external
    loaders.info["pos_weight"]   # neg/pos ratio of the training split, for BCEWithLogitsLoss

Smoke test (from the folder that contains ARTSS/):
    python ARTSS/src/dataset.py --save-grid
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

log = logging.getLogger("dataset")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test_internal", "test_external")
BICUBIC = T.InterpolationMode.BICUBIC


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class FundusDataset(Dataset):
    """Loads processed fundus PNGs; returns (image_tensor, float label)."""

    def __init__(self, df: pd.DataFrame, processed_dir, transform=None,
                 label_col: str = "dr_label", check_files: bool = True):
        for col in ("processed_rel", label_col):
            if col not in df.columns:
                raise KeyError(f"DataFrame is missing column {col!r}")
        labels = df[label_col].to_numpy(dtype=np.float32)
        if np.isnan(labels).any():
            raise ValueError(f"{label_col} contains missing values")
        root = Path(processed_dir)
        # Plain arrays, not pandas rows: indexing a DataFrame per item is slow and not worker-friendly.
        self.paths = [str(root / rel) for rel in df["processed_rel"]]
        self.labels = labels
        self.transform = transform
        if check_files:
            missing = [p for p in self.paths if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(f"{len(missing)} image files are missing, e.g. {missing[:3]}. "
                                        "Re-run preprocess_fundus.py or check --processed-dir.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as im:  # closes the file handle
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.as_tensor(self.labels[idx])


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def read_preprocess_config(processed_dir: Path) -> dict:
    path = processed_dir / "preprocess_config.json"
    if not path.is_file():
        log.warning("%s not found; assuming Ben Graham-enhanced images (grey background).", path.name)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("config", {})
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s (%s); using defaults.", path.name, exc)
        return {}


def build_transforms(image_size: int, enhanced: bool, train: bool, stored_size: int | None = None):
    """Train: gentle geometric augmentation. Eval: resize + normalise only.

    * Rotation fill matches the preprocessing background (grey 128 after Ben Graham, black otherwise);
      the default fill of 0 would paint black corners the model never sees at test time.
    * Fundus photos have no canonical orientation, so flips and full rotation are label-preserving.
    * Brightness/contrast jitter is skipped for enhanced images: Ben Graham already removes illumination.
    """
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        steps = [] if stored_size == image_size else [T.Resize((image_size, image_size), interpolation=BICUBIC, antialias=True)]
        return T.Compose(steps + [T.ToTensor(), normalize])

    fill = (128, 128, 128) if enhanced else (0, 0, 0)
    steps = [
        T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05), interpolation=BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180, interpolation=T.InterpolationMode.BILINEAR, fill=fill),
    ]
    if not enhanced:
        steps.append(T.ColorJitter(brightness=0.15, contrast=0.15))
    return T.Compose(steps + [T.ToTensor(), normalize])


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test_internal: DataLoader
    test_external: DataLoader
    info: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_splits(processed_dir: Path) -> pd.DataFrame:
    splits_path = processed_dir / "splits.csv"
    if not splits_path.is_file():
        raise FileNotFoundError(f"{splits_path} not found. Run make_splits.py first.")
    df = pd.read_csv(splits_path, encoding="utf-8")

    meta_path, labels_path = processed_dir / "splits_meta.json", processed_dir / "training_labels.csv"
    if meta_path.is_file() and labels_path.is_file():
        recorded = json.loads(meta_path.read_text(encoding="utf-8")).get("labels_sha256")
        if recorded and recorded != _sha256(labels_path):
            raise RuntimeError("training_labels.csv changed after splits.csv was created. "
                               "Re-run make_splits.py --force (and retrain), or restore the old labels file.")
    unknown = set(df["split"]) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown split names in splits.csv: {sorted(unknown)}")
    return df


def default_num_workers() -> int:
    return min(4, max(0, (os.cpu_count() or 2) - 1))


def get_dataloaders(processed_dir, image_size: int = 384, batch_size: int = 32, num_workers: int | None = None,
                    seed: int = 42, balanced_sampler: bool = False, label_col: str = "dr_label") -> Loaders:
    processed_dir = Path(processed_dir).resolve()
    df = load_splits(processed_dir)
    cfg = read_preprocess_config(processed_dir)
    enhanced = bool(cfg.get("enhance", True))
    stored_size = cfg.get("size")
    if stored_size and image_size > stored_size:
        log.warning("image_size=%d is larger than the stored images (%d): pixels will be upsampled, not gained.",
                    image_size, stored_size)
    num_workers = default_num_workers() if num_workers is None else num_workers

    frames = {name: df[df["split"] == name] for name in SPLITS}
    empty = [n for n, f in frames.items() if f.empty]
    if empty:
        raise ValueError(f"Empty split(s) in splits.csv: {empty}")

    datasets = {
        name: FundusDataset(frame, processed_dir, label_col=label_col,
                            transform=build_transforms(image_size, enhanced, train=(name == "train"), stored_size=stored_size))
        for name, frame in frames.items()
    }

    generator = torch.Generator().manual_seed(seed)  # makes shuffling and augmentation reproducible
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)  # persistent workers matter on Windows (slow spawn)

    train_labels = frames["train"][label_col].to_numpy()
    n_pos = int(train_labels.sum())
    n_neg = int(len(train_labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("The training split contains only one class.")

    if balanced_sampler:
        weights = np.where(train_labels == 1, 0.5 / n_pos, 0.5 / n_neg)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_labels),
                                        replacement=True, generator=generator)
        train_loader = DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler, drop_last=True, **common)
    else:
        train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, drop_last=True,
                                  generator=generator, **common)
    eval_loaders = {n: DataLoader(datasets[n], batch_size=batch_size, shuffle=False, **common)
                    for n in ("val", "test_internal", "test_external")}

    info = {
        "image_size": image_size, "stored_size": stored_size, "enhanced": enhanced, "label_col": label_col,
        "num_workers": num_workers, "balanced_sampler": balanced_sampler,
        "counts": {n: int(len(f)) for n, f in frames.items()},
        "positive_rate": {n: round(float(f[label_col].mean()), 4) for n, f in frames.items()},
        "pos_weight": n_neg / n_pos,  # pass to torch.nn.BCEWithLogitsLoss(pos_weight=...)
        "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }
    return Loaders(train_loader, **eval_loaders, info=info)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the ARTSS DataLoaders and check one batch.")
    p.add_argument("--root", default=os.getcwd())
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--balanced-sampler", action="store_true")
    p.add_argument("--save-grid", action="store_true", help="Save an augmented training batch as qa_train_batch.png")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    root = Path(args.root).resolve()
    processed = Path(args.processed_dir).resolve() if args.processed_dir else root / "ARTSS" / "data" / "processed"
    loaders = get_dataloaders(processed, args.image_size, args.batch_size, args.num_workers,
                              balanced_sampler=args.balanced_sampler)
    info = loaders.info
    print("DataLoaders ready")
    for name in SPLITS:
        print(f"  {name:14s} {info['counts'][name]:6d} images, positive rate {info['positive_rate'][name]:.3f}")
    print(f"  pos_weight (train neg/pos): {info['pos_weight']:.3f}")

    images, labels = next(iter(loaders.train))
    print(f"Batch images: {tuple(images.shape)} (B, C, H, W) | labels: {tuple(labels.shape)} "
          f"| batch positive rate {labels.mean().item():.2f}")
    print(f"Value range after normalisation: {images.min().item():.2f} .. {images.max().item():.2f}")
    if not torch.isfinite(images).all():
        raise RuntimeError("Non-finite values in a batch")

    if args.save_grid:
        from torchvision.utils import save_image
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        out = processed / "qa_train_batch.png"
        save_image((images[:16] * std + mean).clamp(0, 1), out, nrow=4)
        print(f"Saved augmented batch for visual inspection: {out}")
    return 0


if __name__ == "__main__":  # required on Windows when num_workers > 0
    sys.exit(main())