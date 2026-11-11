"""Dataset helpers shared by baseline, CNN and evaluation.

Folder layout everywhere (Kaggle renders and our own photos alike):

    root/
      3001 brick 2x4/      img001.png ...
      3022 Plate 2x2/      ...

The class name is the folder name; the part number is its first token.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def part_id(class_name: str) -> str:
    return class_name.strip().split()[0]


def list_image_folder(root, classes: list[str] | None = None):
    """Return (paths, labels, classes). Pass `classes` to keep the same order as training."""
    root = Path(root)
    found = sorted(d.name for d in root.iterdir() if d.is_dir())
    if classes is None:
        classes = found
    by_part = {part_id(c): i for i, c in enumerate(classes)}
    paths, labels = [], []
    for d in found:
        idx = by_part.get(part_id(d))
        if idx is None:
            print(f"[data] skipping folder '{d}': part not in class list")
            continue
        for p in sorted((root / d).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                paths.append(p)
                labels.append(idx)
    return paths, labels, classes


def stratified_split(labels, val_frac=0.15, seed=0):
    rng = random.Random(seed)
    by_cls: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(y, []).append(i)
    tr, va = [], []
    for idx in by_cls.values():
        rng.shuffle(idx)
        k = max(1, int(len(idx) * val_frac))
        va += idx[:k]
        tr += idx[k:]
    return tr, va


def read_grey(path, size: int | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def find_class_root(path) -> Path:
    """Kaggle datasets often nest folders - find the dir whose subfolders are classes
    (folder names starting with a part number)."""
    path = Path(path)
    best, best_n = path, 0
    for d in [path, *[p for p in path.rglob("*") if p.is_dir()]]:
        n = sum(1 for c in d.iterdir() if c.is_dir() and c.name[:1].isdigit())
        if n > best_n:
            best, best_n = d, n
    return best
