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


def class_from_filename(stem: str) -> str:
    """'3001 brick 2x4 000L' -> '3001 brick 2x4' (flat Kaggle layout: class in the file name)."""
    parts = stem.strip().split()
    return " ".join(parts[:-1]) if len(parts) > 1 else stem


def _flat_classes(d: Path) -> set[str]:
    return {class_from_filename(p.stem) for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT and p.name[:1].isdigit()}


def list_image_folder(root, classes: list[str] | None = None):
    """Return (paths, labels, classes). Pass `classes` to keep the same order as training.

    Two layouts are supported:
      root/<class name>/<img>         one subfolder per class
      root/<class name> <nr>.png      flat, class in the file name (Kaggle 50-class version)
    """
    root = Path(root)
    subdirs = sorted(d.name for d in root.iterdir() if d.is_dir())
    flat = not subdirs
    if flat:
        files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
        groups: dict[str, list[Path]] = {}
        for p in files:
            groups.setdefault(class_from_filename(p.stem), []).append(p)
    else:
        groups = {d: sorted(p for p in (root / d).iterdir() if p.suffix.lower() in IMG_EXT) for d in subdirs}
    if classes is None:
        classes = sorted(groups)
    by_part = {part_id(c): i for i, c in enumerate(classes)}
    paths, labels = [], []
    for name, ps in groups.items():
        idx = by_part.get(part_id(name))
        if idx is None:
            print(f"[data] skipping '{name}': part not in class list")
            continue
        paths += ps
        labels += [idx] * len(ps)
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
    """Find the folder with the most classes below `path` - either as class subfolders
    or as flat files named '<part> <name> <nr>.png'. Kaggle nests folders, and the LEGO
    dataset ships an old 16-class version next to the 50-class one."""
    path = Path(path)
    best, best_n = path, 0
    for d in [path, *[p for p in path.rglob("*") if p.is_dir()]]:
        n_sub = sum(1 for c in d.iterdir() if c.is_dir() and c.name[:1].isdigit())
        n = max(n_sub, len(_flat_classes(d)) if n_sub == 0 else 0)
        if n > best_n:
            best, best_n = d, n
    print(f"[data] using {best} ({best_n} classes)")
    return best
