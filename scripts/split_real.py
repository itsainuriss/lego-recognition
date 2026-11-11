"""Split our crops into train / test - BY PHOTO, not by crop.

Crops from the same photo share light, camera angle and often the same
physical brick. If they end up on both sides, the test score is too
optimistic. So all crops of one photo go to the same side.

    python scripts/split_real.py data/real_all data/real --test 0.3
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--test", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for cls in sorted(p for p in Path(args.src).iterdir() if p.is_dir()):
        crops = [p for p in cls.glob("*.png") if not p.stem.endswith(("_rgb", "_mask"))]
        photos = sorted({p.stem.rsplit("_", 1)[0] for p in crops})
        rng.shuffle(photos)
        k = max(1, round(len(photos) * args.test)) if len(photos) > 1 else 0
        test_photos = set(photos[:k])
        for p in crops:
            split = "test" if p.stem.rsplit("_", 1)[0] in test_photos else "train"
            out = Path(args.dst) / split / cls.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out / p.name)
        print(f"{cls.name:30s} photos: {len(photos) - k} train / {k} test")


if __name__ == "__main__":
    main()
