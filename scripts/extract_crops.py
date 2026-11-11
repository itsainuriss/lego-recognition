"""Cut our own photos into labelled crops (for fine-tuning and testing).

Labelling trick that saves almost all manual work: photograph ONE part type
per photo (e.g. ten 3001 bricks in different colours and poses), and put the
photo in a folder named like the class:

    photos/
      3001 brick 2x4/   IMG_0001.jpg IMG_0002.jpg ...
      3022 Plate 2x2/   ...

    python scripts/extract_crops.py photos data/real_all
    python scripts/split_real.py data/real_all data/real --test 0.3

Every crop inherits the folder name as its label. Mixed photos (for the
inventory demo) do not need crops - they are labelled in CVAT instead.
Check the crops by eye once: a wrong crop (shadow, two bricks) -> delete it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lego import capture, segment  # noqa: E402
from lego.config import load_config  # noqa: E402
from lego.data import IMG_EXT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--config")
    ap.add_argument("--colour", action="store_true", help="also save the colour crop + mask (for colour calibration)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    roi = cfg["capture"].get("grey_card_roi")

    total = 0
    for cls_dir in sorted(p for p in Path(args.photos).iterdir() if p.is_dir()):
        out_dir = Path(args.out) / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for photo in sorted(cls_dir.iterdir()):
            if photo.suffix.lower() not in IMG_EXT:
                continue
            img = capture.load_image(photo, cfg["capture"].get("max_side"))
            if roi:
                img, _ = capture.white_balance(img, roi, cfg["capture"].get("grey_target", 119))
            dets, _ = segment.find_bricks(img, cfg["segment"], roi)
            for d in dets:
                stem = f"{photo.stem}_{d.idx:02d}"
                cv2.imwrite(str(out_dir / f"{stem}.png"), segment.to_model_input(d, args.size))
                if args.colour:
                    cv2.imwrite(str(out_dir / f"{stem}_rgb.png"), cv2.cvtColor(d.crop, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), d.mask)
                n += 1
        print(f"{cls_dir.name:30s} {n:4d} crops")
        total += n
    print("total", total)


if __name__ == "__main__":
    main()
