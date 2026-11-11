"""Fake 'renders' and fake class photos, to smoke-test the training/eval scripts
without the 40k Kaggle images.  python tests/make_fake_dataset.py /tmp/fake"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import COLOURS, draw_brick  # noqa: E402

CLASSES = {"3001 brick 2x4": (4, 2), "3003 brick 2x2": (2, 2), "3004 brick 1x2": (2, 1), "3010 brick 1x4": (4, 1)}


def main(out):
    out = Path(out)
    rng = np.random.default_rng(0)
    for name, (w, h) in CLASSES.items():
        d = out / "renders" / "LEGO brick images v1" / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(40):
            img = np.full((400, 400, 3), 230, np.uint8)
            grey = int(rng.integers(90, 200))
            draw_brick(img, 200, 200, w, h, (grey, grey, grey), angle=float(rng.uniform(0, 360)), unit=40)
            cv2.imwrite(str(d / f"{i:04d}.png"), img)
        p = out / "photos" / name
        p.mkdir(parents=True, exist_ok=True)
        for k in range(4):  # 4 photos per class, 6 bricks each
            H, W = 1100, 1600
            img = np.full((H, W, 3), 215, np.uint8)
            cv2.rectangle(img, (40, 25), (215, 150), (119, 119, 119), -1)
            for j in range(6):
                cx, cy = 400 + (j % 3) * 400, 380 + (j // 3) * 400
                cid = list(COLOURS)[int(rng.integers(len(COLOURS)))]
                draw_brick(img, cx, cy, w, h, COLOURS[cid], angle=float(rng.uniform(0, 360)))
            img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(str(p / f"IMG_{k}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake")
