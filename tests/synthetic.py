"""Synthetic test data: a fake 'photo' of bricks and a tiny fake Rebrickable dump.

Not realistic - only used to check that the plumbing works end to end.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

COLOURS = {  # rb_color_id: RGB
    4: (201, 26, 9),
    1: (0, 85, 191),
    14: (242, 205, 55),
    0: (5, 19, 29),
    2: (35, 120, 65),
}


def draw_brick(img, cx, cy, studs_w, studs_h, rgb, angle=0, unit=28):
    w, h = studs_w * unit, studs_h * unit
    tile = np.zeros((h + 8, w + 8, 4), np.uint8)
    cv2.rectangle(tile, (4, 4), (4 + w, 4 + h), (*rgb, 255), -1)
    shade = tuple(int(min(255, c * 1.12 + 10)) for c in rgb)
    for i in range(studs_w):
        for j in range(studs_h):
            c = (4 + unit // 2 + i * unit, 4 + unit // 2 + j * unit)
            cv2.circle(tile, c, unit // 3, (*shade, 255), -1)
            cv2.circle(tile, c, unit // 3, tuple(int(x * 0.8) for x in rgb) + (255,), 1)
    M = cv2.getRotationMatrix2D((tile.shape[1] / 2, tile.shape[0] / 2), angle, 1.0)
    side = int(np.hypot(*tile.shape[:2])) + 2
    M[0, 2] += (side - tile.shape[1]) / 2
    M[1, 2] += (side - tile.shape[0]) / 2
    rot = cv2.warpAffine(tile, M, (side, side), flags=cv2.INTER_LINEAR)
    x0, y0 = cx - side // 2, cy - side // 2
    a = rot[..., 3:4].astype(np.float32) / 255.0
    # soft shadow
    sh = cv2.GaussianBlur(rot[..., 3], (21, 21), 0).astype(np.float32)[..., None] / 255.0 * 0.25
    roi = img[y0 + 6:y0 + 6 + side, x0 + 6:x0 + 6 + side].astype(np.float32)
    img[y0 + 6:y0 + 6 + side, x0 + 6:x0 + 6 + side] = (roi * (1 - sh)).astype(np.uint8)
    roi = img[y0:y0 + side, x0:x0 + side].astype(np.float32)
    img[y0:y0 + side, x0:x0 + side] = (roi * (1 - a) + rot[..., :3] * a).astype(np.uint8)


BRICKS = [  # (cx, cy, studs_w, studs_h, colour_id, angle, part)
    (400, 300, 4, 2, 4, 10, "3001"),
    (750, 320, 4, 2, 4, -25, "3001"),
    (1100, 280, 2, 2, 1, 40, "3003"),
    (400, 650, 2, 2, 1, 0, "3003"),
    (750, 680, 2, 1, 14, 70, "3004"),
    (1050, 650, 2, 1, 14, 15, "3004"),
    (1300, 600, 4, 2, 2, 5, "3001"),
    (600, 900, 2, 1, 0, 30, "3004"),
]


def make_photo(cast=(1.08, 1.0, 0.86)) -> np.ndarray:
    """1600x1100 photo: light background with a lighting gradient, grey card, bricks,
    and a warm colour cast that the grey card has to remove."""
    H, W = 1100, 1600
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = 225 - 25 * (xx / W) - 15 * (yy / H)
    img = np.repeat(bg[..., None], 3, 2).astype(np.uint8)
    cv2.rectangle(img, (40, 25), (215, 150), (119, 119, 119), -1)  # grey card (roi 0.02..0.14)
    for cx, cy, sw, sh, cid, ang, _ in BRICKS:
        draw_brick(img, cx, cy, sw, sh, COLOURS[cid], ang)
    rng = np.random.default_rng(0)
    img = img.astype(np.float32) * np.array(cast, np.float32)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_fake_rebrickable(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"set_num": ["1-1", "2-1", "3-1"], "name": ["Small car", "Mini house", "Tower"],
         "year": [2020, 2021, 2022], "theme_id": [1, 1, 1], "num_parts": [6, 9, 30], "img_url": ["", "", ""]}
    ).to_csv(d / "sets.csv", index=False)
    pd.DataFrame({"id": [10, 11, 20, 30], "version": [1, 2, 1, 1], "set_num": ["1-1", "1-1", "2-1", "3-1"]}).to_csv(
        d / "inventories.csv", index=False)
    rows = [
        (11, "3001", 4, 2, "f"), (11, "3003", 1, 2, "f"), (11, "3004", 14, 2, "f"), (11, "3004", 14, 1, "t"),
        (10, "9999", 4, 50, "f"),  # old version - must be ignored
        (20, "3001", 4, 3, "f"), (20, "3003", 1, 2, "f"), (20, "3069b", 15, 4, "f"),
        (30, "3001", 72, 20, "f"), (30, "3004", 0, 10, "f"),
    ]
    pd.DataFrame(rows, columns=["inventory_id", "part_num", "color_id", "quantity", "is_spare"]).to_csv(
        d / "inventory_parts.csv", index=False)
