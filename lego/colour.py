"""Step 3b - Colour: measure and compare (no neural network).

1. The photo is already white-balanced with the grey card (capture.py).
2. Take the brick's pixels, shrink the mask (edges and shadows lie),
   drop specular highlights and deep shadows.
3. Median colour in CIE Lab.
4. Nearest reference colour by CIEDE2000. The references start as the
   official Rebrickable RGB values and should be REPLACED by colours you
   measure under your own lamps (scripts/calibrate_colours.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000, rgb2lab


@dataclass
class ColourResult:
    colour_id: int
    name: str
    confidence: float  # 0..1, from the gap between best and second-best match
    delta_e: float  # distance to the best reference (lower = better)
    lab: tuple
    unsure: bool
    ranking: list  # [(name, delta_e), ...] best first


class ColourPalette:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.ids = df["rb_color_id"].astype(int).tolist()
        self.names = df["name"].tolist()
        rgb = df[["r", "g", "b"]].to_numpy(np.float32) / 255.0
        self.lab = rgb2lab(rgb[None])[0]  # (N, 3)

    def __len__(self):
        return len(self.ids)


def brick_lab(crop_rgb: np.ndarray, mask: np.ndarray, erode: int = 4) -> np.ndarray | None:
    """Robust median Lab colour of the brick pixels."""
    m = mask > 0
    if erode > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        m2 = cv2.erode(mask, ker) > 0
        if m2.sum() > 30:  # keep the original for tiny bricks
            m = m2
    px = crop_rgb[m].astype(np.float32) / 255.0
    if len(px) < 10:
        return None
    lab = rgb2lab(px[None])[0]
    L = lab[:, 0]
    # drop the brightest 8 % (glare on studs) and darkest 8 % (shadows between studs)
    lo, hi = np.percentile(L, [8, 92])
    keep = (L >= lo) & (L <= hi)
    if keep.sum() >= 10:
        lab = lab[keep]
    return np.median(lab, axis=0)


def classify_colour(lab: np.ndarray, palette: ColourPalette, max_delta_e=18.0, min_margin=4.0) -> ColourResult:
    d = deltaE_ciede2000(np.repeat(lab[None], len(palette), 0), palette.lab)
    order = np.argsort(d)
    best, second = order[0], order[1]
    margin = float(d[second] - d[best])
    # soft confidence: softmax over negative distances (temperature 3 dE)
    w = np.exp(-(d - d[best]) / 3.0)
    conf = float(w[best] / w.sum())
    unsure = d[best] > max_delta_e or margin < min_margin
    return ColourResult(
        colour_id=palette.ids[best],
        name=palette.names[best],
        confidence=conf,
        delta_e=float(d[best]),
        lab=tuple(float(x) for x in lab),
        unsure=bool(unsure),
        ranking=[(palette.names[i], float(d[i])) for i in order[:3]],
    )
