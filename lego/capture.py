"""Step 1 - Capture: load a photo and fix the white balance with the grey card.

Setup assumed (slide 4): fixed camera, two LED panels, plain background,
grey card always in the same corner of the frame.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path_or_bytes, max_side: int | None = 2000) -> np.ndarray:
    """Return an RGB uint8 image. Accepts a path or raw bytes (Streamlit upload)."""
    if isinstance(path_or_bytes, (str, Path)):
        bgr = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path_or_bytes)
    else:
        buf = np.frombuffer(path_or_bytes, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def roi_to_pixels(roi_frac, shape) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x0, y0, x1, y1 = roi_frac
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def white_balance(img: np.ndarray, roi_frac, target: float = 119.0):
    """Scale R, G, B so the grey card becomes neutral grey.

    Works in linear light (undo sRGB gamma) so the correction is physically
    correct, then converts back. Returns (corrected_image, gains).
    """
    x0, y0, x1, y1 = roi_to_pixels(roi_frac, img.shape)
    card = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if card.size == 0:
        raise ValueError("grey_card_roi is empty - check the config")
    # robust mean: drop the brightest/darkest 10 % (edges, dust, glare)
    lo, hi = np.percentile(card, [10, 90], axis=0)
    keep = np.all((card >= lo) & (card <= hi), axis=1)
    mean = card[keep].mean(axis=0) if keep.any() else card.mean(axis=0)

    lin = _srgb_to_linear(img.astype(np.float32) / 255.0)
    mean_lin = _srgb_to_linear(mean / 255.0)
    target_lin = _srgb_to_linear(np.array([target / 255.0] * 3, np.float32))
    gains = target_lin / np.maximum(mean_lin, 1e-6)
    out = _linear_to_srgb(np.clip(lin * gains, 0, 1))
    return (out * 255 + 0.5).astype(np.uint8), gains


def _srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
