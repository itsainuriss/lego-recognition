"""Make renders look like our crops - the tool for the core experiment (slide 5).

Two levels, so we can measure what they are worth:
  basic : rotation, flip, scale/shift          (geometry only)
  real  : basic + everything a phone photo does to a brick:
          blur, sensor noise, JPEG, gamma/contrast, uneven light,
          a soft shadow, slight perspective, low resolution,
          and a slightly wrong mask (segmentation is never perfect)

All functions work on uint8 greyscale numpy arrays (cv2 only), so they can
be tested without PyTorch and reused for the SVM baseline.
"""
from __future__ import annotations

import cv2
import numpy as np


def render_mask(grey: np.ndarray) -> np.ndarray:
    """Brick mask of a render: everything that differs from the border colour."""
    border = np.concatenate([grey[0], grey[-1], grey[:, 0], grey[:, -1]])
    diff = cv2.absdiff(grey, np.full_like(grey, int(np.median(border))))
    _, m = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return out


def render_to_input(grey: np.ndarray, size: int, background: int = 255, pad: float = 0.15,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Turn a render into what segment.to_model_input() produces for a real photo:
    brick on a uniform background, tight square crop with the same padding."""
    m = render_mask(grey) if mask is None else mask
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(m, (5, 5), 0).astype(np.float32) / 255.0
    comp = (grey * a + background * (1 - a)).astype(np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad)) + 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    canvas = np.full((side, side), background, np.uint8)
    sx0, sy0 = cx - side // 2, cy - side // 2
    H, W = grey.shape
    ix0, iy0, ix1, iy1 = max(0, sx0), max(0, sy0), min(W, sx0 + side), min(H, sy0 + side)
    canvas[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = comp[iy0:iy1, ix0:ix1]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _geometry(g, rng, background):
    h, w = g.shape
    ang = rng.uniform(0, 360)
    scale = rng.uniform(0.8, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[:, 2] += rng.uniform(-0.06, 0.06, 2) * w
    g = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=background)
    if rng.random() < 0.5:
        g = np.ascontiguousarray(g[:, ::-1])
    return g


def _perspective(g, rng, background, strength=0.06):
    h, w = g.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-strength, strength, (4, 2)).astype(np.float32) * [w, h]
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(g, M, (w, h), borderValue=background)


def _shadow(g, rng, background):
    """Soft drop shadow next to the brick (our LED panels still cast some)."""
    fg = (cv2.absdiff(g, np.full_like(g, background)) > 12).astype(np.uint8) * 255
    dx, dy = rng.integers(-8, 9, 2)
    sh = np.roll(np.roll(fg, dy, 0), dx, 1)
    sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(3, 8)).astype(np.float32) / 255.0
    sh *= rng.uniform(0.1, 0.35) * (fg == 0)
    return np.clip(g * (1 - sh), 0, 255).astype(np.uint8)


def _mask_jitter(g, rng, background):
    """Eat or grow the brick edge by a few px - our masks are never perfect."""
    fg = (cv2.absdiff(g, np.full_like(g, background)) > 10).astype(np.uint8) * 255
    k = int(rng.integers(1, 4))
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    if rng.random() < 0.5:
        m2 = cv2.erode(fg, ker)
        g = np.where((fg > 0) & (m2 == 0), background, g).astype(np.uint8)
    else:
        m2 = cv2.dilate(fg, ker)
        ring = (m2 > 0) & (fg == 0)
        g = g.copy()
        g[ring] = np.clip(background * rng.uniform(0.6, 0.9), 0, 255)
    return g


def _photometric(g, rng):
    x = g.astype(np.float32) / 255.0
    x = x ** rng.uniform(0.7, 1.4)  # gamma
    x = (x - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.1, 0.1)  # contrast/brightness
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5)) * rng.uniform(0, 0.25)
    x = x + grad  # uneven light
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def _camera(g, rng):
    if rng.random() < 0.5:  # low resolution (small brick far from camera)
        s = rng.uniform(0.25, 0.6)
        h, w = g.shape
        g = cv2.resize(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.6:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.7:
        g = np.clip(g + rng.normal(0, rng.uniform(2, 8), g.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        g = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return g


def augment(g: np.ndarray, rng: np.random.Generator, level: str = "real", background: int = 255) -> np.ndarray:
    """g: model input (brick on uniform background). level: 'none' | 'basic' | 'real'."""
    if level == "none":
        return g
    g = _geometry(g, rng, background)
    if level == "basic":
        return g
    if rng.random() < 0.4:
        g = _perspective(g, rng, background)
    if rng.random() < 0.5:
        g = _shadow(g, rng, background)
    if rng.random() < 0.4:
        g = _mask_jitter(g, rng, background)
    g = _photometric(g, rng)
    return _camera(g, rng)
