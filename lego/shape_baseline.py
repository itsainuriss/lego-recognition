"""Step 3a (baseline) - Shape with hand-made features + SVM.

This is the 'simple classic method' from slide 6. It exists so we can show
how much the CNN adds. Features:
  * HOG on a 64x64 greyscale crop (edges, studs, slopes)
  * silhouette features: aspect ratio, extent, solidity, log Hu moments
Training uses random 90-degree rotations/flips so the SVM sees several poses.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

SIZE = 64


def _silhouette(grey: np.ndarray) -> np.ndarray:
    # brick vs uniform background: background = value at the border
    border = np.concatenate([grey[0], grey[-1], grey[:, 0], grey[:, -1]])
    diff = np.abs(grey.astype(np.int16) - int(np.median(border))).astype(np.uint8)
    _, m = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(10, np.float32)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c) + 1e-6
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    w, h = max(w, 1), max(h, 1)
    aspect = min(w, h) / max(w, h)
    extent = area / (w * h)
    hull = cv2.contourArea(cv2.convexHull(c)) + 1e-6
    solidity = area / hull
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return np.concatenate([[aspect, extent, solidity], hu]).astype(np.float32)


def features(grey: np.ndarray) -> np.ndarray:
    g = cv2.resize(grey, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    g = cv2.equalizeHist(g)
    h = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return np.concatenate([h.astype(np.float32), _silhouette(g) * 3.0])


def augment(grey: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    g = np.rot90(grey, rng.integers(4))
    if rng.random() < 0.5:
        g = np.fliplr(g)
    return np.ascontiguousarray(g)


class BaselineShape:
    """Thin wrapper so the pipeline can call .predict() like the CNN."""

    def __init__(self, path):
        import joblib

        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = bundle["classes"]

    def predict(self, greys: list[np.ndarray], topk: int = 3):
        X = np.stack([features(g) for g in greys])
        P = self.model.predict_proba(X)
        out = []
        for p in P:
            order = np.argsort(p)[::-1][:topk]
            out.append([(self.classes[i], float(p[i])) for i in order])
        return out
