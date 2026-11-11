"""Step 2 - Find bricks: classic image processing for *separated* bricks.

Idea: the background is plain, so model it (a smooth surface, which also
absorbs uneven lighting), mark everything that differs from it as "brick",
clean the mask up and cut out each connected blob.

Touching bricks: a light watershed split for blobs that are clearly not
convex. Real piles need a detector (YOLO-seg) - that is the stretch goal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .capture import roi_to_pixels


@dataclass
class Detection:
    idx: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in the full image
    area: int
    crop: np.ndarray  # RGB, padded square crop
    mask: np.ndarray  # uint8 0/255, same size as crop
    contour: np.ndarray = field(repr=False)  # in full-image coordinates
    # filled in by later steps
    part: str | None = None
    part_conf: float = 0.0
    part_top: list = field(default_factory=list)
    colour_id: int | None = None
    colour_name: str | None = None
    colour_conf: float = 0.0
    colour_delta_e: float = 0.0
    unsure: bool = False


def _fit_background(lab: np.ndarray, iters: int = 3) -> np.ndarray:
    """Fit a 2nd-order polynomial surface per Lab channel to background pixels."""
    h, w = lab.shape[:2]
    small = cv2.resize(lab, (160, max(1, int(160 * h / w))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    xx /= sw
    yy /= sh
    A = np.stack([np.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy], -1).reshape(-1, 6)
    vals = small.reshape(-1, 3).astype(np.float32)
    keep = np.ones(len(vals), bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
        resid = np.linalg.norm(vals - A @ coef, axis=1)
        keep = resid <= np.percentile(resid, 60)  # bricks cover < 40 % of the frame
    Yf, Xf = np.mgrid[0:h, 0:w].astype(np.float32)
    Xf /= w
    Yf /= h
    full = np.stack([np.ones_like(Xf), Xf, Yf, Xf * Xf, Yf * Yf, Xf * Yf], -1) @ coef
    return full.astype(np.float32)


def foreground_mask(img: np.ndarray, cfg: dict, card_roi=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask uint8 0/255, difference map float32)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = _fit_background(lab)
    diff = np.linalg.norm(lab - bg, axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    thr = cfg.get("diff_threshold")
    if thr is None:
        d8 = np.clip(diff * 2, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(t / 2.0, 8.0)  # never go below a sensible floor
    mask = (diff > thr).astype(np.uint8) * 255

    k = int(cfg.get("close_kernel", 7))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = _fill_holes(mask)

    if card_roi is not None:  # the grey card is not a brick
        x0, y0, x1, y1 = roi_to_pixels(card_roi, img.shape)
        pad = int(0.02 * max(img.shape[:2]))
        mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 0
    return mask, diff


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def _split_touching(mask: np.ndarray, solidity_thr: float = 0.88) -> np.ndarray:
    """Return an int label image. Non-convex blobs get a watershed split."""
    n, labels = cv2.connectedComponents(mask)
    out = np.zeros(mask.shape, np.int32)
    next_label = 1
    for lab in range(1, n):
        comp = (labels == lab).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(cnts[0])
        hull = cv2.contourArea(cv2.convexHull(cnts[0]))
        if hull == 0 or area / hull >= solidity_thr:
            out[comp > 0] = next_label
            next_label += 1
            continue
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        _, peaks = cv2.threshold(dist, 0.55 * dist.max(), 255, cv2.THRESH_BINARY)
        peaks = peaks.astype(np.uint8)
        k, markers = cv2.connectedComponents(peaks)
        if k <= 2:  # only one seed: nothing to split
            out[comp > 0] = next_label
            next_label += 1
            continue
        markers = markers + 1
        markers[(comp > 0) & (peaks == 0)] = 0
        ws = cv2.watershed(cv2.cvtColor(comp * 255, cv2.COLOR_GRAY2BGR), markers)
        for m in range(2, k + 1):
            region = (ws == m) & (comp > 0)
            if region.any():
                out[region] = next_label
                next_label += 1
    return out


def find_bricks(img: np.ndarray, cfg: dict, card_roi=None) -> tuple[list[Detection], np.ndarray]:
    """Find all separated bricks. Returns (detections, foreground mask)."""
    mask, _ = foreground_mask(img, cfg, card_roi)
    labels = _split_touching(mask) if cfg.get("split_touching", True) else cv2.connectedComponents(mask)[1]

    H, W = img.shape[:2]
    a_min = cfg.get("min_area_frac", 0.0008) * H * W
    a_max = cfg.get("max_area_frac", 0.25) * H * W
    pad_f = cfg.get("crop_padding", 0.15)

    dets: list[Detection] = []
    for lab in range(1, labels.max() + 1):
        comp = (labels == lab).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = int(cv2.contourArea(cnt))
        if not (a_min <= area <= a_max):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # square crop around the brick, with padding
        side = int(max(w, h) * (1 + 2 * pad_f))
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = cx - side // 2, cy - side // 2
        crop = _crop_pad(img, x0, y0, side, fill=None)
        cmask = _crop_pad(comp, x0, y0, side, fill=0)
        dets.append(Detection(len(dets), (x, y, x + w, y + h), area, crop, cmask, cnt))
    return dets, mask


def _crop_pad(img, x0, y0, side, fill=None):
    """Crop a side x side square, padding outside the image (edge-replicate or fill)."""
    H, W = img.shape[:2]
    pads = (max(0, -y0), max(0, y0 + side - H), max(0, -x0), max(0, x0 + side - W))
    sub = img[max(0, y0):min(H, y0 + side), max(0, x0):min(W, x0 + side)]
    if any(pads):
        if fill is None:
            sub = cv2.copyMakeBorder(sub, *pads, cv2.BORDER_REPLICATE)
        else:
            sub = cv2.copyMakeBorder(sub, *pads, cv2.BORDER_CONSTANT, value=fill)
    return sub


def to_model_input(det: Detection, size: int, background: int = 255, grey: bool = True) -> np.ndarray:
    """Brick on a uniform background, greyscale, resized - what the shape model sees.

    Greyscale on purpose: the shape network must not 'cheat' with colour.
    """
    img = det.crop.copy()
    m = cv2.GaussianBlur(det.mask, (5, 5), 0).astype(np.float32)[..., None] / 255.0
    img = (img * m + background * (1 - m)).astype(np.uint8)
    if grey:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def draw_detections(img: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Annotated copy of the photo: green = sure, orange = unsure."""
    out = img.copy()
    for d in dets:
        col = (255, 150, 0) if d.unsure else (40, 200, 40)
        cv2.drawContours(out, [d.contour], -1, col, 2)
        label = f"#{d.idx} {d.part or '?'} {d.colour_name or ''}".strip()
        x0, y0 = d.bbox[:2]
        fs = max(0.4, img.shape[1] / 2500)
        cv2.putText(out, label, (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3)
        cv2.putText(out, label, (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1)
    return out
