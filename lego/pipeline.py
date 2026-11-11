"""End-to-end pipeline: photo -> detections -> inventory.

    from lego.pipeline import Pipeline
    pipe = Pipeline()                     # reads configs/default.yaml
    result = pipe.run("photo.jpg")
    result.inventory                      # pandas DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import capture, colour, inventory, segment
from .config import load_config, resolve


@dataclass
class Result:
    image: np.ndarray  # white-balanced RGB
    detections: list
    detections_df: pd.DataFrame
    inventory: pd.DataFrame
    annotated: np.ndarray
    mask: np.ndarray
    gains: np.ndarray | None


class Pipeline:
    def __init__(self, cfg: dict | str | None = None, shape_model=None):
        self.cfg = cfg if isinstance(cfg, dict) else load_config(cfg)
        self.palette = colour.ColourPalette(resolve(self.cfg["colour"]["palette"]))
        self.shape = shape_model if shape_model is not None else self._load_shape()

    def _load_shape(self):
        sc = self.cfg["shape"]
        method = sc.get("method", "cnn")
        if method == "cnn":
            ck = resolve(sc["cnn_checkpoint"])
            try:
                from .shape_cnn import CNNShape

                if ck.exists():
                    return CNNShape(ck)
                print(f"[pipeline] no CNN checkpoint at {ck} - trying the baseline")
            except ImportError:
                print("[pipeline] PyTorch/timm not installed - trying the baseline")
        bm = resolve(sc["baseline_model"])
        if bm.exists():
            from .shape_baseline import BaselineShape

            return BaselineShape(bm)
        print("[pipeline] no shape model found - parts will be '?' (colour + count still work)")
        return None

    def run(self, image) -> Result:
        cc, sc_, cs, ss = self.cfg["capture"], self.cfg["segment"], self.cfg["colour"], self.cfg["shape"]
        img = image if isinstance(image, np.ndarray) else capture.load_image(image, cc.get("max_side"))
        gains = None
        roi = cc.get("grey_card_roi")
        if roi:
            img, gains = capture.white_balance(img, roi, cc.get("grey_target", 119))

        dets, mask = segment.find_bricks(img, sc_, card_roi=roi)

        # 3a shape
        if self.shape is not None and dets:
            greys = [segment.to_model_input(d, ss.get("image_size", 224)) for d in dets]
            for d, top in zip(dets, self.shape.predict(greys)):
                d.part, d.part_conf = top[0]
                d.part_top = top

        # 3b colour
        for d in dets:
            lab = colour.brick_lab(d.crop, d.mask, cs.get("erode", 4))
            if lab is None:
                d.unsure = True
                continue
            r = colour.classify_colour(lab, self.palette, cs.get("max_delta_e", 18), cs.get("min_margin", 4))
            d.colour_id, d.colour_name = r.colour_id, r.name
            d.colour_conf, d.colour_delta_e = r.confidence, r.delta_e
            d.unsure = r.unsure or (self.shape is not None and d.part_conf < ss.get("min_confidence", 0.6))

        det_df = inventory.detections_table(dets)
        inv = inventory.aggregate(det_df)
        return Result(img, dets, det_df, inv, segment.draw_detections(img, dets), mask, gains)
