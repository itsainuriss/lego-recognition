"""Grad-CAM: where does the network look? (Studs? Edges? The background?)

    pip install grad-cam
    python scripts/gradcam.py models/shape_cnn.pt data/real/test --n 16 --out results/gradcam.png

Good for the report: show that plate vs. tile is decided on the studs.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402

from lego.data import list_image_folder, part_id, read_grey  # noqa: E402
from lego.shape_cnn import CNNShape  # noqa: E402


def last_conv(model):
    import torch.nn as nn

    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="results/gradcam.png")
    args = ap.parse_args()

    m = CNNShape(args.model, device="cpu")
    paths, labels, _ = list_image_folder(args.data, m.classes)
    sel = random.Random(0).sample(range(len(paths)), min(args.n, len(paths)))
    greys = [read_grey(paths[i], m.size) for i in sel]
    x = m._prep(greys)
    cam = GradCAM(model=m.model, target_layers=[last_conv(m.model)])
    heat = cam(input_tensor=x)
    preds = m.predict(greys, topk=1, tta=False)

    cols = 4
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, g, h, i, p in zip(np.atleast_1d(axes).ravel(), greys, heat, sel, preds):
        ax.imshow(g, cmap="gray")
        ax.imshow(h, cmap="jet", alpha=0.4)
        ok = "ok" if part_id(p[0][0]) == part_id(m.classes[labels[i]]) else "WRONG"
        ax.set_title(f"true {part_id(m.classes[labels[i]])} / pred {part_id(p[0][0])} {ok}", fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
