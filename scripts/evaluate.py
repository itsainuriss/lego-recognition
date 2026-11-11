"""Evaluate a shape model (CNN .pt or baseline .joblib) on labelled crops.

    python scripts/evaluate.py models/shape_cnn.pt data/real/test --tag C_finetuned
    python scripts/evaluate.py models/shape_baseline.joblib data/real/test --tag baseline

Writes results/eval_<tag>/:
    metrics.json       accuracy, top-3, macro-F1, mean confidence when right / wrong
    per_class.csv      precision / recall / F1 per part
    confusions.csv     the most confused pairs  -> the 'look-alikes' for the report
    confusion.png      confusion matrix
    errors.png         gallery of wrong predictions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lego.augment import render_to_input  # noqa: E402
from lego.data import find_class_root, list_image_folder, part_id, read_grey  # noqa: E402


def load_model(path):
    if str(path).endswith(".joblib"):
        from lego.shape_baseline import BaselineShape

        return BaselineShape(path)
    from lego.shape_cnn import CNNShape

    return CNNShape(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("--renders", action="store_true", help="data are Kaggle renders (preprocess like training)")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()

    model = load_model(args.model)
    classes = model.classes
    root = find_class_root(args.data) if args.renders else Path(args.data)
    paths, labels, _ = list_image_folder(root, classes)
    greys = [render_to_input(read_grey(p), 224) if args.renders else read_grey(p) for p in paths]

    preds = []
    for i in range(0, len(greys), args.bs):
        preds += model.predict(greys[i:i + args.bs], topk=3)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array(labels)
    p1 = np.array([idx[t[0][0]] for t in preds])
    conf = np.array([t[0][1] for t in preds])
    top3 = np.array([labels[i] in [idx[c] for c, _ in t] for i, t in enumerate(preds)])

    K = len(classes)
    cm = np.zeros((K, K), int)
    for a, b in zip(y, p1):
        cm[a, b] += 1
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    present = cm.sum(1) > 0
    right = p1 == y
    metrics = {
        "tag": args.tag, "model": str(args.model), "data": str(args.data), "n": int(len(y)),
        "acc": float(right.mean()), "top3": float(top3.mean()), "macro_f1": float(f1[present].mean()),
        "conf_when_right": float(conf[right].mean()) if right.any() else None,
        "conf_when_wrong": float(conf[~right].mean()) if (~right).any() else None,
    }
    out = Path("results") / f"eval_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"class": classes, "n": cm.sum(1), "precision": prec, "recall": rec, "f1": f1})[present] \
        .sort_values("f1").to_csv(out / "per_class.csv", index=False)
    pairs = [(classes[a], classes[b], cm[a, b]) for a in range(K) for b in range(K) if a != b and cm[a, b]]
    pd.DataFrame(sorted(pairs, key=lambda t: -t[2]), columns=["true", "predicted", "count"]).to_csv(out / "confusions.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 11))
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ticks = [part_id(c) for c in classes]
        ax.set_xticks(range(K), ticks, rotation=90, fontsize=6)
        ax.set_yticks(range(K), ticks, fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{args.tag}: acc {metrics['acc']:.1%}")
        fig.tight_layout()
        fig.savefig(out / "confusion.png", dpi=150)
        plt.close(fig)

        wrong = np.nonzero(~right)[0][:24]
        if len(wrong):
            cols = 6
            rows = int(np.ceil(len(wrong) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.3))
            for ax in np.atleast_1d(axes).ravel():
                ax.axis("off")
            for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
                ax.imshow(greys[i], cmap="gray")
                ax.set_title(f"true {part_id(classes[y[i]])}\npred {part_id(classes[p1[i]])} {conf[i]:.0%}", fontsize=7)
            fig.tight_layout()
            fig.savefig(out / "errors.png", dpi=120)
            plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(metrics, indent=2))
    print("written to", out)


if __name__ == "__main__":
    main()
