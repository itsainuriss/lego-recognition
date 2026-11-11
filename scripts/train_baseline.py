"""Train the classic baseline: HOG + silhouette features -> SVM.

    python scripts/train_baseline.py --renders data/renders --per-class 300 \
        [--real-train data/real/train] [--real-test data/real/test] --out models/shape_baseline.joblib

Same data and same test set as the CNN, so the numbers are comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from lego.augment import augment, render_to_input  # noqa: E402
from lego.data import find_class_root, list_image_folder, read_grey  # noqa: E402
from lego.shape_baseline import features  # noqa: E402

SIZE = 128


def load(paths, labels, is_render, n_aug, rng, aug_level):
    X, y = [], []
    for p, lab in zip(paths, labels):
        g = render_to_input(read_grey(p), SIZE) if is_render else read_grey(p, SIZE)
        for _ in range(n_aug):
            X.append(features(augment(g, rng, aug_level)))
            y.append(lab)
    return np.array(X), np.array(y)


def subsample(paths, labels, per_class, rng):
    by = {}
    for p, lab in zip(paths, labels):
        by.setdefault(lab, []).append(p)
    P, L = [], []
    for lab, ps in by.items():
        sel = rng.permutation(len(ps))[:per_class] if per_class else range(len(ps))
        P += [ps[i] for i in sel]
        L += [lab] * len(sel)
    return P, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--real-train")
    ap.add_argument("--real-test")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--aug", default="real", choices=["none", "basic", "real"])
    ap.add_argument("--n-aug", type=int, default=2, help="augmented copies per image")
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--out", default="models/shape_baseline.joblib")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    paths, labels, classes = list_image_folder(find_class_root(Path(args.renders)))
    paths, labels = subsample(paths, labels, args.per_class, rng)
    t0 = time.time()
    X, y = load(paths, labels, True, args.n_aug, rng, args.aug)
    if args.real_train:
        rp, rl, _ = list_image_folder(args.real_train, classes)
        Xr, yr = load(rp, rl, False, args.n_aug * 2, rng, "basic")
        X, y = np.vstack([X, Xr]), np.concatenate([y, yr])
    print(f"features: {X.shape} in {time.time() - t0:.0f}s")

    clf = make_pipeline(StandardScaler(), PCA(n_components=min(150, X.shape[1], len(X) - 1), random_state=0),
                        CalibratedClassifierCV(SVC(C=args.C, kernel="rbf", random_state=0), cv=3, ensemble=False))
    clf.fit(X, y)
    print(f"trained in {time.time() - t0:.0f}s")

    res = {}
    if args.real_test:
        tp, tl, _ = list_image_folder(args.real_test, classes)
        Xt, yt = load(tp, tl, False, 1, rng, "none")
        pred = clf.predict(Xt)
        res = {"acc": accuracy_score(yt, pred), "macro_f1": f1_score(yt, pred, average="macro"), "n": len(yt)}
        print("real test:", res)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": classes}, args.out)
    Path("results").mkdir(exist_ok=True)
    with open("results/experiments.jsonl", "a") as f:
        f.write(json.dumps({"tag": args.tag, "model": "hog+svm", "aug": args.aug, "finetuned": bool(args.real_train),
                            "after_finetune" if args.real_train else "after_renders": res, "time": time.strftime("%F %T")}) + "\n")
    print("saved", args.out)


if __name__ == "__main__":
    main()
