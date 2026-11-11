"""Measure our own LEGO colour references under OUR lamps.

The official RGB values (configs/colors.csv) describe the plastic under
ideal light; our camera sees something else. So: photograph a few bricks of
each colour with the normal setup (grey card in the frame!), one colour per
folder named "<rebrickable colour id> <name>":

    colour_photos/
      4 Red/                 IMG_1.jpg ...
      71 Light Bluish Gray/  ...
      72 Dark Bluish Gray/   ...

    python scripts/calibrate_colours.py colour_photos configs/colors_measured.csv

Then set  colour: palette: configs/colors_measured.csv  in the config.
Also prints which colour pairs are closest (= the risky ones, e.g. the greys)
and a leave-one-photo-out accuracy so you know how good colour is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000, lab2rgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lego import capture, colour, segment  # noqa: E402
from lego.config import load_config  # noqa: E402
from lego.data import IMG_EXT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos")
    ap.add_argument("out")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    roi = cfg["capture"].get("grey_card_roi")

    samples = []  # (colour_id, name, photo, lab)
    for d in sorted(p for p in Path(args.photos).iterdir() if p.is_dir()):
        cid, _, name = d.name.partition(" ")
        for photo in sorted(d.iterdir()):
            if photo.suffix.lower() not in IMG_EXT:
                continue
            img = capture.load_image(photo, cfg["capture"].get("max_side"))
            if roi:
                img, _ = capture.white_balance(img, roi, cfg["capture"].get("grey_target", 119))
            dets, _ = segment.find_bricks(img, cfg["segment"], roi)
            for det in dets:
                lab = colour.brick_lab(det.crop, det.mask, cfg["colour"].get("erode", 4))
                if lab is not None:
                    samples.append((int(cid), name.strip(), photo.name, lab))
        print(f"{d.name:28s} {sum(1 for s in samples if s[0] == int(cid)):4d} bricks")

    df = pd.DataFrame(samples, columns=["id", "name", "photo", "lab"])
    rows = []
    for (cid, name), g in df.groupby(["id", "name"]):
        lab = np.median(np.stack(g["lab"]), axis=0)
        rgb = np.clip(lab2rgb(lab[None, None])[0, 0] * 255, 0, 255).round().astype(int)
        rows.append({"rb_color_id": cid, "name": name, "r": rgb[0], "g": rgb[1], "b": rgb[2], "n": len(g)})
    ref = pd.DataFrame(rows)
    ref[["rb_color_id", "name", "r", "g", "b"]].to_csv(args.out, index=False)
    print("\nwritten", args.out)

    # closest pairs = where mistakes will happen
    labs = {r.name: np.median(np.stack(df[df["name"] == r.name]["lab"]), axis=0) for r in ref.itertuples()}
    names = list(labs)
    pairs = sorted((float(deltaE_ciede2000(labs[a], labs[b])), a, b) for i, a in enumerate(names) for b in names[i + 1:])
    print("\nclosest colour pairs (CIEDE2000, < 10 is risky):")
    for de, a, b in pairs[:6]:
        print(f"  {de:5.1f}  {a} <-> {b}")

    # leave-one-photo-out accuracy
    correct = 0
    for photo in df["photo"].unique():
        train, test = df[df["photo"] != photo], df[df["photo"] == photo]
        cen = {n: np.median(np.stack(g["lab"]), axis=0) for n, g in train.groupby("name")}
        for t in test.itertuples():
            pred = min(cen, key=lambda n: deltaE_ciede2000(t.lab, cen[n]))
            correct += pred == t.name
    print(f"\nleave-one-photo-out colour accuracy: {correct / len(df):.1%} on {len(df)} bricks")


if __name__ == "__main__":
    main()
