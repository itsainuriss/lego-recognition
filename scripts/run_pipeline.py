"""Photo in, brick list out - from the command line.

    python scripts/run_pipeline.py photo.jpg --out results/run1 [--sets data/rebrickable/prepared.pkl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lego import sets as S  # noqa: E402
from lego.data import part_id  # noqa: E402
from lego.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--config")
    ap.add_argument("--sets", help="prepared.pkl from scripts/prepare_rebrickable.py")
    ap.add_argument("--no-colour-match", action="store_true", help="match sets by shape only")
    args = ap.parse_args()

    pipe = Pipeline(args.config)
    res = pipe.run(args.photo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "annotated.jpg"), cv2.cvtColor(res.annotated, cv2.COLOR_RGB2BGR))
    res.detections_df.to_csv(out / "detections.csv", index=False)
    res.inventory.to_csv(out / "inventory.csv", index=False)
    print(res.inventory.to_string(index=False))
    print(f"\n{len(res.detections)} bricks, {int(res.detections_df['unsure'].sum()) if len(res.detections_df) else 0} unsure")

    if args.sets:
        sp, sets, _ = S.load_prepared(args.sets)
        known = {part_id(c) for c in pipe.shape.classes} if pipe.shape is not None else None
        top = S.what_can_i_build(res.inventory, sp, sets, use_colour=not args.no_colour_match, known_parts=known)
        top.to_csv(out / "sets.csv", index=False)
        print("\nWhat can I build?\n", top.drop(columns=["img_url"]).to_string(index=False))
    print("\nwritten to", out)


if __name__ == "__main__":
    main()
