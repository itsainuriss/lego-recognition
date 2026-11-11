"""Turn the Rebrickable CSV dumps into one small table for "What can I build?".

1. Download from https://rebrickable.com/downloads/ :
   sets.csv.gz, inventories.csv.gz, inventory_parts.csv.gz, colors.csv.gz
   into data/rebrickable/
2. python scripts/prepare_rebrickable.py data/rebrickable
   -> data/rebrickable/prepared.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lego.sets import _read, build_set_parts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rb_dir", nargs="?", default="data/rebrickable")
    ap.add_argument("--out")
    args = ap.parse_args()
    sp, sets = build_set_parts(args.rb_dir)
    try:
        cols = _read(Path(args.rb_dir), "colors")
        colour_names = dict(zip(cols["id"].astype(int), cols["name"]))
    except FileNotFoundError:
        colour_names = {}
    out = Path(args.out or Path(args.rb_dir) / "prepared.pkl")
    pd.to_pickle({"set_parts": sp, "sets": sets, "colour_names": colour_names}, out)
    print(f"{sets['set_num'].nunique()} sets, {len(sp)} part lines -> {out}")


if __name__ == "__main__":
    main()
