"""Step 4 - Inventory: part, colour, count + how sure we are."""
from __future__ import annotations

import pandas as pd

from .data import part_id


def detections_table(dets) -> pd.DataFrame:
    """One row per detected brick (this is what the user corrects in the app)."""
    rows = []
    for d in dets:
        rows.append(
            {
                "idx": d.idx,
                "part": part_id(d.part) if d.part else None,
                "part_name": d.part,
                "part_conf": round(d.part_conf, 3),
                "alt_parts": ", ".join(f"{part_id(n)} {p:.0%}" for n, p in d.part_top[1:3]),
                "colour_id": d.colour_id,
                "colour": d.colour_name,
                "colour_conf": round(d.colour_conf, 3),
                "delta_e": round(d.colour_delta_e, 1),
                "unsure": d.unsure,
                "x0": d.bbox[0],
                "y0": d.bbox[1],
                "x1": d.bbox[2],
                "y1": d.bbox[3],
            }
        )
    return pd.DataFrame(rows)


def aggregate(det_df: pd.DataFrame) -> pd.DataFrame:
    """Inventory: count per (part, colour).

    `expected` = sum of part_conf * colour_conf -> the 'honest' count used to
    score sets; unsure bricks count less than sure ones.
    """
    cols = ["part", "part_name", "colour_id", "colour", "count", "unsure", "expected", "mean_conf"]
    if det_df.empty:
        return pd.DataFrame(columns=cols)
    df = det_df.dropna(subset=["part"]).copy()
    df["joint"] = df["part_conf"].astype(float) * df["colour_conf"].astype(float)
    # bricks the user confirmed by hand get confidence 1
    if "confirmed" in df:
        df.loc[df["confirmed"].fillna(False).astype(bool), "joint"] = 1.0
    g = df.groupby(["part", "colour_id"], dropna=False)
    inv = g.agg(
        part_name=("part_name", "first"),
        colour=("colour", "first"),
        count=("idx", "size"),
        unsure=("unsure", "sum"),
        expected=("joint", "sum"),
        mean_conf=("joint", "mean"),
    ).reset_index()
    inv["expected"] = inv["expected"].round(2)
    inv["mean_conf"] = inv["mean_conf"].round(2)
    return inv[cols].sort_values(["part", "colour"]).reset_index(drop=True)
