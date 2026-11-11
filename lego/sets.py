"""Step 5 - "What can I build?": compare the inventory with official set part lists.

Data: Rebrickable CSV dumps (https://rebrickable.com/downloads/):
    sets.csv.gz, inventories.csv.gz, inventory_parts.csv.gz, colors.csv.gz, parts.csv.gz
Run scripts/prepare_rebrickable.py once to turn them into one small table.

Coverage of a set = sum over its part lines of min(have, need) / sum(need).
`have` can be the plain count or the confidence-weighted `expected` count,
so unsure bricks lower the score (slide 7).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_BASE = re.compile(r"^(\d+)")


def norm_part(p: str, loose: bool) -> str:
    """'3069b' / '3069bpr0001' -> '3069' when loose (mould variants and prints merged)."""
    p = str(p).strip().lower()
    if not loose:
        return p
    m = _BASE.match(p)
    return m.group(1) if m else p


def _read(d: Path, name: str) -> pd.DataFrame:
    for ext in (".csv.gz", ".csv"):
        f = d / f"{name}{ext}"
        if f.exists():
            return pd.read_csv(f, dtype=str, keep_default_na=False)
    raise FileNotFoundError(f"{name}.csv(.gz) not found in {d}")


def build_set_parts(rb_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (set_parts, sets) from raw Rebrickable dumps."""
    d = Path(rb_dir)
    sets = _read(d, "sets")
    inv = _read(d, "inventories")
    ip = _read(d, "inventory_parts")

    inv["version"] = inv["version"].astype(int)
    inv = inv.sort_values("version").drop_duplicates("set_num", keep="last")  # newest version per set
    ip = ip[~ip["is_spare"].str.lower().isin(["t", "true", "1"])]
    sp = ip.merge(inv[["id", "set_num"]], left_on="inventory_id", right_on="id")
    sp = sp.rename(columns={"part_num": "part"})
    sp["color_id"] = sp["color_id"].astype(int)
    sp["qty"] = sp["quantity"].astype(int)
    sp = sp.groupby(["set_num", "part", "color_id"], as_index=False)["qty"].sum()

    sets = sets[["set_num", "name", "year", "num_parts", "img_url"]].copy()
    sets["year"] = pd.to_numeric(sets["year"], errors="coerce")
    sets["num_parts"] = pd.to_numeric(sets["num_parts"], errors="coerce")
    return sp, sets


def load_prepared(path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    bundle = pd.read_pickle(path)
    return bundle["set_parts"], bundle["sets"], bundle.get("colour_names", {})


def what_can_i_build(
    inventory: pd.DataFrame,
    set_parts: pd.DataFrame,
    sets: pd.DataFrame,
    *,
    use_colour: bool = True,
    weighted: bool = True,
    known_parts: set[str] | None = None,
    loose_parts: bool = True,
    min_parts: int = 5,
    max_parts: int = 500,
    top: int = 20,
) -> pd.DataFrame:
    """Rank sets by how much of them we own.

    known_parts: if given, only those part lines count (the parts our model can
    recognise). Otherwise almost every set looks impossible, simply because
    it contains parts outside our 50 classes. `recognisable` shows how much
    of the set that is.
    """
    keys = ["p", "color_id"] if use_colour else ["p"]
    have = _have(inventory, keys, use_colour, weighted, loose_parts)

    sp = set_parts.copy()
    sp["p"] = sp["part"].map(lambda p: norm_part(p, loose_parts))
    sp = sp.groupby(["set_num", *keys], as_index=False)["qty"].sum()

    # restrict to reasonable sets
    ok = sets[(sets["num_parts"] >= min_parts) & (sets["num_parts"] <= max_parts)]["set_num"]
    sp = sp[sp["set_num"].isin(ok)]

    total_need = sp.groupby("set_num")["qty"].sum().rename("all_parts")
    if known_parts is not None:
        known = {norm_part(p, loose_parts) for p in known_parts}
        sp = sp[sp["p"].isin(known)]
    if sp.empty:
        return pd.DataFrame()
    if use_colour:
        sp["color_id"] = sp["color_id"].astype("Int64")

    m = sp.merge(have, on=keys, how="left")
    m["have"] = m["have"].fillna(0.0)
    m["covered"] = m[["qty", "have"]].min(axis=1)
    m["missing"] = (m["qty"] - m["have"]).clip(lower=0).round().astype(int)

    res = m.groupby("set_num").agg(need=("qty", "sum"), covered=("covered", "sum"), missing=("missing", "sum"))
    res = res.join(total_need)
    res["coverage"] = (res["covered"] / res["need"]).round(3)
    res["recognisable"] = (res["need"] / res["all_parts"]).round(3)
    res = res.reset_index().merge(sets, on="set_num", how="left")
    res = res.sort_values(["coverage", "need"], ascending=[False, False]).head(top)
    return res[["set_num", "name", "year", "num_parts", "need", "coverage", "missing", "recognisable", "img_url"]]


def _have(inventory, keys, use_colour, weighted, loose_parts):
    have_col = "expected" if weighted else "count"
    inv = inventory.dropna(subset=["part"]).copy()
    inv["p"] = inv["part"].map(lambda p: norm_part(p, loose_parts))
    if use_colour:
        inv["color_id"] = pd.to_numeric(inv["colour_id"], errors="coerce").astype("Int64")
    return inv.groupby(keys, as_index=False)[have_col].sum().rename(columns={have_col: "have"})


def missing_parts(set_num: str, inventory: pd.DataFrame, set_parts: pd.DataFrame, *, use_colour=True, weighted=True, loose_parts=True) -> pd.DataFrame:
    """Exactly which bricks are still missing for one set."""
    keys = ["p", "color_id"] if use_colour else ["p"]
    have = _have(inventory, keys, use_colour, weighted, loose_parts)
    sp = set_parts[set_parts["set_num"] == set_num].copy()
    sp["p"] = sp["part"].map(lambda p: norm_part(p, loose_parts))
    if use_colour:
        sp["color_id"] = sp["color_id"].astype("Int64")
    else:
        sp = sp.groupby(["set_num", "p"], as_index=False).agg(part=("part", "first"), qty=("qty", "sum"))
    m = sp.merge(have, on=keys, how="left").fillna({"have": 0.0})
    m["have"] = m["have"].round(2)
    m["missing"] = (m["qty"] - m["have"]).clip(lower=0).round().astype(int)
    return m[m["missing"] > 0].drop(columns=["p"]).sort_values("missing", ascending=False).reset_index(drop=True)
