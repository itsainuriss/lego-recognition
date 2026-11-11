"""Streamlit app: photo in, brick list out, "What can I build?"

    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lego import inventory as I
from lego import sets as S
from lego.config import ROOT, load_config
from lego.data import part_id
from lego.pipeline import Pipeline

st.set_page_config(page_title="LEGO Brick Recognition", layout="wide")


@st.cache_resource
def get_pipeline():
    return Pipeline(load_config())


@st.cache_resource
def get_sets(path):
    return S.load_prepared(path)


pipe = get_pipeline()
st.title("LEGO Brick Recognition")
st.caption("Photo in, brick list out, then: what can I build?")

with st.sidebar:
    st.header("Settings")
    sets_path = st.text_input("Rebrickable data", str(ROOT / "data/rebrickable/prepared.pkl"))
    use_colour = st.toggle("Colour must match", True)
    weighted = st.toggle("Unsure bricks count less", True)
    known_only = st.toggle("Only parts our model knows", True,
                           help="Otherwise every set looks impossible because of parts outside our 50 classes.")
    max_parts = st.slider("Max set size", 10, 1000, 300)
    if pipe.shape is None:
        st.warning("No shape model loaded - parts will be '?'. Train one (see README).")

photo = st.file_uploader("Photo of loose bricks (plain background, grey card in the corner)", type=["jpg", "jpeg", "png"])
if not photo:
    st.info("Upload a photo to start.")
    st.stop()

key = photo.name + str(photo.size)
if st.session_state.get("key") != key:
    with st.spinner("Finding bricks..."):
        res = pipe.run(photo.getvalue())
    st.session_state.update(key=key, res=res, det=res.detections_df.assign(confirmed=False))
res = st.session_state["res"]

c1, c2 = st.columns([3, 2])
with c1:
    st.image(res.annotated, caption=f"{len(res.detections)} bricks - green: sure, orange: unsure", use_container_width=True)
with c2:
    st.subheader("Unsure bricks")
    unsure = [d for d in res.detections if d.unsure]
    if not unsure:
        st.success("None.")
    for d in unsure[:12]:
        a, b = st.columns([1, 2])
        a.image(d.crop, width=90)
        b.markdown(f"**#{d.idx}** {part_id(d.part) if d.part else '?'} ({d.part_conf:.0%}), "
                   f"{d.colour_name} (ΔE {d.colour_delta_e:.0f})  \nalternatives: "
                   + ", ".join(f"{part_id(n)} {p:.0%}" for n, p in d.part_top[1:3]))

st.subheader("1 · Detected bricks (edit to correct, tick 'confirmed')")
colour_names = pipe.palette.names
det = st.data_editor(
    st.session_state["det"],
    column_order=["idx", "part", "colour", "part_conf", "alt_parts", "colour_conf", "unsure", "confirmed"],
    column_config={
        "colour": st.column_config.SelectboxColumn(options=colour_names),
        "part_conf": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        "colour_conf": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    },
    disabled=["idx", "part_conf", "alt_parts", "colour_conf", "unsure"],
    hide_index=True, use_container_width=True,
)
name_to_id = dict(zip(pipe.palette.names, pipe.palette.ids))
det = det.assign(colour_id=det["colour"].map(name_to_id))

st.subheader("2 · Inventory")
inv = I.aggregate(det)
st.dataframe(inv, hide_index=True, use_container_width=True)
st.download_button("Download inventory CSV", inv.to_csv(index=False), "inventory.csv")

st.subheader("3 · What can I build?")
try:
    sp, sets, rb_colours = get_sets(sets_path)
except FileNotFoundError:
    st.warning("Rebrickable data not found - run scripts/prepare_rebrickable.py (see README).")
    st.stop()
known = {part_id(c) for c in pipe.shape.classes} if (known_only and pipe.shape is not None) else None
top = S.what_can_i_build(inv, sp, sets, use_colour=use_colour, weighted=weighted, known_parts=known, max_parts=max_parts)
if top.empty:
    st.info("No matching sets.")
    st.stop()
st.dataframe(
    top.assign(coverage=(top["coverage"] * 100).round()), hide_index=True, use_container_width=True,
    column_config={
        "coverage": st.column_config.ProgressColumn("we own", min_value=0, max_value=100, format="%d%%"),
        "recognisable": st.column_config.NumberColumn("share of set we can recognise", format="%.2f"),
        "img_url": st.column_config.ImageColumn("set"),
    },
)
pick = st.selectbox("Missing parts for set", top["set_num"] + " · " + top["name"].fillna(""))
miss = S.missing_parts(pick.split(" · ")[0], inv, sp, use_colour=use_colour, weighted=weighted)
if use_colour and len(miss):
    miss["colour"] = miss["color_id"].map(lambda c: rb_colours.get(int(c), c))
st.dataframe(pd.DataFrame(miss), hide_index=True, use_container_width=True)
