"""Streamlit dashboard over a gate report folder.

    pip install -e ".[dashboard]"
    streamlit run app/dashboard.py -- --report demo/reports/current_vs_candidate

If you pass --images (the BDD100K frames root) the clip review tab draws the
labelled boxes on the actual frames, marked by whether each model found them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from releasegate import db, plots

CURRENT, CANDIDATE = plots.CURRENT, plots.CANDIDATE


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="demo/reports/current_vs_candidate")
    p.add_argument("--images", default="")
    return p.parse_args(sys.argv[1:])


@st.cache_resource
def load(report: str):
    con = db.connect(report)
    tables = {
        name: con.execute(f"SELECT * FROM {name}").df()
        for name in db.VIEWS
        if (Path(report) / f"{name}.parquet").exists()
    }
    return con, tables


def decision_tab(t, report):
    checks = t["checks"]
    blocking = checks[checks["blocking"]]
    decision = (
        "HOLD"
        if (blocking["verdict"] == "FAIL").any()
        else "NEEDS MORE DATA"
        if (blocking["verdict"] == "INCONCLUSIVE").any()
        else "SHIP"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", decision)
    c2.metric("Blocking checks failed", int((blocking["verdict"] == "FAIL").sum()))
    c3.metric("Inconclusive", int((blocking["verdict"] == "INCONCLUSIVE").sum()))
    fig = Path(report) / "figures" / "gate_checks.png"
    if fig.exists():
        st.image(str(fig))
    show = checks[
        [
            "check",
            "current",
            "candidate",
            "diff",
            "diff_lo",
            "diff_hi",
            "margin",
            "n_clips",
            "verdict",
            "blocking",
        ]
    ]
    st.dataframe(show, hide_index=True, width="stretch")
    summary = Path(report) / "summary.md"
    if summary.exists():
        with st.expander("Summary for the review meeting"):
            st.markdown(summary.read_text().split("![")[0])


def scenario_tab(t, con):
    obj = t["objects"]
    c1, c2 = st.columns(2)
    cls = c1.selectbox(
        "Class",
        sorted(obj["cls"].unique()),
        index=sorted(obj["cls"].unique()).index("pedestrian")
        if "pedestrian" in set(obj["cls"])
        else 0,
    )
    by = c2.multiselect(
        "Split by",
        ["timeofday", "weather", "scene", "size", "occluded", "far", "in_path", "crowded"],
        default=["timeofday", "size"],
    )
    o = obj[obj["cls"] == cls]
    if by:
        r = o.groupby(by + ["model"], observed=True)["det"].agg(["mean", "size"]).unstack("model")
        r.columns = [f"{a}_{b}" for a, b in r.columns]
        r = r.rename(
            columns={
                "mean_current": "recall current",
                "mean_candidate": "recall candidate",
                "size_current": "boxes",
            }
        ).drop(columns="size_candidate")
        r["change (pts)"] = 100 * (r["recall candidate"] - r["recall current"])
        r = r.reset_index().sort_values("change (pts)")
        st.dataframe(r, hide_index=True, width="stretch")
        st.caption(
            "Box-level splits, no intervals: use them to find a slice, "
            "then add it as a gate check to get a real verdict."
        )
    st.subheader("Biggest paired drops (all classes)")
    st.dataframe(db.run(con, "biggest_regressions"), hide_index=True, width="stretch")


def time_tab(t, con):
    trk = t["tracks"]
    ped = trk[(trk["cls"] == "pedestrian") & trk["detected_within_1s"].notna()]
    grid = np.linspace(0, 3, 31)
    rows = {}
    for m in ("current", "candidate"):
        for night in (False, True):
            x = ped[(ped["model"] == m) & (ped["night"] == night)]["ttfd_s"].fillna(np.inf)
            rows[f"{m}, {'night' if night else 'day'}"] = [(x <= g).mean() for g in grid]
    st.subheader("Share of pedestrians detected within t seconds of appearing")
    st.line_chart(
        pd.DataFrame(rows, index=grid),
        x_label="seconds",
        y_label="share detected",
        color=[CURRENT, "#86b6ef", CANDIDATE, "#f2a27f"],
    )
    st.subheader("In-path pedestrians the candidate saw too late")
    st.dataframe(db.run(con, "late_pedestrians"), hide_index=True, width="stretch")
    st.subheader("Objects the candidate keeps dropping")
    st.dataframe(db.run(con, "dropout_hotspots"), hide_index=True, width="stretch")


def map_tab(t):
    v = t.get("videos")
    obj = t["objects"]
    if v is None or "lat" not in v or v["lat"].isna().all():
        # Waymo gives a city per segment instead of GPS points.
        if "location" in obj and (obj["location"] != "undefined").any():
            st.subheader("Pedestrian recall by city")
            st.dataframe(
                db.run(load(st.session_state["report"])[0], "recall_by_location"),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No GPS in this dataset. Ingest with --info to get clip locations.")
        return
    ped = obj[obj["cls"] == "pedestrian"]
    r = ped.groupby(["video", "model"])["det"].mean().unstack()
    r["diff"] = 100 * (r["candidate"] - r["current"])
    m = v.set_index("video")[["lat", "lon"]].join(r["diff"]).dropna().reset_index()
    lim = max(5.0, float(np.nanpercentile(m["diff"].abs(), 95)))
    x = np.clip(m["diff"] / lim, -1, 1)
    # red = candidate worse, blue = better, gray = no change
    # interpolate gray (#f0efec) -> red (#e34948) or blue (#2a78d6)
    m["r"] = np.where(x < 0, 240 - 13 * -x, 240 - 198 * x).astype(int)
    m["g"] = np.where(x < 0, 239 - 166 * -x, 239 - 119 * x).astype(int)
    m["b"] = np.where(x < 0, 236 - 164 * -x, 236 - 22 * x).astype(int)
    import pydeck as pdk

    layer = pdk.Layer(
        "ScatterplotLayer",
        m,
        get_position="[lon, lat]",
        get_fill_color="[r, g, b]",
        get_radius=400,
        pickable=True,
        stroked=True,
        get_line_color=[80, 80, 80],
        line_width_min_pixels=1,
    )
    view = pdk.ViewState(
        latitude=float(m["lat"].median()), longitude=float(m["lon"].median()), zoom=8
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "{video}\nchange in pedestrian recall: {diff} pts"},
        )
    )
    st.caption(
        f"Each dot is a clip. Red: candidate finds fewer pedestrians, blue: more "
        f"(colour saturates at ±{lim:.0f} pts)."
    )


def clip_tab(t, con, image_root):
    clips = db.run(con, "failure_clips")
    if clips.empty:
        st.info("No pedestrian differences to review.")
        return
    video = st.selectbox("Clip (most pedestrians lost first)", clips["video"])
    obj = t["objects"]
    o = obj[(obj["video"] == video) & (obj["cls"] == "pedestrian")]
    strip = o.pivot_table(
        index="track_id", columns=["model", "frame_index"], values="det", aggfunc="first"
    )
    st.caption("Each row is a pedestrian, each column a frame. 1 = detected.")
    for m in ("current", "candidate"):
        if m in strip.columns.get_level_values(0):
            st.write(m)
            st.dataframe(strip[m].astype("Int64"), width="stretch")

    frames = sorted(o["frame_index"].unique())
    if not frames:
        return
    k = st.select_slider("Frame", frames)
    if not image_root:
        st.caption("Pass --images to see the frame itself.")
        return
    from PIL import Image, ImageDraw

    fr = o[o["frame_index"] == k]
    path = Path(image_root) / fr["image_id"].iloc[0]
    if not path.exists():
        st.warning(f"missing {path}")
        return
    cols = st.columns(2)
    for col, m in zip(cols, ("current", "candidate"), strict=True):
        im = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for r in fr[fr["model"] == m].itertuples():
            colour = (42, 120, 214) if r.det else (208, 59, 59)
            draw.rectangle([r.x1, r.y1, r.x2, r.y2], outline=colour, width=3)
            draw.text((r.x1, max(r.y1 - 12, 0)), "found" if r.det else "MISSED", fill=colour)
        col.image(im, caption=m)


def main():
    a = args()
    st.set_page_config(page_title="Perception release gate", layout="wide")
    report = st.sidebar.text_input("Report folder", a.report)
    image_root = st.sidebar.text_input("Frames root (optional)", a.images)
    if not (Path(report) / "checks.parquet").exists():
        st.error(f"No gate report in {report}. Run `releasegate demo` or `releasegate gate` first.")
        return
    st.session_state["report"] = report
    con, t = load(report)
    st.title("Perception release gate")
    tabs = st.tabs(["Decision", "Scenarios", "Over time", "Map", "Clip review"])
    with tabs[0]:
        decision_tab(t, report)
    with tabs[1]:
        scenario_tab(t, con)
    with tabs[2]:
        time_tab(t, con)
    with tabs[3]:
        map_tab(t)
    with tabs[4]:
        clip_tab(t, con, image_root)


main()
