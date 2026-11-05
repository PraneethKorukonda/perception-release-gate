"""Attach GPS position and ego speed to every frame.

BDD100K ships an "info" file per video with phone GPS at roughly 1 Hz:

    {"startTime": 1503828163000, "endTime": ...,
     "locations": [{"timestamp": ..., "latitude": ..., "longitude": ...,
                    "speed": ..., "course": ...}, ...]}

Speed is in m/s. Tracking frames are 5 fps from the start of the clip, so
frame k is at startTime + 200*k ms; I linearly interpolate between GPS fixes.
That alignment is approximate (phone clocks drift a bit), which is fine for
"how fast was the car going", not for anything precise.

Ego speed matters because a detection that comes 0.4 s late costs about
4 m at city speed and 11 m on a highway. safety.py uses it.

If a clip has no info file, its frames get NaN and safety.py falls back to a
typical speed for the scene type.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from releasegate.bdd import FPS


def read_info(path: str | Path) -> tuple[int | None, pd.DataFrame]:
    with open(path) as f:
        info = json.load(f)
    locs = pd.DataFrame(info.get("locations") or [])
    if locs.empty:
        return info.get("startTime"), locs
    locs = locs.dropna(subset=["timestamp"]).sort_values("timestamp")
    # GPS sometimes reports -1 when it has no speed fix.
    if "speed" in locs:
        locs.loc[locs["speed"] < 0, "speed"] = np.nan
    return info.get("startTime", int(locs["timestamp"].iloc[0])), locs


def frame_gps(images: pd.DataFrame, info_dir: str | Path) -> pd.DataFrame:
    """Return images with lat, lon, speed_mps columns added."""
    info_dir = Path(info_dir)
    out = images.copy()
    for col in ("lat", "lon", "speed_mps"):
        out[col] = np.nan

    for video, idx in out.groupby("video").groups.items():
        path = info_dir / f"{video}.json"
        if not path.exists():
            continue
        start, locs = read_info(path)
        if locs.empty or start is None:
            continue
        t = start + 1000.0 / FPS * out.loc[idx, "frame_index"].to_numpy(float)
        ts = locs["timestamp"].to_numpy(float)
        for src, dst in (("latitude", "lat"), ("longitude", "lon"), ("speed", "speed_mps")):
            if src not in locs:
                continue
            v = locs[src].to_numpy(float)
            ok = np.isfinite(v)
            if ok.sum() >= 1:
                out.loc[idx, dst] = np.interp(t, ts[ok], v[ok])
    return out


def video_gps(images: pd.DataFrame) -> pd.DataFrame:
    """Median position and speed per clip."""
    return (
        images.groupby("video")
        .agg(lat=("lat", "median"), lon=("lon", "median"), speed_mps=("speed_mps", "median"))
        .reset_index()
    )
