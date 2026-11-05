"""Scenario tags and scenario sets.

Tags are plain boolean columns on the box and object tables, so gate checks
and SQL can filter on them directly (`where = "night and in_path"`).

A scenario set is a named, versioned list of frames, stored as a small JSON
manifest. Rule-based sets come from the labels; mined sets come from a CLIP
text search (mining.py) and cover things the labels don't record, like
"pedestrian stepping out between parked cars". Keeping the frame list in a
file, with a fingerprint, means the number you report next month is computed
on the same frames as this month unless you deliberately changed the set,
and `diff_sets` tells you exactly what changed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

CROWDED_MIN_OBJECTS = 15
FAR_M = 30.0


def tag_objects(obj: pd.DataFrame, tracks: pd.DataFrame | None = None) -> pd.DataFrame:
    obj = obj.copy()
    obj["night"] = obj["timeofday"].eq("night")
    obj["rain"] = obj["weather"].isin(["rainy", "snowy"])
    obj["highway"] = obj["scene"].eq("highway")
    obj["small"] = obj["size"].eq("small")
    obj["far"] = obj["distance_m"].gt(FAR_M)
    obj["crowded"] = obj.groupby("image_id")["gt_id"].transform("size").ge(CROWDED_MIN_OBJECTS)
    if tracks is not None:
        obj = obj.merge(tracks[["track_id", "in_path"]], on="track_id", how="left")
        obj["in_path"] = obj["in_path"].fillna(False).astype(bool)
    return obj


def tag_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    tracks = tracks.copy()
    tracks["night"] = tracks["timeofday"].eq("night")
    tracks["rain"] = tracks["weather"].isin(["rainy", "snowy"])
    tracks["highway"] = tracks["scene"].eq("highway")
    tracks["far_at_appearance"] = tracks["distance_at_appear_m"].gt(FAR_M)
    return tracks


# ---- scenario sets ----------------------------------------------------------


def fingerprint(ids) -> str:
    h = hashlib.sha256("\n".join(sorted(map(str, ids))).encode())
    return h.hexdigest()[:16]


def make_set(name: str, image_ids, method: str, definition: str, notes: str = "") -> dict:
    ids = sorted(set(map(str, image_ids)))
    return {
        "name": name,
        "version": 1,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": method,
        "definition": definition,
        "notes": notes,
        "n_frames": len(ids),
        "n_clips": len({i.split("/", 1)[0] for i in ids}),
        "fingerprint": fingerprint(ids),
        "history": [],
        "image_ids": ids,
    }


def rule_set(obj: pd.DataFrame, name: str, where: str, notes: str = "") -> dict:
    """Every frame that contains at least one box matching `where`."""
    ids = obj.query(where)["image_id"].unique()
    return make_set(name, ids, "rule", where, notes)


def save_set(s: dict, path: str | Path) -> dict:
    """Write a set. If a different version already exists there, bump the version
    and keep a one-line record of the old one."""
    path = Path(path)
    if path.exists():
        old = load_set(path)
        if old["fingerprint"] == s["fingerprint"]:
            return old
        s = dict(s)
        s["version"] = old["version"] + 1
        s["history"] = old.get("history", []) + [
            {k: old[k] for k in ("version", "created", "n_frames", "fingerprint", "definition")}
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s, indent=1))
    return s


def load_set(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def diff_sets(old: dict, new: dict) -> dict:
    a, b = set(old["image_ids"]), set(new["image_ids"])
    return {"added": len(b - a), "removed": len(a - b), "kept": len(a & b)}


def apply_sets(df: pd.DataFrame, sets: list[dict]) -> pd.DataFrame:
    """Add a boolean column `set_<name>` for each scenario set."""
    df = df.copy()
    for s in sets:
        df[f"set_{s['name']}"] = df["image_id"].isin(set(s["image_ids"]))
    return df


def expand_window(image_ids, images: pd.DataFrame, frames: int) -> np.ndarray:
    """Grow a list of frames to include +-`frames` neighbours in the same clip.

    Mining usually embeds every 5th frame to save time; this fills the gaps.
    """
    hits = images[images["image_id"].isin(set(image_ids))][["video", "frame_index"]]
    if hits.empty:
        return np.array([], dtype=object)
    m = images.merge(hits, on="video", suffixes=("", "_hit"))
    m = m[(m["frame_index"] - m["frame_index_hit"]).abs() <= frames]
    return m["image_id"].unique()
