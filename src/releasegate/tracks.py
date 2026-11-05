"""Per-box and per-object tables, including the time-based metrics.

Frame-level recall answers "what fraction of boxes did the model find?". For
a car that's only half the story. What matters is how soon after a
pedestrian becomes visible the model picks them up, and whether it keeps
them or drops them for a frame here and there. Two models can have the same
frame recall and very different behaviour on both.

There is no tracker here on purpose. I use the ground-truth track ids from
BDD100K to link boxes across frames, so these metrics describe the
detector's consistency over time, not a tracker's ability to stitch
detections together.

Definitions (fps = 5 for BDD100K tracking):

time to first detection (s)
    frames from the object's first labelled frame to the first frame the
    model detects it, divided by fps.

detected within 1 s
    the share of objects detected no later than 1 s after they appear.
    Only objects that stay visible for at least 1 s count, otherwise a
    short-lived object that was never detected would look like a miss the
    model had no chance at. That rule depends only on the labels, not on
    the model, so it's the same set of objects for both models.

left-censored
    objects already visible in the first frame of the clip. We don't know
    when they really appeared, so they're excluded from anything about
    first detection.

dropouts
    times the model had an object, lost it on the next labelled frame, and
    picked it up again later. Reported per 100 detected frames. A dropout
    is what makes a downstream tracker coast or spawn a duplicate track.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from releasegate.metrics import detected
from releasegate.safety import SafetyParams, add_geometry, stopping_distance


def object_table(
    gt: pd.DataFrame,
    gt_match: pd.DataFrame,
    images: pd.DataFrame,
    score_thr: float,
    params: SafetyParams = SafetyParams(),
) -> pd.DataFrame:
    """One row per scored box: geometry, scene attributes and `det`."""
    keep = [
        "gt_id",
        "video",
        "frame_index",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "area",
        "size",
        "occluded",
        "truncated",
    ]
    obj = gt_match[["gt_id", "image_id", "cls", "score"]].merge(gt[keep], on="gt_id", how="left")
    img_cols = [c for c in images.columns if c not in ("video", "frame_index", "width", "height")]
    obj = obj.merge(images[img_cols], on="image_id", how="left")
    obj["det"] = detected(obj, score_thr)
    obj = add_geometry(obj, params)
    return obj


def track_table(
    obj: pd.DataFrame, fps: float = 5.0, params: SafetyParams = SafetyParams()
) -> pd.DataFrame:
    """One row per ground-truth object (track)."""
    o = obj.sort_values(["track_id", "frame_index"], kind="mergesort").reset_index(drop=True)
    tid = o["track_id"].to_numpy()
    fi = o["frame_index"].to_numpy()
    det = o["det"].to_numpy(bool)

    first_frame_of_video = o.groupby("video")["frame_index"].transform("min").to_numpy()
    same_track_next = np.r_[tid[1:] == tid[:-1], False]

    g = o.groupby("track_id", sort=False)
    t = g.agg(
        video=("video", "first"),
        cls=("cls", "first"),
        n_frames=("frame_index", "size"),
        first_frame=("frame_index", "min"),
        last_frame=("frame_index", "max"),
        n_det_frames=("det", "sum"),
        timeofday=("timeofday", "first"),
        weather=("weather", "first"),
        scene=("scene", "first"),
        first_area=("area", "first"),
        min_lateral_m=("lateral_m", lambda s: s.abs().min()),
    )

    # first detected frame per track
    det_frames = pd.Series(np.where(det, fi, np.nan), index=o.index)
    t["first_det_frame"] = det_frames.groupby(tid).min().reindex(t.index)
    t["last_det_frame"] = det_frames.groupby(tid).max().reindex(t.index)

    # dropouts: detected now, missed on the next labelled frame, found again later
    last_det = pd.Series(t["last_det_frame"]).reindex(tid).to_numpy()
    drop = det & same_track_next & np.r_[~det[1:], True] & (fi < last_det)
    t["dropouts"] = pd.Series(drop, index=o.index).groupby(tid).sum().reindex(t.index)

    t["left_censored"] = (
        pd.Series(fi == first_frame_of_video, index=o.index).groupby(tid).first().reindex(t.index)
    )
    span = t["last_frame"] - t["first_frame"]
    t["ttfd_s"] = (t["first_det_frame"] - t["first_frame"]) / fps
    eligible = ~t["left_censored"] & (span >= fps)
    within = t["ttfd_s"].le(1.0)
    t["detected_within_1s"] = np.where(eligible, within.astype(float), np.nan)

    # Stopping margin at the first detection, for objects in the path.
    at_det = o[det].groupby("track_id").first()
    t["distance_at_det_m"] = at_det["distance_m"].reindex(t.index)
    t["speed_at_det_mps"] = at_det["speed_mps"].reindex(t.index)
    t["distance_at_appear_m"] = g["distance_m"].first()
    t["in_path"] = t["min_lateral_m"] < params.path_half_width_m
    t["stop_margin_m"] = t["distance_at_det_m"] - stopping_distance(t["speed_at_det_mps"], params)
    late = t["first_det_frame"].isna() | (t["stop_margin_m"] < 0)
    # A margin can only be judged if we know how far away the object was.
    speed_at_appear = g["speed_mps"].first()
    avoidable = t["distance_at_appear_m"] > stopping_distance(speed_at_appear, params)
    judged = (
        t["in_path"]
        & ~t["left_censored"]
        & avoidable
        & (t["first_det_frame"].isna() | t["distance_at_det_m"].notna())
    )
    t["late_in_path"] = np.where(judged, late.astype(float), np.nan)

    return t.reset_index().rename(columns={"index": "track_id"})
