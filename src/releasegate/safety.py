"""From box geometry to "would the car have had room to stop?"

This is the bridge from a perception metric to driving behaviour, and it is a
deliberately rough one. It is a screening metric that ranks models and points
at clips worth a closer look, not a safety case.

Distance. With a pinhole camera, an object of real height H that appears h
pixels tall is about d = f * H / h metres away. BDD100K doesn't publish
per-clip camera intrinsics. f = 1000 px corresponds to a ~65 degree
horizontal field of view on a 1280 px wide frame, which is typical for a
dashcam. Errors here scale every distance by the same factor, so they move
both models together and mostly cancel in a comparison. Truncated boxes are
cut off by the frame edge and would look too far away, so they get no
distance.

Stopping. Once the object is detected the system still needs a reaction
time tau before braking starts, then brakes at a constant deceleration a:

    required distance = v * tau + v^2 / (2 * a)

I use a = 3 m/s^2, a firm but comfortable stop, not an emergency one
(6-8 m/s^2). A perception system that only leaves room for an emergency
stop has already used up the margin the planner needs. With v = 10 m/s
(36 km/h) and tau = 0.5 s that's 21.7 m.

A pedestrian detection is "late" if, at the frame the model first sees them,
they're closer than that. A pedestrian in the path who is never detected is
late by definition. Pedestrians who were already closer than that when they
first became visible are left out: no detector could have done better, so
they'd only dilute the comparison.

"In the path" means the estimated lateral offset comes within 2 m of the
camera axis at some point while the object is visible, so pedestrians who
cross count, not only ones standing in the lane. That ignores where the road
actually goes (curves, turns), which is the biggest weakness of this proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from releasegate.categories import OBJECT_HEIGHT_M

# Used when a clip has no GPS speed.
FALLBACK_SPEED_MPS = {"highway": 25.0, "residential": 8.0}
DEFAULT_SPEED_MPS = 10.0


@dataclass(frozen=True)
class SafetyParams:
    focal_px: float = 1000.0
    image_width: float = 1280.0
    reaction_s: float = 0.5
    decel_mps2: float = 3.0
    path_half_width_m: float = 2.0


def distance_m(cls, height_px, truncated, focal_px=1000.0) -> np.ndarray:
    H = np.array([OBJECT_HEIGHT_M.get(c, np.nan) for c in cls], float)
    h = np.asarray(height_px, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = focal_px * H / h
    return np.where(np.asarray(truncated, bool) | (h <= 0), np.nan, d)


def lateral_m(center_x, dist, focal_px: float = 1000.0, image_width: float = 1280.0):
    return (np.asarray(center_x, float) - image_width / 2) * np.asarray(dist, float) / focal_px


def fill_speed(speed, scene) -> tuple[np.ndarray, np.ndarray]:
    """Replace missing GPS speed with a typical speed for the scene."""
    speed = np.asarray(speed, float)
    fallback = np.array([FALLBACK_SPEED_MPS.get(s, DEFAULT_SPEED_MPS) for s in scene])
    missing = ~np.isfinite(speed)
    return np.where(missing, fallback, speed), np.where(missing, "assumed", "gps")


def stopping_distance(speed_mps, params: SafetyParams = SafetyParams()) -> np.ndarray:
    v = np.asarray(speed_mps, float)
    return v * params.reaction_s + v**2 / (2 * params.decel_mps2)


def add_geometry(obj: pd.DataFrame, params: SafetyParams = SafetyParams()) -> pd.DataFrame:
    """Per box: distance, lateral offset, ego speed."""
    obj = obj.copy()
    h = (obj["y2"] - obj["y1"]).to_numpy(float)
    cx = ((obj["x1"] + obj["x2"]) / 2).to_numpy(float)
    # Waymo ships real intrinsics per camera; use them when the ingest provides them.
    f = obj["focal_px"].to_numpy(float) if "focal_px" in obj else params.focal_px
    cu = obj["center_x_px"].to_numpy(float) if "center_x_px" in obj else params.image_width / 2
    obj["distance_m"] = distance_m(obj["cls"], h, obj["truncated"], f)
    obj["lateral_m"] = (cx - cu) * obj["distance_m"].to_numpy(float) / f
    speed = obj["speed_mps"] if "speed_mps" in obj else np.full(len(obj), np.nan)
    obj["speed_mps"], obj["speed_source"] = fill_speed(speed, obj["scene"])
    return obj
