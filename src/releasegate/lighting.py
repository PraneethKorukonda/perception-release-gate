"""Fill in time of day from the pixels when the labels don't have it.

Some BDD100K tracking label files carry frame attributes (time of day,
weather, scene) and some don't. Night is the most important of the three
for perception, and it's also the easiest to recover: night frames are
dark. I take the mean brightness of the upper half of the frame (sky and
buildings; the lower half has headlights and street lights that make night
frames look brighter than they are) and threshold it.

Thresholds were picked by eye, not tuned. If you have even a few hundred
frames with labels, check the rule against them with `check_rule` before
trusting it, and report the agreement next to any night result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NIGHT_BELOW = 55
DUSK_BELOW = 95


def upper_brightness(path: str | Path) -> float:
    from PIL import Image

    with Image.open(path) as im:
        g = np.asarray(im.convert("L").reduce(4), dtype=float)
    return float(g[: g.shape[0] // 2].mean())


def classify(brightness) -> np.ndarray:
    b = np.asarray(brightness, float)
    return np.where(b < NIGHT_BELOW, "night", np.where(b < DUSK_BELOW, "dawn/dusk", "daytime"))


def estimate_timeofday(images: pd.DataFrame, image_root, stride: int = 10) -> pd.DataFrame:
    """Per clip: median brightness over every `stride`-th frame, then classify.

    Time of day doesn't change within a 40 s clip, so one label per clip is
    both cheaper and less noisy than one per frame.
    """
    sub = images[images["frame_index"] % stride == 0]
    b = [upper_brightness(Path(image_root) / i) for i in sub["image_id"]]
    per_clip = pd.Series(b, index=sub["video"].to_numpy()).groupby(level=0).median()
    return pd.DataFrame(
        {
            "video": per_clip.index,
            "brightness": per_clip.to_numpy(),
            "timeofday_est": classify(per_clip.to_numpy()),
        }
    )


def apply_estimate(images: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    """Use the estimate only where the label is missing."""
    out = images.merge(est[["video", "timeofday_est"]], on="video", how="left")
    missing = out["timeofday"].isin(["undefined", None]) | out["timeofday"].isna()
    out.loc[missing, "timeofday"] = out.loc[missing, "timeofday_est"]
    return out.drop(columns="timeofday_est")


def check_rule(est: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """Confusion table of the brightness rule against clips that do have labels."""
    m = est.merge(labelled[["video", "timeofday"]], on="video")
    m = m[m["timeofday"] != "undefined"]
    return pd.crosstab(m["timeofday"], m["timeofday_est"], margins=True)
