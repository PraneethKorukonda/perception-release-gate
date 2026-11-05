"""Read BDD100K multi-object tracking labels into flat tables.

You have to download BDD100K yourself (free account at
https://bdd-data.berkeley.edu). For this project you need:

    MOT 2020 Images   images/track/val/<video>/<frame>.jpg
    MOT 2020 Labels   labels/box_track_20/val/<video>.json
    Info (optional)   info/100k/val/<video>.json   GPS and speed, see gps.py

The tracking set is what makes the time-based metrics possible: frames are
5 fps and every box has an `id` that stays fixed while the object is visible.

Output tables (parquet):

    images  one row per frame: image_id, video, frame_index, timeofday,
            weather, scene, width, height
    gt      one row per box: gt_id, image_id, video, frame_index, track_id,
            cls, label, ignore, x1..y2, area, size, occluded, truncated
    videos  one row per clip

`cls` is the class a box competes in during matching. Rider and crowd boxes
are ignore regions (ignore=True), and `label` keeps the original category.
`image_id` is "<video>/<frame name>", which is also its path under the
images folder.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from releasegate.boxes import area, size_bucket
from releasegate.categories import map_bdd

BDD_WIDTH, BDD_HEIGHT = 1280, 720
FPS = 5  # the tracking labels are at 5 frames per second

IMAGE_COLUMNS = [
    "image_id",
    "video",
    "frame_index",
    "timeofday",
    "weather",
    "scene",
    "width",
    "height",
]
GT_COLUMNS = [
    "gt_id",
    "image_id",
    "video",
    "frame_index",
    "track_id",
    "cls",
    "label",
    "ignore",
    "x1",
    "y1",
    "x2",
    "y2",
    "area",
    "size",
    "occluded",
    "truncated",
]


def parse_video(frames: list[dict]) -> tuple[list[tuple], list[tuple]]:
    image_rows, gt_rows = [], []
    for frame in frames:
        video = frame.get("videoName") or frame["name"].rsplit("-", 1)[0]
        image_id = f"{video}/{frame['name']}"
        fidx = int(frame.get("frameIndex", frame.get("index", 0)))
        # Some releases put scene attributes on every frame, some on none.
        attrs = frame.get("attributes") or {}
        image_rows.append(
            (
                image_id,
                video,
                fidx,
                attrs.get("timeofday", "undefined"),
                attrs.get("weather", "undefined"),
                attrs.get("scene", "undefined"),
                BDD_WIDTH,
                BDD_HEIGHT,
            )
        )
        for lab in frame.get("labels") or []:
            box = lab.get("box2d")
            if box is None:
                continue
            la = lab.get("attributes") or {}
            eval_cls, ignore_for = map_bdd(lab["category"])
            if la.get("crowd") and eval_cls is not None:
                eval_cls, ignore_for = None, eval_cls
            if eval_cls is None and ignore_for is None:
                continue
            gt_rows.append(
                (
                    image_id,
                    video,
                    fidx,
                    f"{video}:{lab.get('id', '')}",
                    eval_cls or ignore_for,
                    lab["category"],
                    ignore_for is not None,
                    float(box["x1"]),
                    float(box["y1"]),
                    float(box["x2"]),
                    float(box["y2"]),
                    bool(la.get("occluded", False)),
                    bool(la.get("truncated", False)),
                )
            )
    return image_rows, gt_rows


def finish_gt(gt: pd.DataFrame) -> pd.DataFrame:
    """Add ids, area and size bucket, and fix the column order."""
    gt = gt.reset_index(drop=True).copy()
    gt["gt_id"] = np.arange(len(gt), dtype=np.int64)
    boxes = gt[["x1", "y1", "x2", "y2"]].to_numpy(float)
    gt["area"] = area(boxes)
    gt["size"] = size_bucket(gt["area"].to_numpy())
    return gt[GT_COLUMNS]


def summarize_videos(images: pd.DataFrame) -> pd.DataFrame:
    """One row per clip. Scene attributes are the most common value over its frames."""

    def mode(s: pd.Series) -> str:
        return Counter(s).most_common(1)[0][0]

    return (
        images.groupby("video", sort=True)
        .agg(
            n_frames=("image_id", "size"),
            first_frame=("frame_index", "min"),
            timeofday=("timeofday", mode),
            weather=("weather", mode),
            scene=("scene", mode),
        )
        .reset_index()
    )


def load_tracking_dir(label_dir: str | Path, limit: int | None = None, seed: int = 0):
    """Parse every <video>.json in a box_track_20 split folder."""
    files = sorted(Path(label_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no .json label files in {label_dir}")
    if limit is not None and limit < len(files):
        rng = np.random.default_rng(seed)
        files = sorted(rng.choice(files, size=limit, replace=False))

    image_rows, gt_rows = [], []
    for f in files:
        with open(f) as fh:
            i, g = parse_video(json.load(fh))
        image_rows += i
        gt_rows += g

    images = pd.DataFrame(image_rows, columns=IMAGE_COLUMNS)
    images["location"] = "undefined"  # BDD has GPS instead; see gps.py
    gt = pd.DataFrame(
        gt_rows,
        columns=[
            "image_id",
            "video",
            "frame_index",
            "track_id",
            "cls",
            "label",
            "ignore",
            "x1",
            "y1",
            "x2",
            "y2",
            "occluded",
            "truncated",
        ],
    )
    return images, finish_gt(gt), summarize_videos(images)


def save_dataset(out_dir, images, gt, videos) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images.to_parquet(out / "images.parquet", index=False)
    gt.to_parquet(out / "gt.parquet", index=False)
    videos.to_parquet(out / "videos.parquet", index=False)


def load_dataset(data_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir)
    return (
        pd.read_parquet(d / "images.parquet"),
        pd.read_parquet(d / "gt.parquet"),
        pd.read_parquet(d / "videos.parquet"),
    )
