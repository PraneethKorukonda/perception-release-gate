"""Read the Waymo Open Dataset (perception v2, parquet) into the same tables
the rest of the package uses.

Waymo splits each 20 s segment into one parquet file per "component". This
reads five of them, front camera only:

    camera_box          2D boxes with a per-camera tracking id (camera_object_id)
    camera_calibration  intrinsics, so distance uses the real focal length
    stats               time of day, weather and location for every frame
    vehicle_pose        world-from-vehicle transform, which gives ego speed
    camera_image        JPEG bytes (only needed to run a detector, see extract_images)

Expected layout (what `gsutil cp` or a browser download gives you):

    <root>/<component>/<segment>.parquet
    <root>/<component>/<split>-<component>-<segment>.parquet   (browser download names)

Frames are 10 Hz. Ego speed comes from differencing the vehicle pose over
time, which is more accurate than BDD's phone GPS.

Differences from BDD worth knowing:
  * Classes are vehicle / pedestrian / cyclist / sign. Signs are dropped,
    cyclists become ignore regions for pedestrians (categories.py has why).
  * There's no occlusion flag, and the camera boxes' difficulty level is
    LEVEL_2 for every box I've looked at, so it carries no information.
    `occluded` is always False for Waymo; checks that use it just don't
    split anything.
  * Waymo labels objects much further out than BDD: in the segments I
    checked, the median front-camera box is ~27 px tall and half the
    pedestrians are 80 m+ away. Overall recall is dominated by those far
    boxes, so the Waymo gate config scores near-field pedestrians
    (within 40 m) separately. That's the range that matters for stopping.
  * Locations are cities (phx, sf, other), not GPS points, so the map view
    becomes a per-city breakdown.

The Waymo license is non-commercial and doesn't allow re-sharing the data, so
nothing produced from it is committed to the repo except aggregate numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from releasegate.bdd import finish_gt, summarize_videos

FPS = 10
FRONT = 1

# Waymo's Type enum for camera labels.
TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_SIGN, TYPE_CYCLIST = 1, 2, 3, 4

TIME_OF_DAY = {"Day": "daytime", "Night": "night", "Dawn/Dusk": "dawn/dusk"}
WEATHER = {"sunny": "clear", "rain": "rainy"}

BOX = "[CameraBoxComponent]"
CAL = "[CameraCalibrationComponent]"
STATS = "[StatsComponent]"
POSE = "[VehiclePoseComponent]"
IMG = "[CameraImageComponent]"
SEG = "key.segment_context_name"
TS = "key.frame_timestamp_micros"


def segment_files(root: str | Path, component: str) -> dict[str, Path]:
    """segment name -> file, accepting both naming styles."""
    out = {}
    for f in sorted((Path(root) / component).glob("*.parquet")):
        name = f.stem
        prefix = f"-{component}-"
        if prefix in name:
            name = name.split(prefix, 1)[1]
        out[name] = f
    return out


def available_segments(root) -> list[str]:
    """Segments that have every label component (images are checked separately)."""
    sets = [
        set(segment_files(root, c))
        for c in ("camera_box", "camera_calibration", "stats", "vehicle_pose")
    ]
    return sorted(set.intersection(*sets))


def _speed_from_pose(pose: pd.DataFrame) -> pd.Series:
    """Ego speed (m/s) per frame from the world-from-vehicle translation."""
    pose = pose.sort_values(TS)
    m = np.stack(pose[f"{POSE}.world_from_vehicle.transform"].to_numpy())
    xyz = m[:, [3, 7, 11]]  # 4x4 row-major: translation is the last column
    t = pose[TS].to_numpy(float) / 1e6
    if len(t) < 2:
        return pd.Series(np.nan, index=pose[TS].to_numpy())
    v = np.gradient(xyz, t, axis=0)
    return pd.Series(np.linalg.norm(v[:, :2], axis=1), index=pose[TS].to_numpy())


def _mode(s: pd.Series):
    return s.mode().iloc[0] if s.notna().any() else "undefined"


def load_segment(root, segment: str, camera: int = FRONT):
    """Returns (images, gt rows before ids) for one segment."""
    read = lambda c: pd.read_parquet(segment_files(root, c)[segment])  # noqa: E731

    stats = read("stats")
    pose = read("vehicle_pose")
    cal = read("camera_calibration")
    box = read("camera_box")

    cam = cal[cal["key.camera_name"] == camera].iloc[0]
    width, height = int(cam[f"{CAL}.width"]), int(cam[f"{CAL}.height"])

    frames = stats[[TS]].drop_duplicates().sort_values(TS).reset_index(drop=True)
    frames["frame_index"] = np.arange(len(frames))
    speed = _speed_from_pose(pose)

    tod = TIME_OF_DAY.get(_mode(stats[f"{STATS}.time_of_day"]), "undefined")
    weather = _mode(stats[f"{STATS}.weather"])
    location = _mode(stats[f"{STATS}.location"]).removeprefix("location_")

    images = pd.DataFrame(
        {
            "image_id": [f"{segment}/{t}.jpg" for t in frames[TS]],
            "video": segment,
            "frame_index": frames["frame_index"],
            "timeofday": tod,
            "weather": WEATHER.get(weather, weather),
            "scene": "undefined",
            "location": location,
            "width": width,
            "height": height,
            "speed_mps": speed.reindex(frames[TS]).to_numpy(),
            "focal_px": float(cam[f"{CAL}.intrinsic.f_v"]),
            "center_x_px": float(cam[f"{CAL}.intrinsic.c_u"]),
            "timestamp_us": frames[TS].to_numpy(),
        }
    )

    b = box[(box["key.camera_name"] == camera) & (box[f"{BOX}.type"] != TYPE_SIGN)].copy()
    b = b.merge(frames, on=TS, how="inner")
    cx, cy = b[f"{BOX}.box.center.x"], b[f"{BOX}.box.center.y"]
    w, h = b[f"{BOX}.box.size.x"], b[f"{BOX}.box.size.y"]
    x1, x2 = (cx - w / 2).clip(0, width), (cx + w / 2).clip(0, width)
    y1, y2 = (cy - h / 2).clip(0, height), (cy + h / 2).clip(0, height)
    typ = b[f"{BOX}.type"].to_numpy()
    is_cyclist = typ == TYPE_CYCLIST
    gt = pd.DataFrame(
        {
            "image_id": [f"{segment}/{t}.jpg" for t in b[TS]],
            "video": segment,
            "frame_index": b["frame_index"].to_numpy(),
            "track_id": segment + ":" + b["key.camera_object_id"].astype(str),
            "cls": np.where(typ == TYPE_VEHICLE, "vehicle", "pedestrian"),
            "label": np.select(
                [typ == TYPE_VEHICLE, is_cyclist], ["vehicle", "cyclist"], "pedestrian"
            ),
            "ignore": is_cyclist,
            "x1": x1.to_numpy(),
            "y1": y1.to_numpy(),
            "x2": x2.to_numpy(),
            "y2": y2.to_numpy(),
            "occluded": False,
            # a box that runs into the frame edge was cut off by it
            "truncated": (
                (x1 <= 1) | (y1 <= 1) | (x2 >= width - 1) | (y2 >= height - 1)
            ).to_numpy(),
        }
    )
    return images, gt


def load_waymo(root, segments: list[str] | None = None, camera: int = FRONT):
    segments = segments or available_segments(root)
    if not segments:
        raise FileNotFoundError(
            f"no complete segments under {root}: need camera_box, camera_calibration, "
            "stats and vehicle_pose for each one"
        )
    parts = [load_segment(root, s, camera) for s in segments]
    images = pd.concat([p[0] for p in parts], ignore_index=True)
    gt = finish_gt(pd.concat([p[1] for p in parts], ignore_index=True))
    videos = summarize_videos(images).merge(
        images.groupby("video")
        .agg(location=("location", "first"), speed_mps=("speed_mps", "median"))
        .reset_index(),
        on="video",
    )
    return images, gt, videos


def extract_images(root, images: pd.DataFrame, out_dir, camera: int = FRONT) -> int:
    """Write each front-camera JPEG to <out_dir>/<segment>/<timestamp>.jpg.

    camera_image files are ~350 MB per segment because they hold all five
    cameras. Reading with a filter on camera_name keeps memory to one camera.
    Returns the number of images written; existing files are skipped, so it
    can be rerun after a partial download.
    """
    import pyarrow.parquet as pq

    files = segment_files(root, "camera_image")
    out = Path(out_dir)
    written = 0
    for seg in images["video"].unique():
        if seg not in files:
            print(f"  no camera_image file for {seg}, skipping")
            continue
        (out / seg).mkdir(parents=True, exist_ok=True)
        t = pq.read_table(
            files[seg],
            columns=[TS, "key.camera_name", f"{IMG}.image"],
            filters=[("key.camera_name", "=", camera)],
        )
        for ts, data in zip(
            t.column(TS).to_pylist(), t.column(f"{IMG}.image").to_pylist(), strict=True
        ):
            p = out / seg / f"{ts}.jpg"
            if not p.exists():
                p.write_bytes(data)
                written += 1
    return written


# ---- choosing which segments to download ------------------------------------


def summarize_stats(stats_dir) -> pd.DataFrame:
    """One row per segment from the (tiny) stats files: conditions and how many
    pedestrians the front camera sees. Used to pick an eval set before
    downloading the big camera_image files."""
    rows = []
    for seg, f in segment_files(Path(stats_dir).parent, Path(stats_dir).name).items():
        s = pd.read_parquet(f)
        ped = 0
        for types, counts in zip(
            s[f"{STATS}.camera_object_counts.types"],
            s[f"{STATS}.camera_object_counts.counts"],
            strict=True,
        ):
            if types is None or counts is None:  # frame with nothing labelled
                continue
            d = dict(zip(types, counts, strict=True))
            ped += d.get(TYPE_PEDESTRIAN, 0)
        rows.append(
            {
                "segment": seg,
                "timeofday": TIME_OF_DAY.get(_mode(s[f"{STATS}.time_of_day"]), "undefined"),
                "weather": _mode(s[f"{STATS}.weather"]),
                "location": _mode(s[f"{STATS}.location"]).removeprefix("location_"),
                "pedestrian_boxes": ped,
            }
        )
    return pd.DataFrame(rows)


def pick_segments(summary: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Every night and rain segment that has pedestrians, then fill up with a
    random draw of the rest.

    This over-represents night and rain on purpose: those are the checks that
    come back INCONCLUSIVE when the set is small. It doesn't bias any single
    check, because every check is computed inside its own condition.
    """
    has_peds = summary["pedestrian_boxes"] > 0
    rare = summary[
        has_peds & ((summary["timeofday"] != "daytime") | (summary["weather"] != "sunny"))
    ]
    rest = summary.drop(rare.index)
    k = max(0, n - len(rare))
    fill = rest.sample(n=min(k, len(rest)), random_state=seed)
    return pd.concat([rare, fill]).head(n).sort_values("segment").reset_index(drop=True)
