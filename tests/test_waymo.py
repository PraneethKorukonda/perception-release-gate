"""Waymo ingest on small synthetic files with the same schema as perception v2.0.1.

The real files can't be committed (license), so these are built column for
column from the real schema. I checked the reader against two real validation
segments by hand: frame counts, speeds and class counts matched.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from releasegate import waymo
from releasegate.categories import map_coco
from releasegate.gate import load_config

SEG = "1234567890_1000_000_1020_000"
T0 = 1_522_688_014_970_187


def _write(root, component, name, table: pa.Table):
    d = root / component
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, d / name)


def transform(x, y):
    m = np.eye(4)
    m[0, 3], m[1, 3] = x, y
    return m.ravel().tolist()


@pytest.fixture
def root(tmp_path):
    n = 20  # frames, 10 Hz
    ts = [T0 + 100_000 * k for k in range(n)]
    # browser-style names for some components, gsutil-style for others
    browser = lambda c: f"validation-{c}-{SEG}.parquet"  # noqa: E731

    _write(
        tmp_path,
        "stats",
        browser("stats"),
        pa.table(
            {
                "key.segment_context_name": [SEG] * n,
                "key.frame_timestamp_micros": ts,
                "[StatsComponent].time_of_day": ["Night"] * n,
                "[StatsComponent].location": ["location_sf"] * n,
                "[StatsComponent].weather": ["rain"] * n,
                # real files leave these null on frames with nothing labelled
                "[StatsComponent].camera_object_counts.types": [None] + [[1, 2]] * (n - 1),
                "[StatsComponent].camera_object_counts.counts": [None] + [[3, 2]] * (n - 1),
            }
        ),
    )
    # ego drives +x at 8 m/s
    fsl = pa.list_(pa.float64(), 16)
    _write(
        tmp_path,
        "vehicle_pose",
        f"{SEG}.parquet",
        pa.table(
            {
                "key.segment_context_name": [SEG] * n,
                "key.frame_timestamp_micros": ts,
                "[VehiclePoseComponent].world_from_vehicle.transform": pa.array(
                    [transform(0.8 * k, 0.0) for k in range(n)], type=fsl
                ),
            }
        ),
    )
    _write(
        tmp_path,
        "camera_calibration",
        browser("camera_calibration"),
        pa.table(
            {
                "key.segment_context_name": [SEG] * 2,
                "key.camera_name": pa.array([1, 2], pa.int8()),
                "[CameraCalibrationComponent].intrinsic.f_u": [2000.0, 2000.0],
                "[CameraCalibrationComponent].intrinsic.f_v": [2000.0, 2000.0],
                "[CameraCalibrationComponent].intrinsic.c_u": [960.0, 960.0],
                "[CameraCalibrationComponent].intrinsic.c_v": [640.0, 640.0],
                "[CameraCalibrationComponent].width": pa.array([1920, 1920], pa.int32()),
                "[CameraCalibrationComponent].height": pa.array([1280, 1280], pa.int32()),
            }
        ),
    )
    rows = []
    for k, t in enumerate(ts):
        # pedestrian 1.7 m tall at 40 m -> 85 px; sign; cyclist; side-camera box
        rows.append((t, 1, "ped-1", 960.0, 600.0, 30.0, 85.0, 2))
        rows.append((t, 1, "sign-1", 300.0, 300.0, 20.0, 20.0, 3))
        if k < 5:
            rows.append((t, 1, "cyc-1", 1500.0, 650.0, 60.0, 100.0, 4))
        rows.append((t, 2, "ped-side", 500.0, 600.0, 30.0, 85.0, 2))
    r = list(zip(*rows, strict=True))
    _write(
        tmp_path,
        "camera_box",
        browser("camera_box"),
        pa.table(
            {
                "key.segment_context_name": [SEG] * len(rows),
                "key.frame_timestamp_micros": list(r[0]),
                "key.camera_name": pa.array(r[1], pa.int8()),
                "key.camera_object_id": list(r[2]),
                "[CameraBoxComponent].box.center.x": list(r[3]),
                "[CameraBoxComponent].box.center.y": list(r[4]),
                "[CameraBoxComponent].box.size.x": list(r[5]),
                "[CameraBoxComponent].box.size.y": list(r[6]),
                "[CameraBoxComponent].type": pa.array(r[7], pa.int8()),
                "[CameraBoxComponent].difficulty_level.detection": pa.array(
                    [2] * len(rows), pa.int8()
                ),
            }
        ),
    )
    _write(
        tmp_path,
        "camera_image",
        f"{SEG}.parquet",
        pa.table(
            {
                "key.segment_context_name": [SEG] * (2 * n),
                "key.frame_timestamp_micros": ts + ts,
                "key.camera_name": pa.array([1] * n + [2] * n, pa.int8()),
                "[CameraImageComponent].image": [b"\xff\xd8front" + bytes([k]) for k in range(n)]
                + [b"\xff\xd8side"] * n,
            }
        ),
    )
    return tmp_path


def test_reads_both_file_naming_styles(root):
    assert waymo.available_segments(root) == [SEG]


def test_frames_conditions_and_speed(root):
    images, gt, videos = waymo.load_waymo(root)
    assert len(images) == 20 and list(images["frame_index"]) == list(range(20))
    first = images.iloc[0]
    assert (first["timeofday"], first["weather"], first["location"]) == ("night", "rainy", "sf")
    np.testing.assert_allclose(images["speed_mps"], 8.0, rtol=1e-4)  # float timestamps
    assert first["focal_px"] == 2000.0
    assert videos.loc[0, "location"] == "sf"


def test_boxes_front_camera_classes_and_ignore(root):
    _, gt, _ = waymo.load_waymo(root)
    assert set(gt["label"]) == {"pedestrian", "cyclist"}  # sign dropped, side camera dropped
    cyc = gt[gt["label"] == "cyclist"]
    assert cyc["ignore"].all() and (cyc["cls"] == "pedestrian").all()
    ped = gt[gt["label"] == "pedestrian"]
    assert ped["track_id"].nunique() == 1 and len(ped) == 20
    r = ped.iloc[0]
    assert (r.x1, r.y1, r.x2, r.y2) == pytest.approx((945, 557.5, 975, 642.5))
    assert not gt["occluded"].any()


def test_distance_uses_real_focal_length(root):
    from releasegate.tracks import object_table

    images, gt, _ = waymo.load_waymo(root)
    gm = gt.loc[~gt["ignore"], ["gt_id", "image_id", "cls"]].assign(score=0.9)
    obj = object_table(gt, gm, images, 0.5)
    assert obj["distance_m"].iloc[0] == pytest.approx(2000 * 1.7 / 85)  # 40 m
    assert obj["lateral_m"].iloc[0] == pytest.approx(0.0)


def test_extract_front_images_only(root, tmp_path):
    images, _, _ = waymo.load_waymo(root)
    out = tmp_path / "imgs"
    assert waymo.extract_images(root, images, out) == 20
    files = sorted((out / SEG).glob("*.jpg"))
    assert len(files) == 20 and files[0].read_bytes().startswith(b"\xff\xd8front")
    assert waymo.extract_images(root, images, out) == 0  # rerun skips what exists


def test_summarize_stats_handles_empty_frames(root):
    s = waymo.summarize_stats(root / "stats")
    assert s.loc[0, "pedestrian_boxes"] == 2 * 19
    assert (s.loc[0, "timeofday"], s.loc[0, "location"]) == ("night", "sf")


def test_pick_segments_keeps_rare_conditions():
    s = pd.DataFrame(
        {
            "segment": [f"s{i}" for i in range(10)],
            "timeofday": ["night", "dawn/dusk"] + ["daytime"] * 8,
            "weather": ["sunny"] * 9 + ["rain"],
            "location": "phx",
            "pedestrian_boxes": [5, 5, 0, 1, 1, 1, 1, 1, 1, 3],
        }
    )
    picked = waymo.pick_segments(s, n=5, seed=1)
    assert len(picked) == 5
    assert {"s0", "s1", "s9"} <= set(picked["segment"])


def test_waymo_class_map_and_config():
    assert map_coco("truck", "waymo") == "vehicle"
    assert map_coco("bicycle", "waymo") is None
    assert map_coco("bicycle", "bdd") == "bicycle"
    cfg = load_config("waymo")
    assert cfg.fps == 10
    assert all("car" not in (c.where or "") for c in cfg.checks)
