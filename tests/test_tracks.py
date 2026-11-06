import numpy as np
import pandas as pd
import pytest

from releasegate.safety import SafetyParams, distance_m, stopping_distance
from releasegate.tracks import track_table


def make_obj(tracks: dict, video="v", night=False, speed=10.0, first_frame=0, last_frame=20):
    """tracks: id -> (frames, det flags, box height px, centre x)."""
    rows = []
    for tid, (frames, dets, h, cx) in tracks.items():
        for f, d in zip(frames, dets, strict=True):
            hh = h(f) if callable(h) else h
            rows.append(
                {
                    "gt_id": len(rows),
                    "image_id": f"{video}/{f}",
                    "video": video,
                    "frame_index": f,
                    "track_id": tid,
                    "cls": "pedestrian",
                    "det": bool(d),
                    "timeofday": "night" if night else "daytime",
                    "weather": "clear",
                    "scene": "city street",
                    "area": 30 * hh,
                    "speed_mps": speed,
                    "distance_m": 1000 * 1.7 / hh,
                    "lateral_m": (cx - 640) * (1.7 / hh),
                }
            )
    # a background object spanning the whole clip fixes the clip's first frame
    for f in range(first_frame, last_frame + 1):
        rows.append(
            {
                "gt_id": len(rows),
                "image_id": f"{video}/{f}",
                "video": video,
                "frame_index": f,
                "track_id": "bg",
                "cls": "car",
                "det": True,
                "timeofday": "daytime",
                "weather": "clear",
                "scene": "city street",
                "area": 5000.0,
                "speed_mps": speed,
                "distance_m": 20.0,
                "lateral_m": 10.0,
            }
        )
    return pd.DataFrame(rows)


def test_time_to_first_detection_and_window():
    obj = make_obj(
        {
            "fast": (range(5, 15), [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 40, 640),
            "slow": (range(5, 15), [0] * 7 + [1, 1, 1], 40, 640),
            "never": (range(5, 15), [0] * 10, 40, 640),
            "short": (range(5, 8), [0, 0, 0], 40, 640),  # visible < 1 s: not judged
            "censored": (range(0, 10), [0, 0, 1] + [1] * 7, 40, 640),
        }
    )
    t = track_table(obj).set_index("track_id")
    assert t.loc["fast", "ttfd_s"] == pytest.approx(0.2)
    assert t.loc["slow", "ttfd_s"] == pytest.approx(1.4)
    assert t.loc["fast", "detected_within_1s"] == 1.0
    assert t.loc["slow", "detected_within_1s"] == 0.0
    assert t.loc["never", "detected_within_1s"] == 0.0
    assert np.isnan(t.loc["short", "detected_within_1s"])
    assert t.loc["censored", "left_censored"]
    assert np.isnan(t.loc["censored", "detected_within_1s"])


def test_dropouts_only_count_when_reacquired():
    obj = make_obj(
        {
            "flicker": (range(3, 13), [1, 0, 1, 1, 0, 0, 1, 1, 1, 0], 40, 640),
        }
    )
    t = track_table(obj).set_index("track_id")
    # two losses that were picked up again; the final miss isn't a dropout
    assert t.loc["flicker", "dropouts"] == 2
    assert t.loc["flicker", "n_det_frames"] == 6


def test_late_detection_uses_distance_at_first_detection():
    p = SafetyParams()
    need = stopping_distance(10.0, p)  # ~21.7 m at 10 m/s
    # approaching pedestrian: height grows 4 px per frame from 40 px (42.5 m)
    h = lambda f: 40 + 4 * (f - 5)  # noqa: E731
    d_at = lambda f: 1000 * 1.7 / h(f)  # noqa: E731
    early, late = 6, 15
    assert d_at(early) > need > d_at(late)
    obj = make_obj(
        {
            "early": (range(5, 20), [0] * (early - 5) + [1] * (20 - early), h, 650),
            "late": (range(5, 20), [0] * (late - 5) + [1] * (20 - late), h, 650),
            "never": (range(5, 20), [0] * 15, h, 650),
            "sidewalk": (range(5, 20), [0] * 15, h, 1100),  # never in path, not judged
        },
        last_frame=25,
    )
    t = track_table(obj, params=p).set_index("track_id")
    assert t.loc["early", "late_in_path"] == 0.0
    assert t.loc["late", "late_in_path"] == 1.0
    assert t.loc["never", "late_in_path"] == 1.0
    assert np.isnan(t.loc["sidewalk", "late_in_path"])


def test_missed_pedestrian_the_car_never_approached_is_not_late():
    # stays ~42 m away the whole time (40 px), well beyond the ~21.7 m stopping distance
    obj = make_obj({"far": (range(5, 20), [0] * 15, 40, 650)}, last_frame=25)
    t = track_table(obj).set_index("track_id")
    assert not t.loc["far", "reached_stop_range"]
    assert t.loc["far", "late_in_path"] == 0.0


def test_pedestrian_already_too_close_is_not_judged():
    obj = make_obj({"close": (range(5, 15), [1] * 10, 120, 640)})  # ~14 m when first seen
    t = track_table(obj).set_index("track_id")
    assert np.isnan(t.loc["close", "late_in_path"])


def test_distance_inverts_pinhole_and_skips_truncated():
    d = distance_m(["pedestrian", "pedestrian"], [85.0, 85.0], [False, True])
    assert d[0] == pytest.approx(20.0)
    assert np.isnan(d[1])
