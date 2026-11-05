import numpy as np
import pytest
from conftest import FIXTURES

from releasegate import bdd, gps


@pytest.fixture
def parsed():
    return bdd.load_tracking_dir(FIXTURES / "track_val")


def test_frames_and_clip_summary(parsed):
    images, gt, videos = parsed
    assert len(images) == 3  # the frame with labels: null still counts
    assert images["image_id"].iloc[0] == "a1b2c3d4-00000001/a1b2c3d4-00000001-0000001.jpg"
    assert list(images["frame_index"]) == [0, 1, 2]
    assert videos.loc[0, "n_frames"] == 3 and videos.loc[0, "timeofday"] == "night"


def test_class_mapping_and_ignore_regions(parsed):
    _, gt, _ = parsed
    by_label = gt.set_index("label")
    # rider -> ignore region for pedestrians
    assert by_label.loc["rider", "cls"] == "pedestrian" and by_label.loc["rider", "ignore"]
    # bicycle kept and scored
    assert by_label.loc["bicycle", "cls"] == "bicycle" and not by_label.loc["bicycle", "ignore"]
    assert "train" not in set(gt["label"])
    crowd = gt[gt["track_id"].str.endswith(":4")]
    assert crowd["ignore"].all() and (crowd["cls"] == "pedestrian").all()


def test_track_ids_are_unique_per_clip(parsed):
    _, gt, _ = parsed
    ped = gt[gt["track_id"] == "a1b2c3d4-00000001:1"]
    assert list(ped["frame_index"]) == [0, 1]
    assert gt["gt_id"].is_unique


def test_gps_interpolation(parsed):
    images, _, _ = parsed
    g = gps.frame_gps(images, FIXTURES / "info").set_index("frame_index")
    # 5 fps -> frames at +0, +200, +400 ms; speed 10 -> 12 over the first second
    np.testing.assert_allclose(g["speed_mps"], [10.0, 10.4, 10.8])
    # the -1 "no fix" speed is ignored rather than interpolated towards
    assert g.loc[1, "lat"] == pytest.approx(40.702)


def test_missing_info_gives_nan(parsed, tmp_path):
    images, _, _ = parsed
    g = gps.frame_gps(images, tmp_path)
    assert g["speed_mps"].isna().all()
