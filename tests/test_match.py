import numpy as np
import pandas as pd
import pytest

from releasegate.boxes import iou_matrix, size_bucket
from releasegate.match import match

GT_COLS = ["gt_id", "image_id", "cls", "ignore", "x1", "y1", "x2", "y2"]
P_COLS = ["pred_id", "image_id", "cls", "score", "x1", "y1", "x2", "y2"]


def gt_frame(rows):
    return pd.DataFrame(rows, columns=GT_COLS)


def pred_frame(rows):
    return pd.DataFrame(rows, columns=P_COLS)


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 5, 5]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0, 0.25])


def test_iou_empty_and_degenerate():
    assert iou_matrix(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)
    assert iou_matrix(np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]]))[0, 0] == 0.0


def test_size_buckets_use_coco_edges():
    assert list(size_bucket([31 * 31, 32 * 32, 96 * 96 - 1, 96 * 96])) == [
        "small",
        "medium",
        "medium",
        "large",
    ]


def test_duplicate_prediction_is_a_false_positive():
    gt = gt_frame([(0, "f", "car", False, 0, 0, 10, 10)])
    p = pred_frame([(0, "f", "car", 0.6, 0, 0, 10, 10), (1, "f", "car", 0.9, 0, 0, 10, 10)])
    gm, pm = match(gt, p)
    pm = pm.set_index("pred_id")
    assert pm.loc[1, "tp"] and not pm.loc[0, "tp"]  # higher score wins
    assert gm.loc[0, "pred_id"] == 1 and gm.loc[0, "score"] == 0.9


def test_best_iou_wins_among_free_boxes():
    gt = gt_frame([(0, "f", "car", False, 0, 0, 10, 10), (1, "f", "car", False, 2, 0, 12, 10)])
    p = pred_frame([(0, "f", "car", 0.9, 2, 0, 12, 10)])
    gm, pm = match(gt, p)
    assert pm.loc[0, "gt_id"] == 1
    assert np.isnan(gm.set_index("gt_id").loc[0, "score"])


def test_classes_never_cross():
    gt = gt_frame([(0, "f", "car", False, 0, 0, 10, 10)])
    p = pred_frame([(0, "f", "truck", 0.9, 0, 0, 10, 10)])
    gm, pm = match(gt, p)
    assert not pm.loc[0, "tp"] and gm.loc[0, "pred_id"] == -1


def test_iou_threshold_is_inclusive():
    gt = gt_frame([(0, "f", "car", False, 0, 0, 10, 10)])
    p = pred_frame([(0, "f", "car", 0.9, 0, 0, 10, 5)])  # IoU exactly 0.5
    _, pm = match(gt, p, iou_thr=0.5)
    assert pm.loc[0, "tp"]


def test_rider_region_absorbs_pedestrian_predictions():
    gt = gt_frame(
        [
            (0, "f", "pedestrian", False, 0, 0, 10, 20),
            (1, "f", "pedestrian", True, 100, 0, 110, 20),  # a rider
        ]
    )
    p = pred_frame(
        [
            (0, "f", "pedestrian", 0.9, 0, 0, 10, 20),
            (1, "f", "pedestrian", 0.8, 100, 0, 110, 20),
            (2, "f", "pedestrian", 0.7, 101, 0, 111, 20),  # ignore regions take any number
            (3, "f", "pedestrian", 0.6, 300, 0, 310, 20),
        ]
    )
    gm, pm = match(gt, p)
    pm = pm.set_index("pred_id")
    assert list(pm["tp"]) == [True, False, False, False]
    assert list(pm["ignored"]) == [False, True, True, False]
    assert len(gm) == 1  # the rider itself isn't scored


def test_regular_box_preferred_over_ignore_region():
    gt = gt_frame(
        [
            (0, "f", "pedestrian", True, 0, 0, 10, 20),
            (1, "f", "pedestrian", False, 1, 0, 11, 20),
        ]
    )
    p = pred_frame([(0, "f", "pedestrian", 0.9, 0, 0, 10, 20)])
    _, pm = match(gt, p)
    assert pm.loc[0, "tp"] and pm.loc[0, "gt_id"] == 1


def test_threshold_after_matching_equals_matching_after_threshold():
    """The property the whole package relies on: match once, filter by score later."""
    rng = np.random.default_rng(0)
    rows_g, rows_p = [], []
    for f in range(30):
        for _ in range(5):
            x = rng.uniform(0, 500)
            rows_g.append((len(rows_g), f"f{f}", "car", False, x, 0, x + 40, 40))
            for _ in range(rng.integers(0, 3)):
                j = rng.normal(0, 8, 4)
                rows_p.append(
                    (
                        len(rows_p),
                        f"f{f}",
                        "car",
                        rng.random(),
                        x + j[0],
                        j[1],
                        x + 40 + j[2],
                        40 + j[3],
                    )
                )
    gt, p = gt_frame(rows_g), pred_frame(rows_p)
    gm_all, _ = match(gt, p)
    for thr in (0.3, 0.6):
        gm_thr, _ = match(gt, p[p["score"] >= thr])
        a = np.nan_to_num(gm_all["score"].to_numpy(), nan=-1) >= thr
        b = gm_thr["pred_id"].to_numpy() >= 0
        assert (a == b).all()


@pytest.mark.parametrize("n_pred", [0, 1])
def test_empty_inputs(n_pred):
    gt = gt_frame([(0, "f", "car", False, 0, 0, 10, 10)])
    p = pred_frame([(0, "g", "car", 0.5, 0, 0, 10, 10)][:n_pred])
    gm, pm = match(gt, p)
    assert len(gm) == 1 and gm.loc[0, "pred_id"] == -1
    assert len(pm) == n_pred
