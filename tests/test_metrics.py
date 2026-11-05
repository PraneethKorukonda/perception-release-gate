import numpy as np
import pandas as pd
import pytest

from releasegate.boxes import xyxy_to_xywh
from releasegate.metrics import ap_from_arrays, class_table


def test_ap_perfect_and_empty():
    assert ap_from_arrays([True, True], [0.9, 0.8], 2) == pytest.approx(1.0)
    assert ap_from_arrays([], [], 3) == 0.0
    assert np.isnan(ap_from_arrays([True], [0.5], 0))


def test_ap_by_hand():
    # TP, FP, TP over 2 ground-truth boxes.
    # recall:    .5  .5   1
    # precision:  1  .5  2/3   -> envelope 1, 2/3, 2/3
    # 101-point: r in [0, .5] -> 1 (51 points), r in (.5, 1] -> 2/3 (50 points)
    ap = ap_from_arrays([True, False, True], [0.9, 0.8, 0.7], 2)
    assert ap == pytest.approx((51 * 1 + 50 * 2 / 3) / 101)
    allpoint = ap_from_arrays([True, False, True], [0.9, 0.8, 0.7], 2, method="allpoint")
    assert allpoint == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_weights_equal_duplication():
    tp = np.array([True, False, True, True, False])
    score = np.array([0.9, 0.85, 0.7, 0.4, 0.3])
    w = np.array([2, 1, 3, 1, 2])
    weighted = ap_from_arrays(tp, score, 7, weights=w, method="allpoint")
    dup = ap_from_arrays(np.repeat(tp, w), np.repeat(score, w), 7, method="allpoint")
    assert weighted == pytest.approx(dup)


def test_class_table_counts(small_sim):
    d, runs = small_sim
    r = runs["current"]
    t = class_table(r.gt_match, r.pred_match, len(d.images), 0.25).set_index("cls")
    for cls, row in t.drop("all").iterrows():
        assert row["tp"] + row["fn"] == row["n_gt"], cls
    assert t.loc["all", "n_gt"] == len(r.gt_match)


def test_ap_matches_pycocotools(small_sim):
    """Same AP as the reference implementation, class by class."""
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    d, runs = small_sim
    run = runs["current"]
    gt = d.gt[~d.gt["ignore"]]
    # keep it quick: a subset of frames
    frames = sorted(gt["image_id"].unique())[:800]
    gt = gt[gt["image_id"].isin(frames)]
    pm = run.pred_match[run.pred_match["image_id"].isin(frames)]
    preds = d.preds["current"].set_index("pred_id").loc[pm["pred_id"]].reset_index()

    img_id = {f: i + 1 for i, f in enumerate(frames)}
    classes = sorted(gt["cls"].unique())
    cat_id = {c: i + 1 for i, c in enumerate(classes)}
    gt_xywh = xyxy_to_xywh(gt[["x1", "y1", "x2", "y2"]].to_numpy())
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [{"id": i} for i in img_id.values()],
        "categories": [{"id": i, "name": c} for c, i in cat_id.items()],
        "annotations": [
            {
                "id": k + 1,
                "image_id": img_id[f],
                "category_id": cat_id[c],
                "bbox": list(b),
                "area": float(b[2] * b[3]),
                "iscrowd": 0,
            }
            for k, (f, c, b) in enumerate(zip(gt["image_id"], gt["cls"], gt_xywh, strict=True))
        ],
    }
    coco_gt.createIndex()
    p_xywh = xyxy_to_xywh(preds[["x1", "y1", "x2", "y2"]].to_numpy())
    dets = [
        {
            "image_id": img_id[f],
            "category_id": cat_id[c],
            "bbox": list(map(float, b)),
            "score": float(s),
        }
        for f, c, b, s in zip(preds["image_id"], preds["cls"], p_xywh, preds["score"], strict=True)
    ]
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.iouThrs = np.array([0.5])
    ev.params.areaRng = [[0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.params.maxDets = [10_000]
    ev.evaluate()
    ev.accumulate()
    ref = ev.eval["precision"][0, :, :, 0, 0].mean(axis=0)  # per class

    gm = run.gt_match[run.gt_match["image_id"].isin(frames)]
    for c, k in cat_id.items():
        p = pm[(pm["cls"] == c) & ~pm["ignored"]]
        ours = ap_from_arrays(p["tp"], p["score"], int((gm["cls"] == c).sum()))
        assert ours == pytest.approx(ref[k - 1], abs=1e-9), c


def test_class_table_has_all_row(small_sim):
    d, runs = small_sim
    t = class_table(runs["current"].gt_match, runs["current"].pred_match, len(d.images))
    assert t["cls"].iloc[-1] == "all"
    assert isinstance(t, pd.DataFrame)
