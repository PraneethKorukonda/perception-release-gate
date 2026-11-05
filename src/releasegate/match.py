"""Greedy prediction-to-ground-truth matching, same rules as pycocotools.

Within each (image, class):
  1. Sort predictions by score, highest first.
  2. Each prediction takes the unmatched regular box with the highest IoU,
     as long as IoU >= threshold.
  3. If no regular box is available it may land on an ignore box (riders, for
     pedestrians). Then it's flagged `ignored` and doesn't count either way.
     Ignore boxes can absorb any number of predictions.
  4. Anything left is a false positive.

One property I rely on later: because matching runs in score order, raising the
score threshold never changes which box a surviving prediction matched. So I
match once at a very low threshold and get P/R at any threshold by filtering.
That's also why inference should run with a low conf (0.01), not the 0.25
default: AP needs the low-score tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from releasegate.boxes import iou_matrix

BOX = ["x1", "y1", "x2", "y2"]


def match(gt: pd.DataFrame, preds: pd.DataFrame, iou_thr: float = 0.5):
    """Returns (gt_match, pred_match).

    gt_match has one row per non-ignored ground-truth box:
        gt_id, image_id, cls, pred_id (-1 if missed), score (nan if missed), iou
    pred_match has one row per prediction:
        pred_id, image_id, cls, score, tp, ignored, gt_id, iou
    """
    thr = min(iou_thr, 1 - 1e-10)

    gt = gt.reset_index(drop=True)
    preds = preds.sort_values(
        ["image_id", "cls", "score"], ascending=[True, True, False], kind="mergesort"
    ).reset_index(drop=True)

    g_box = gt[BOX].to_numpy(float)
    g_ign = gt["ignore"].to_numpy(bool)
    p_box = preds[BOX].to_numpy(float)

    p_gt = np.full(len(preds), -1, dtype=np.int64)  # row index into gt
    p_iou = np.zeros(len(preds))
    p_ignored = np.zeros(len(preds), dtype=bool)
    g_pred = np.full(len(gt), -1, dtype=np.int64)  # row index into preds
    g_iou = np.zeros(len(gt))

    gt_groups = gt.groupby(["image_id", "cls"], sort=False).indices
    pred_groups = preds.groupby(["image_id", "cls"], sort=False).indices

    for key, p_idx in pred_groups.items():
        g_idx = gt_groups.get(key)
        if g_idx is None:
            continue
        # regular boxes first, then ignore boxes (stable, like COCO)
        g_idx = g_idx[np.argsort(g_ign[g_idx], kind="mergesort")]
        ious = iou_matrix(p_box[p_idx], g_box[g_idx])
        is_ign = g_ign[g_idx]
        taken = np.zeros(len(g_idx), dtype=bool)

        for k, p in enumerate(p_idx):  # already in score order
            row = ious[k]
            free = (~taken) & (~is_ign) & (row >= thr)
            if free.any():
                j = int(np.argmax(np.where(free, row, -1.0)))
                taken[j] = True
                p_gt[p] = g_idx[j]
                p_iou[p] = row[j]
                g_pred[g_idx[j]] = p
                g_iou[g_idx[j]] = row[j]
                continue
            ign = is_ign & (row >= thr)
            if ign.any():
                j = int(np.argmax(np.where(ign, row, -1.0)))
                p_ignored[p] = True
                p_gt[p] = g_idx[j]
                p_iou[p] = row[j]

    pred_match = pd.DataFrame(
        {
            "pred_id": preds["pred_id"].to_numpy(),
            "image_id": preds["image_id"].to_numpy(),
            "cls": preds["cls"].to_numpy(),
            "score": preds["score"].to_numpy(float),
            "tp": (p_gt >= 0) & ~p_ignored,
            "ignored": p_ignored,
            "gt_id": np.where(p_gt >= 0, gt["gt_id"].to_numpy()[np.maximum(p_gt, 0)], -1),
            "iou": p_iou,
        }
    )

    hit = g_pred >= 0
    pred_ids = preds["pred_id"].to_numpy()
    scores = preds["score"].to_numpy(float)
    gt_match = pd.DataFrame(
        {
            "gt_id": gt["gt_id"].to_numpy(),
            "image_id": gt["image_id"].to_numpy(),
            "cls": gt["cls"].to_numpy(),
            "pred_id": np.where(hit, pred_ids[np.maximum(g_pred, 0)] if len(preds) else -1, -1),
            "score": np.where(hit, scores[np.maximum(g_pred, 0)] if len(preds) else np.nan, np.nan),
            "iou": g_iou,
        }
    )[~g_ign].reset_index(drop=True)

    return gt_match, pred_match
