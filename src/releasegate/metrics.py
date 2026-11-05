"""Precision, recall, F1 and AP from matched tables.

AP here is COCO's 101-point interpolated AP. With the IoU threshold fixed at
0.5 it's what most people call mAP@0.5 (it is not the same number as the old
VOC 11-point AP, and it's a bit different from all-point AP too, so I keep the
method explicit).

The weighted variants exist for the bootstrap: resampling images with
replacement is the same as giving each image an integer weight, so I never
have to copy rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def detected(gt_match: pd.DataFrame, score_thr: float) -> np.ndarray:
    """Per ground-truth box: did a matched prediction survive the threshold?"""
    s = gt_match["score"].to_numpy(float)
    return np.nan_to_num(s, nan=-np.inf) >= score_thr


def ap_from_arrays(tp, score, n_gt, weights=None, method: str = "coco101") -> float:
    """AP for one class.

    tp, score: one entry per (non-ignored) prediction.
    n_gt: number of ground-truth boxes (or their summed weight).
    weights: optional per-prediction weight.
    """
    tp = np.asarray(tp, dtype=bool)
    score = np.asarray(score, dtype=float)
    if n_gt <= 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    w = np.ones(len(tp)) if weights is None else np.asarray(weights, dtype=float)

    order = np.argsort(-score, kind="mergesort")
    tp, w = tp[order], w[order]
    ctp = np.cumsum(w * tp)
    cfp = np.cumsum(w * ~tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, np.spacing(1))

    # Precision envelope: best precision at this recall or any higher recall.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]

    if method == "coco101":
        idx = np.searchsorted(recall, RECALL_POINTS, side="left")
        q = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
        return float(q.mean())
    if method == "allpoint":
        r = np.concatenate([[0.0], recall])
        return float(np.sum((r[1:] - r[:-1]) * envelope))
    raise ValueError(f"unknown AP method {method!r}")


def pr_curve(pred_match: pd.DataFrame, n_gt: int, cls: str) -> pd.DataFrame:
    p = pred_match[(pred_match["cls"] == cls) & ~pred_match["ignored"]]
    p = p.sort_values("score", ascending=False, kind="mergesort")
    ctp = p["tp"].cumsum().to_numpy()
    cfp = (~p["tp"]).cumsum().to_numpy()
    return pd.DataFrame(
        {
            "score": p["score"].to_numpy(),
            "recall": ctp / max(n_gt, 1),
            "precision": ctp / np.maximum(ctp + cfp, 1),
        }
    )


def class_table(
    gt_match: pd.DataFrame,
    pred_match: pd.DataFrame,
    n_images: int,
    score_thr: float = 0.25,
) -> pd.DataFrame:
    """One row per class plus an 'all' row (micro P/R, macro AP = mAP)."""
    det = detected(gt_match, score_thr)
    rows = []
    for cls in sorted(set(gt_match["cls"]) | set(pred_match["cls"])):
        g = gt_match["cls"].to_numpy() == cls
        p = pred_match[(pred_match["cls"] == cls) & ~pred_match["ignored"]]
        n_gt = int(g.sum())
        kept = p["score"].to_numpy() >= score_thr
        tp = int(p["tp"].to_numpy()[kept].sum())
        fp = int(kept.sum() - tp)
        rows.append(
            {
                "cls": cls,
                "n_gt": n_gt,
                "tp": tp,
                "fp": fp,
                "fn": int(n_gt - det[g].sum()),
                "ap50": ap_from_arrays(p["tp"], p["score"], n_gt),
            }
        )
    t = pd.DataFrame(rows)
    t = t[t["n_gt"] > 0].reset_index(drop=True)

    total = {
        "cls": "all",
        "n_gt": t["n_gt"].sum(),
        "tp": t["tp"].sum(),
        "fp": t["fp"].sum(),
        "fn": t["fn"].sum(),
        "ap50": t["ap50"].mean(),
    }
    t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)
    t["precision"] = t["tp"] / (t["tp"] + t["fp"]).replace(0, np.nan)
    t["recall"] = t["tp"] / t["n_gt"]
    t["f1"] = 2 * t["precision"] * t["recall"] / (t["precision"] + t["recall"])
    t["fp_per_image"] = t["fp"] / n_images
    return t
