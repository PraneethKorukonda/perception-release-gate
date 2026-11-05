"""Box helpers. Everything here uses (x1, y1, x2, y2) in pixels.

IoU is computed on continuous coordinates (no +1 on width/height), which is
what pycocotools does. The old VOC convention adds 1 and gives slightly
different numbers on small boxes, so it matters if you compare against other
tools.
"""

from __future__ import annotations

import numpy as np

# COCO area buckets, in pixels^2 at original resolution.
SMALL_MAX = 32**2
MEDIUM_MAX = 96**2


def area(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return w * h


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between every box in `a` (n x 4) and `b` (m x 4)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    union = area(a)[:, None] + area(b)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def size_bucket(box_area: np.ndarray) -> np.ndarray:
    box_area = np.asarray(box_area, dtype=float)
    return np.where(
        box_area < SMALL_MAX, "small", np.where(box_area < MEDIUM_MAX, "medium", "large")
    )


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    return boxes
