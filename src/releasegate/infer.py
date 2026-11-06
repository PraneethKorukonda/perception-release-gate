"""Run an ultralytics YOLO model over the frames and save its boxes.

Output is a model-independent table, so any detector can be evaluated as
long as it writes the same columns:

    pred_id, model, image_id, cls, score, x1, y1, x2, y2

Two settings that matter:

conf=0.01   keep low-score boxes. The gate applies its own deployment
            threshold, and AP needs the low-score tail.
imgsz=1280  BDD frames are 1280x720. The ultralytics default of 640 halves
            the resolution, and a pedestrian 40 m away is ~40 px tall at
            full size, so it gets noticeably worse on exactly the objects
            the gate cares about. Whatever you choose, use the same value for
            both models or you're comparing input sizes, not models.

On an M1/M2 Mac use device="mps". yolo11n at 1280 does roughly 15-25
frames/s there, so the 200-clip val split (~40k frames) takes 30-45 minutes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from releasegate.categories import map_coco

PRED_COLUMNS = ["pred_id", "model", "image_id", "cls", "score", "x1", "y1", "x2", "y2"]


def results_to_rows(result, image_id: str, names: dict, classes: str = "bdd") -> list[tuple]:
    """Turn one ultralytics Results object into rows, keeping only mapped classes."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = np.asarray(boxes.xyxy.cpu() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy, float)
    conf = np.asarray(boxes.conf.cpu() if hasattr(boxes.conf, "cpu") else boxes.conf, float)
    cls = np.asarray(boxes.cls.cpu() if hasattr(boxes.cls, "cpu") else boxes.cls).astype(int)
    rows = []
    for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls, strict=True):
        mapped = map_coco(names[int(c)], classes)
        if mapped is not None:
            rows.append((image_id, mapped, float(s), float(x1), float(y1), float(x2), float(y2)))
    return rows


def predict(
    model,
    image_ids: list[str],
    image_root: str | Path,
    model_name: str,
    imgsz: int = 1280,
    conf: float = 0.01,
    device: str | None = None,
    batch: int = 16,
    log_every: int = 200,
    classes: str = "bdd",
) -> pd.DataFrame:
    """`model` is anything with an ultralytics-style predict() and .names."""
    root = Path(image_root)
    # Only keep the COCO classes we score. Filtering before NMS matters: at conf
    # 0.01 the other 74 classes add thousands of candidates per image and NMS
    # hits its time limit, which silently drops detections.
    keep = [i for i, n in model.names.items() if map_coco(n, classes)]
    rows: list[tuple] = []
    t0 = time.time()
    for start in range(0, len(image_ids), batch):
        ids = image_ids[start : start + batch]
        paths = [str(root / i) for i in ids]
        results = model.predict(
            paths, imgsz=imgsz, conf=conf, device=device, verbose=False, max_det=300, classes=keep
        )
        for image_id, r in zip(ids, results, strict=True):
            rows += results_to_rows(r, image_id, model.names, classes)
        done = start + len(ids)
        if log_every and (done // log_every) != ((done - len(ids)) // log_every):
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(image_ids)} frames, {rate:.1f} frames/s", flush=True)

    out = pd.DataFrame(rows, columns=PRED_COLUMNS[2:])
    out.insert(0, "model", model_name)
    out.insert(0, "pred_id", np.arange(len(out), dtype=np.int64))
    return out


def run_yolo(weights: str, images: pd.DataFrame, image_root, model_name=None, **kw):
    from ultralytics import YOLO  # heavy import, only when actually running a model

    model = YOLO(weights)
    name = model_name or Path(weights).stem
    return predict(model, images["image_id"].tolist(), image_root, name, **kw)
