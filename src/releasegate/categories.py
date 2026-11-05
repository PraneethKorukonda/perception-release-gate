"""Mapping between BDD100K labels, COCO detector outputs and the classes I score.

This file looks boring but it moves the numbers a lot.

BDD100K labels a person on a bike as `rider` and the bike as `bicycle`. A
COCO-trained detector has no rider class: it calls that person a `person`.
If you map person -> pedestrian and stop there, every correctly detected rider
counts as a pedestrian false positive.

So rider boxes become ignore regions for the pedestrian class: a pedestrian
prediction that lands on a rider is neither a hit nor a false positive.
Riders themselves aren't scored, since the detector can't output that class.
BDD's `crowd` boxes (a group too dense to box one by one) are handled the
same way. It's the COCO crowd-region idea.

Dropped: train (a handful of boxes, nothing to learn from) and traffic
lights/signs (not in the tracking labels).
"""

from __future__ import annotations

EVAL_CLASSES = ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"]

BDD_TO_EVAL = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motor": "motorcycle",
}

# BDD classes kept only as ignore regions, keyed by the class they shield.
BDD_IGNORE = {"rider": "pedestrian"}

# COCO names as ultralytics reports them.
COCO_TO_EVAL = {
    "person": "pedestrian",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
}

# Rough real-world heights in metres, used to turn box height into distance.
# Only pedestrian distance feeds a gate check; the rest are for plots.
OBJECT_HEIGHT_M = {
    "pedestrian": 1.7,
    "car": 1.5,
    "truck": 3.0,
    "bus": 3.2,
    "bicycle": 1.0,
    "motorcycle": 1.1,
}


def map_bdd(category: str) -> tuple[str | None, str | None]:
    """Return (eval_class, ignore_for). At most one is set."""
    category = category.strip().lower()
    if category in BDD_TO_EVAL:
        return BDD_TO_EVAL[category], None
    if category in BDD_IGNORE:
        return None, BDD_IGNORE[category]
    return None, None


def map_coco(name: str) -> str | None:
    return COCO_TO_EVAL.get(name.strip().lower())
