"""Confidence intervals by resampling whole video clips.

Frames from the same 20-second clip are nearly copies of each other: same
road, same light, same camera, often the same pedestrians. Treating 40,000
frames as 40,000 independent samples gives intervals that are far too
narrow. The independent unit is the clip, so that's what gets resampled.
(tests/test_bootstrap.py shows how badly the frame-level version undercovers.)

A bootstrap replicate is a vector of counts, one per clip (multinomial with
n draws). Any metric that is a ratio of per-clip sums is then two matrix
products, so thousands of replicates are cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Interval:
    estimate: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def unit_weights(n_units: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=n_boot)


def interval(estimate: float, draws: np.ndarray, level: float = 0.9) -> Interval:
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0 or not np.isfinite(estimate):
        return Interval(float(estimate), np.nan, np.nan)
    a = (1 - level) / 2
    lo, hi = np.quantile(draws, [a, 1 - a])
    return Interval(float(estimate), float(lo), float(hi))


def per_unit_sum(index: pd.Index, unit_ids, values) -> np.ndarray:
    """Sum `values` within each unit (clip), in the order of `index`."""
    pos = index.get_indexer(np.asarray(unit_ids))
    if (pos < 0).any():
        raise KeyError("some rows belong to clips that aren't in the index")
    return np.bincount(pos, weights=np.asarray(values, dtype=float), minlength=len(index))


def ratio_draws(num: np.ndarray, den: np.ndarray, W: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (W @ num) / (W @ den)


def ratio_interval(num, den, W, level: float = 0.9) -> Interval:
    num, den = np.asarray(num, float), np.asarray(den, float)
    est = num.sum() / den.sum() if den.sum() > 0 else np.nan
    return interval(est, ratio_draws(num, den, W), level)


def frame_level_interval(outcome, n_boot, rng, level: float = 0.9) -> Interval:
    """Resample individual boxes as if independent. Kept only to show it's wrong."""
    outcome = np.asarray(outcome, float)
    n = len(outcome)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    return interval(outcome.mean(), (counts @ outcome) / n, level)
