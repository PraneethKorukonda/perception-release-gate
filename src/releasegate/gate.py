"""The release gate: SHIP, HOLD, or NEEDS MORE DATA.

Each check compares the candidate model with the current one on the same
clips and asks a non-inferiority question: "is the candidate worse than the
current model by more than we're willing to accept?" The margin is part of
the check, written down before anyone looks at the results.

For each check I compute the improvement (candidate minus current, flipped
for metrics where lower is better) and a clip-bootstrap interval for it.

    PASS          the lower end of the interval is above -margin.
                  We're confident it isn't a meaningful regression.
    FAIL          the point estimate is below -margin, and the whole
                  interval is below zero. Our best guess is that it got worse
                  by more than allowed, and we're confident it got worse.
    INCONCLUSIVE  neither. Usually the scenario is too rare in the eval set
                  to tell, which is a dataset problem to fix, not a pass.

The "confident it got worse" part of FAIL was added after the simulation
study (studies.py). The first version failed a check whenever the point
estimate crossed the margin. Each check does that by chance now and then,
and with eight blocking checks the gate held an *unchanged* model 25% of
the time.

Decision: any blocking FAIL -> HOLD. Otherwise any blocking INCONCLUSIVE ->
NEEDS MORE DATA. Otherwise SHIP.

All checks reuse the same bootstrap replicates (the same resampled clips), so
the intervals for different checks are consistent with each other.

A design choice worth defending: a regression in one scenario blocks the
release even if the overall numbers improve. Averages hide exactly the
regressions that matter (a few hundred far-away pedestrians at night
barely move overall mAP).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from releasegate.bootstrap import Interval, interval, per_unit_sum, ratio_draws, unit_weights
from releasegate.metrics import ap_from_arrays
from releasegate.scenarios import apply_sets, tag_objects, tag_tracks
from releasegate.tracks import object_table, track_table

HIGHER_IS_BETTER = {
    "frame_recall": True,
    "detected_within_1s": True,
    "map50": True,
    "dropouts_per_100": False,
    "late_in_path": False,
    "fp_per_frame": False,
}
NEEDS_PREDICTIONS = {"map50", "fp_per_frame"}


@dataclass
class Check:
    name: str
    metric: str
    margin: float
    where: str | None = None
    blocking: bool = True
    classes: list[str] | None = None  # map50 only

    def __post_init__(self):
        if self.metric not in HIGHER_IS_BETTER:
            raise ValueError(f"unknown metric {self.metric!r} in check {self.name!r}")


@dataclass
class GateConfig:
    checks: list[Check]
    thresholds: dict = field(default_factory=lambda: {"current": 0.25, "candidate": 0.25})
    level: float = 0.90
    n_boot: int = 2000
    n_boot_map: int = 300
    seed: int = 0
    fps: float = 5.0
    min_clips: int = 20
    iou: float = 0.5


def load_config(path: str | Path | None = None) -> GateConfig:
    """A file path, or the name of a packaged config ("default", "waymo")."""
    name = "default" if path is None else str(path)
    packaged = resources.files("releasegate").joinpath(f"configs/{name}_gate.toml")
    if packaged.is_file():
        text = packaged.read_text()
    else:
        text = Path(path).read_text()
    raw = tomllib.loads(text)
    s = raw.get("settings", {})
    return GateConfig(
        checks=[Check(**c) for c in raw["check"]],
        thresholds=raw.get("thresholds", {"current": 0.25, "candidate": 0.25}),
        **{
            k: s[k]
            for k in ("level", "n_boot", "n_boot_map", "seed", "fps", "min_clips", "iou")
            if k in s
        },
    )


@dataclass
class ModelRun:
    """Matched outputs for one model. pred_match may be None (simulation studies)."""

    name: str
    gt_match: pd.DataFrame
    pred_match: pd.DataFrame | None = None


@dataclass
class GateResult:
    decision: str
    checks: pd.DataFrame
    objects: dict  # role -> per-box table
    tracks: dict  # role -> per-object table
    clips: pd.Index
    config: GateConfig


def _query(df: pd.DataFrame, where: str | None) -> pd.DataFrame:
    if not where:
        return df
    try:
        return df.query(where)
    except Exception as e:  # pandas raises several types here
        raise ValueError(f"can't apply filter {where!r}: {e}") from e


def _per_clip(clips, df, num, den):
    return per_unit_sum(clips, df["video"], num), per_unit_sum(clips, df["video"], den)


def metric_sums(metric, where, obj, trk, preds, images, clips):
    """Per-clip numerator and denominator for one model."""
    if metric == "frame_recall":
        o = _query(obj, where)
        return _per_clip(clips, o, o["det"], np.ones(len(o)))
    if metric == "detected_within_1s":
        t = _query(trk, where).dropna(subset=["detected_within_1s"])
        return _per_clip(clips, t, t["detected_within_1s"], np.ones(len(t)))
    if metric == "dropouts_per_100":
        t = _query(trk, where)
        return _per_clip(clips, t, 100 * t["dropouts"], t["n_det_frames"])
    if metric == "late_in_path":
        t = _query(trk, where).dropna(subset=["late_in_path"])
        return _per_clip(clips, t, t["late_in_path"], np.ones(len(t)))
    if metric == "fp_per_frame":
        p = _query(preds, where)
        num = per_unit_sum(clips, p["video"], (~p["tp"]).to_numpy(float))
        den = per_unit_sum(clips, images["video"], np.ones(len(images)))
        return num, den
    raise ValueError(metric)


class _APBoot:
    """Class AP under clip weights, with the score sort done once."""

    def __init__(self, gm, pm, clips, classes):
        self.parts = []
        for c in classes:
            p = pm[(pm["cls"] == c) & ~pm["ignored"]].sort_values(
                "score", ascending=False, kind="mergesort"
            )
            n_gt = per_unit_sum(
                clips, gm.loc[gm["cls"] == c, "video"], np.ones((gm["cls"] == c).sum())
            )
            self.parts.append(
                (p["tp"].to_numpy(), p["score"].to_numpy(), clips.get_indexer(p["video"]), n_gt)
            )

    def __call__(self, w=None):
        aps = []
        for tp, score, pos, n_gt in self.parts:
            if w is None:
                aps.append(ap_from_arrays(tp, score, n_gt.sum()))
            else:
                aps.append(ap_from_arrays(tp, score, w @ n_gt, weights=w[pos]))
        return np.nanmean(aps)


def verdict(improvement: Interval, margin: float, n_clips: int, min_clips: int) -> str:
    if n_clips < min_clips or not np.isfinite(improvement.lo):
        return "INCONCLUSIVE"
    if improvement.lo >= -margin:
        return "PASS"
    # Worse than allowed AND confidently worse at all. The second condition is
    # what stops chance from blocking releases: see the docstring at the top.
    if improvement.estimate < -margin and improvement.hi < 0:
        return "FAIL"
    return "INCONCLUSIVE"


def decide(checks: pd.DataFrame) -> str:
    b = checks[checks["blocking"]]
    if (b["verdict"] == "FAIL").any():
        return "HOLD"
    if (b["verdict"] == "INCONCLUSIVE").any():
        return "NEEDS MORE DATA"
    return "SHIP"


def prepare(images, gt, run: ModelRun, threshold: float, fps: float, sets=()):
    obj = object_table(gt, run.gt_match, images, threshold)
    trk = tag_tracks(track_table(obj, fps=fps))
    obj = tag_objects(obj, trk)
    if sets:
        obj = apply_sets(obj, list(sets))
        for s in sets:
            col = f"set_{s['name']}"
            trk[col] = trk["track_id"].map(obj.groupby("track_id")[col].any()).fillna(False)
    preds = None
    if run.pred_match is not None:
        p = run.pred_match[~run.pred_match["ignored"] & (run.pred_match["score"] >= threshold)]
        preds = p.merge(images, on="image_id", how="left")
    return obj, trk, preds


def run_gate(
    images, gt, current: ModelRun, candidate: ModelRun, config: GateConfig, sets=()
) -> GateResult:
    clips = pd.Index(np.sort(images["video"].unique()))
    roles = {"current": current, "candidate": candidate}
    prepared = {
        r: prepare(images, gt, m, config.thresholds[r], config.fps, sets) for r, m in roles.items()
    }

    rng = np.random.default_rng(config.seed)
    W = unit_weights(len(clips), config.n_boot, rng)
    W_map = W[: config.n_boot_map]

    rows = []
    for chk in config.checks:
        if chk.metric in NEEDS_PREDICTIONS and (
            current.pred_match is None or candidate.pred_match is None
        ):
            continue
        sign = 1.0 if HIGHER_IS_BETTER[chk.metric] else -1.0

        if chk.metric == "map50":
            classes = chk.classes or sorted(gt.loc[~gt["ignore"], "cls"].unique())
            boots = {
                r: _APBoot(
                    m.gt_match.merge(images[["image_id", "video"]], on="image_id"),
                    m.pred_match.merge(images[["image_id", "video"]], on="image_id"),
                    clips,
                    classes,
                )
                for r, m in roles.items()
            }
            vals = {r: boots[r]() for r in roles}
            draws = np.array([boots["candidate"](w) - boots["current"](w) for w in W_map])
            n_clips = len(clips)
        else:
            sums = {}
            for r in roles:
                obj, trk, preds = prepared[r]
                sums[r] = metric_sums(chk.metric, chk.where, obj, trk, preds, images, clips)
            vals = {
                r: sums[r][0].sum() / sums[r][1].sum() if sums[r][1].sum() > 0 else np.nan
                for r in roles
            }
            draws = ratio_draws(*sums["candidate"], W) - ratio_draws(*sums["current"], W)
            # clips that contribute anything to this check
            n_clips = int(((sums["current"][1] > 0) | (sums["candidate"][1] > 0)).sum())

        diff = interval(vals["candidate"] - vals["current"], draws, config.level)
        improvement = interval(sign * diff.estimate, sign * draws, config.level)
        v = verdict(improvement, chk.margin, n_clips, config.min_clips)
        rows.append(
            {
                "check": chk.name,
                "metric": chk.metric,
                "where": chk.where or "",
                "blocking": chk.blocking,
                "current": vals["current"],
                "candidate": vals["candidate"],
                "diff": diff.estimate,
                "diff_lo": diff.lo,
                "diff_hi": diff.hi,
                "margin": chk.margin,
                "higher_is_better": HIGHER_IS_BETTER[chk.metric],
                "n_clips": n_clips,
                "verdict": v,
            }
        )

    checks = pd.DataFrame(rows)
    return GateResult(
        decision=decide(checks),
        checks=checks,
        objects={r: prepared[r][0] for r in roles},
        tracks={r: prepared[r][1] for r in roles},
        clips=clips,
        config=config,
    )
