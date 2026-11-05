"""How often does the gate get it wrong? Checked on simulated clips.

A gate has two ways to fail:
  * it ships a model that really is worse in a scenario that matters, or
  * it blocks a model that's actually fine, which wastes a release cycle and
    teaches people to ignore the gate.

On real data you never find out which happened. In the simulator I wrote the
candidate's behaviour myself, so I can compute the true value of every
check (from one very large simulated eval set) and then rerun the gate on
many ordinary-sized eval sets to count its mistakes.

This study grades the gate's decision logic. It feeds the gate the
simulator's own detection outcomes instead of running the matcher, which
keeps it fast. The matcher is graded separately: it agrees with those
outcomes on 99.9% of boxes (tests/test_sim.py) and with pycocotools on AP
(tests/test_metrics.py).

A second, smaller study shows why the bootstrap resamples clips. With two
identical models, a box-level bootstrap flags a "significant" difference
far more often than its nominal rate. The clip-level one stays close to it.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from releasegate.bootstrap import interval, per_unit_sum, unit_weights
from releasegate.gate import GateConfig, ModelRun, load_config, run_gate
from releasegate.sim import ModelSpec, SimConfig, simulate

SCENARIOS = {
    "No change": ModelSpec(),
    "Better overall, worse on far pedestrians at night": ModelSpec(
        offset=0.5, small_pedestrian_night=-1.6, frame_sd=1.1
    ),
    "Plain improvement": ModelSpec(offset=0.3, pedestrian=0.15),
    "Subtle night regression": ModelSpec(offset=0.2, small_pedestrian_night=-0.6),
    "Noisier frame to frame": ModelSpec(frame_sd=1.5),
}


def _gate_on_latent(cfg: SimConfig, gate_cfg: GateConfig, seed: int):
    d = simulate(cfg, seed=seed)
    return run_gate(
        d.images,
        d.gt,
        ModelRun("current", d.latent["current"]),
        ModelRun("candidate", d.latent["candidate"]),
        gate_cfg,
    )


def _one(args):
    cfg, gate_cfg, seed = args
    r = _gate_on_latent(cfg, gate_cfg, seed)
    return r.decision, r.checks[["check", "verdict", "diff"]]


def true_values(
    spec: ModelSpec, gate_cfg: GateConfig, n_videos: int = 2000, chunk: int = 400
) -> pd.DataFrame:
    """Each check's value on a large eval set, built in chunks to keep memory flat.
    Averaging chunk estimates instead of pooling sums makes no practical
    difference at this size."""
    quick = replace(gate_cfg, n_boot=20, n_boot_map=0, min_clips=0)
    diffs = []
    for i in range(max(1, n_videos // chunk)):
        cfg = SimConfig(n_videos=chunk, models={"current": ModelSpec(), "candidate": spec})
        r = _gate_on_latent(cfg, quick, seed=10_000 + i)
        diffs.append(r.checks.set_index("check")["diff"])
        t = r.checks[["check", "blocking", "margin", "higher_is_better"]].copy()
        del r
    t["true_diff"] = pd.concat(diffs, axis=1).mean(axis=1).reindex(t["check"]).to_numpy()
    sign = np.where(t["higher_is_better"], 1.0, -1.0)
    t["truly_worse"] = sign * t["true_diff"] < -t["margin"]
    return t


def gate_error_rates(
    reps: int = 40,
    n_videos: int = 200,
    n_boot: int = 1000,
    workers: int = 2,
    scenarios: list[str] | None = None,
    truth_videos: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_cfg = replace(load_config(), n_boot=n_boot)
    summary, per_check = [], []
    for name in scenarios or list(SCENARIOS):
        spec = SCENARIOS[name]
        truth = true_values(spec, gate_cfg, n_videos=truth_videos)
        should = "HOLD" if (truth["truly_worse"] & truth["blocking"]).any() else "SHIP"
        cfg = SimConfig(n_videos=n_videos, models={"current": ModelSpec(), "candidate": spec})
        jobs = [(cfg, gate_cfg, 1000 + i) for i in range(reps)]
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(_one, jobs))
        decisions = pd.Series([o[0] for o in out])
        summary.append(
            {
                "scenario": name,
                "clips": n_videos,
                "correct decision": should,
                "SHIP": (decisions == "SHIP").mean(),
                "HOLD": (decisions == "HOLD").mean(),
                "NEEDS MORE DATA": (decisions == "NEEDS MORE DATA").mean(),
            }
        )
        checks = pd.concat([o[1].assign(rep=i) for i, o in enumerate(out)])
        rates = (
            checks.groupby("check", sort=False)["verdict"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .reindex(columns=["PASS", "FAIL", "INCONCLUSIVE"], fill_value=0.0)
        )
        rates = truth.set_index("check")[["true_diff", "margin", "truly_worse"]].join(rates)
        per_check.append(rates.reset_index().assign(scenario=name))
    return pd.DataFrame(summary), pd.concat(per_check, ignore_index=True)


def _null_one(args):
    seed, n_videos, n_boot, level = args
    cfg = SimConfig(n_videos=n_videos, models={"current": ModelSpec(), "candidate": ModelSpec()})
    d = simulate(cfg, seed=seed)
    ped = d.gt["cls"].to_numpy() == "pedestrian"
    a = np.nan_to_num(d.latent["current"]["score"].to_numpy()[ped], nan=-1) >= 0.25
    b = np.nan_to_num(d.latent["candidate"]["score"].to_numpy()[ped], nan=-1) >= 0.25
    video = d.gt["video"].to_numpy()[ped]
    rng = np.random.default_rng(seed)

    clips = pd.Index(np.unique(video))
    diff = per_unit_sum(clips, video, b.astype(float) - a)
    n = per_unit_sum(clips, video, np.ones(len(a)))
    W = unit_weights(len(clips), n_boot, rng)
    clip_ci = interval(diff.sum() / n.sum(), (W @ diff) / (W @ n), level)

    k = len(a)
    Wb = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot)
    box_ci = interval((b.astype(float) - a).mean(), (Wb @ (b.astype(float) - a)) / k, level)
    return {
        "clip": not (clip_ci.lo <= 0 <= clip_ci.hi),
        "box": not (box_ci.lo <= 0 <= box_ci.hi),
        "clip_width": clip_ci.hi - clip_ci.lo,
        "box_width": box_ci.hi - box_ci.lo,
    }


def resampling_unit_study(reps=200, n_videos=200, n_boot=1000, level=0.90, workers=2):
    with ProcessPoolExecutor(workers) as ex:
        out = pd.DataFrame(ex.map(_null_one, [(i, n_videos, n_boot, level) for i in range(reps)]))
    return pd.DataFrame(
        {
            "resampling unit": ["clip", "box"],
            "nominal false alarm rate": [1 - level, 1 - level],
            "observed false alarm rate": [out["clip"].mean(), out["box"].mean()],
            "median interval width (pts)": [
                100 * out["clip_width"].median(),
                100 * out["box_width"].median(),
            ],
        }
    )
