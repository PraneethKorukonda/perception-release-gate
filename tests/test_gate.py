from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from releasegate.bootstrap import Interval
from releasegate.gate import Check, ModelRun, decide, load_config, run_gate, verdict


def test_verdict_rules():
    assert verdict(Interval(0.0, -0.005, 0.005), 0.01, 50, 15) == "PASS"
    assert verdict(Interval(-0.02, -0.03, -0.01), 0.01, 50, 15) == "FAIL"
    # crossed the margin by chance, but the interval still reaches zero
    assert verdict(Interval(-0.012, -0.03, 0.006), 0.01, 50, 15) == "INCONCLUSIVE"
    # best guess fine, but the interval reaches past the margin
    assert verdict(Interval(-0.005, -0.02, 0.01), 0.01, 50, 15) == "INCONCLUSIVE"
    # too few clips, whatever the numbers
    assert verdict(Interval(0.0, -0.001, 0.001), 0.01, 5, 15) == "INCONCLUSIVE"


def test_decision_only_looks_at_blocking_checks():
    c = pd.DataFrame({"blocking": [True, False], "verdict": ["PASS", "FAIL"]})
    assert decide(c) == "SHIP"
    c.loc[0, "verdict"] = "INCONCLUSIVE"
    assert decide(c) == "NEEDS MORE DATA"
    c.loc[0, "verdict"] = "FAIL"
    assert decide(c) == "HOLD"


def test_default_config_loads():
    cfg = load_config()
    assert len(cfg.checks) >= 8
    assert {c.metric for c in cfg.checks} >= {"frame_recall", "detected_within_1s", "late_in_path"}
    with pytest.raises(ValueError):
        Check(name="x", metric="accuracy", margin=0.1)


def test_same_model_against_itself_ships(small_sim):
    d, runs = small_sim
    cfg = replace(load_config(), n_boot=300, n_boot_map=50, min_clips=5)
    r = run_gate(d.images, d.gt, runs["current"], runs["current"], cfg)
    assert r.decision == "SHIP"
    assert np.allclose(r.checks["diff"], 0)


def test_regressed_candidate_is_held(small_sim):
    d, runs = small_sim
    cfg = replace(load_config(), n_boot=300, n_boot_map=50, min_clips=5)
    r = run_gate(d.images, d.gt, runs["current"], runs["candidate"], cfg)
    assert r.decision == "HOLD"
    failed = set(r.checks.loc[r.checks["verdict"] == "FAIL", "check"])
    assert any("night" in name for name in failed)
    # and the headline number still looks better, which is the point
    m = r.checks.set_index("metric").loc["map50"]
    assert m["candidate"] > m["current"]


def test_gate_runs_without_predictions(small_sim):
    d, _ = small_sim
    cfg = replace(load_config(), n_boot=100, min_clips=5)
    r = run_gate(
        d.images,
        d.gt,
        ModelRun("a", d.latent["current"]),
        ModelRun("b", d.latent["candidate_good"]),
        cfg,
    )
    assert "map50" not in set(r.checks["metric"])


def test_bad_filter_is_reported_clearly(small_sim):
    d, runs = small_sim
    cfg = replace(
        load_config(),
        n_boot=50,
        checks=[Check(name="typo", metric="frame_recall", where="nite", margin=0.01)],
    )
    with pytest.raises(ValueError, match="can't apply filter"):
        run_gate(d.images, d.gt, runs["current"], runs["candidate"], cfg)
