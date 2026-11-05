import numpy as np
import pandas as pd

from releasegate import mining, scenarios


def test_set_versioning_and_diff(tmp_path):
    path = tmp_path / "night.json"
    a = scenarios.save_set(scenarios.make_set("night", ["v/1", "v/2"], "rule", "night"), path)
    assert a["version"] == 1
    # saving the same frames again doesn't bump the version
    same = scenarios.save_set(scenarios.make_set("night", ["v/2", "v/1"], "rule", "night"), path)
    assert same["version"] == 1
    b = scenarios.save_set(scenarios.make_set("night", ["v/2", "v/3"], "rule", "night"), path)
    assert b["version"] == 2 and b["history"][0]["fingerprint"] == a["fingerprint"]
    assert scenarios.diff_sets(a, b) == {"added": 1, "removed": 1, "kept": 1}


def test_expand_window_stays_in_clip():
    images = pd.DataFrame(
        {
            "image_id": [f"{v}/{k}" for v in ("a", "b") for k in range(10)],
            "video": ["a"] * 10 + ["b"] * 10,
            "frame_index": list(range(10)) * 2,
        }
    )
    out = set(scenarios.expand_window(["a/5"], images, 2))
    assert out == {"a/3", "a/4", "a/5", "a/6", "a/7"}


def test_rule_set_and_gate_columns(small_sim):
    d, runs = small_sim
    from releasegate.gate import prepare

    obj, _, _ = prepare(d.images, d.gt, runs["current"], 0.25, 5.0)
    s = scenarios.rule_set(obj, "night_peds", "cls == 'pedestrian' and night")
    assert s["n_frames"] > 0
    obj2, trk2, _ = prepare(d.images, d.gt, runs["current"], 0.25, 5.0, sets=[s])
    assert obj2["set_night_peds"].any() and trk2["set_night_peds"].any()


def test_search_and_precision_with_fake_embeddings():
    rng = np.random.default_rng(0)
    night = rng.normal(size=8)
    vecs = rng.normal(size=(100, 8)) * 0.3
    vecs[:20] += night  # the first 20 frames "look like" night
    vecs = mining.normalize(vecs)
    ids = np.array([f"f{i}" for i in range(100)])
    hits = mining.search(ids, vecs, mining.normalize(night[None])[0], k=20)
    p = mining.precision_at_k(hits["image_id"], ids[:20], ks=(10, 20))
    assert p["p@10"] == 1.0 and p["p@20"] >= 0.9
