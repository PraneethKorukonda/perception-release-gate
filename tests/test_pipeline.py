"""Inference adapter, simulator sanity, SQL, Spark and the CLI end to end."""

import shutil

import numpy as np
import pandas as pd
import pytest

from releasegate import db, infer
from releasegate.cli import main
from releasegate.metrics import detected


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = np.array(xyxy), np.array(conf), np.array(cls)

    def __len__(self):
        return len(self.conf)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYolo:
    names = {0: "person", 2: "car", 9: "traffic light", 11: "stop sign"}

    def predict(self, paths, **kw):
        self.classes_seen = kw.get("classes")
        return [
            FakeResult(
                FakeBoxes(
                    [[0, 0, 10, 20], [5, 5, 50, 30], [1, 1, 2, 2]], [0.9, 0.4, 0.8], [0, 2, 9]
                )
            )
            for _ in paths
        ]


def test_inference_adapter_maps_coco_classes():
    out = infer.predict(FakeYolo(), ["v/1.jpg", "v/2.jpg"], "/nowhere", "fake", log_every=0)
    assert list(out.columns) == infer.PRED_COLUMNS
    assert set(out["cls"]) == {"pedestrian", "car"}  # traffic light dropped
    assert out["pred_id"].is_unique and len(out) == 4


def test_inference_only_asks_for_scored_classes():
    m = FakeYolo()
    infer.predict(m, ["v/1.jpg"], "/nowhere", "fake", log_every=0)
    assert sorted(m.classes_seen) == [0, 2]  # person, car; traffic light and stop sign left out


def test_matcher_recovers_simulated_detections(small_sim):
    """The simulator knows which boxes each model really found. The matcher
    should agree on nearly all of them (a few objects overlap)."""
    d, runs = small_sim
    for name, run in runs.items():
        agree = (detected(run.gt_match, 0.25) == detected(d.latent[name], 0.25)).mean()
        assert agree > 0.995, name


def test_simulated_geometry_is_consistent(small_sim):
    d, _ = small_sim
    ped = d.gt[(d.gt["cls"] == "pedestrian") & ~d.gt["truncated"]]
    h = ped["y2"] - ped["y1"]
    # 12-70 m spawn range with f=1000 and 1.7 m height -> about 24-140 px when first seen
    assert h.min() >= 8 and h.median() < 150
    assert d.images["speed_mps"].notna().mean() > 0.7


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    main(["demo", "--out", str(out), "--clips", "30"])
    return out


def test_demo_writes_reports(demo_dir):
    rep = demo_dir / "reports" / "current_vs_candidate"
    for f in (
        "summary.md",
        "details.md",
        "checks.csv",
        "objects.parquet",
        "tracks.parquet",
        "figures/gate_checks.png",
        "figures/time_to_detection.png",
    ):
        assert (rep / f).exists(), f
    text = (rep / "summary.md").read_text()
    assert "## Decision:" in text


def test_every_saved_query_runs(demo_dir):
    con = db.connect(demo_dir / "reports" / "current_vs_candidate")
    for name in db.list_queries():
        df = db.run(con, name)
        assert isinstance(df, pd.DataFrame), name


def test_cli_match_and_gate_on_saved_runs(demo_dir, tmp_path):
    runs = tmp_path / "runs"
    shutil.copytree(demo_dir / "runs", runs)
    main(["match", "--data", str(demo_dir / "data"), "--run", str(runs / "candidate_good")])
    out = tmp_path / "rep"
    main(
        [
            "gate",
            "--data",
            str(demo_dir / "data"),
            "--current",
            str(runs / "current"),
            "--candidate",
            str(runs / "candidate_good"),
            "--out",
            str(out),
            "--n-boot",
            "200",
        ]
    )
    assert (out / "summary.md").exists()


@pytest.mark.spark
def test_spark_job_matches_duckdb(demo_dir):
    pytest.importorskip("pyspark")
    if shutil.which("java") is None:
        pytest.skip("no JVM")
    from releasegate import spark_job

    rep = demo_dir / "reports" / "current_vs_candidate"
    spark = spark_job.session("test")
    try:
        _, slices = spark_job.build(spark, str(rep / "objects.parquet"))
        s = slices.toPandas()
    finally:
        spark.stop()
    con = db.connect(rep)
    ref = con.execute("""
        SELECT c.cls, c.timeofday, c.weather, c."size",
               COUNT(*) AS n_boxes, AVG(n.det::INT) - AVG(c.det::INT) AS diff
        FROM objects c JOIN objects n USING (gt_id)
        WHERE c.model = 'current' AND n.model = 'candidate'
        GROUP BY ALL
    """).df()
    m = s.merge(ref, on=["cls", "timeofday", "weather", "size"], suffixes=("_spark", "_duck"))
    assert len(m) == len(ref) == len(s)
    assert (m["n_boxes_spark"] == m["n_boxes_duck"]).all()
    np.testing.assert_allclose(m["diff_spark"], m["diff_duck"], atol=1e-12)
