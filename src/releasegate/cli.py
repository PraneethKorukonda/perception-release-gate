"""Command line entry point: `releasegate <command>`.

Typical run on real data:

    releasegate ingest  --labels bdd100k/labels/box_track_20/val --info bdd100k/info/100k/val --out data/val
    releasegate infer   --data data/val --images bdd100k/images/track/val --weights yolo11s.pt --out runs/yolo11s
    releasegate infer   --data data/val --images bdd100k/images/track/val --weights yolo11m.pt --out runs/yolo11m
    releasegate match   --data data/val --run runs/yolo11s
    releasegate match   --data data/val --run runs/yolo11m
    releasegate gate    --data data/val --current runs/yolo11s --candidate runs/yolo11m --out reports/s_vs_m
    releasegate sql     --report reports/s_vs_m late_pedestrians

No data yet? `releasegate demo` runs all of it on simulated clips.
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


def _load_run(path, name=None):
    from releasegate.gate import ModelRun

    p = Path(path)
    return ModelRun(
        name or p.name,
        pd.read_parquet(p / "gt_match.parquet"),
        pd.read_parquet(p / "pred_match.parquet"),
    )


def cmd_ingest(a):
    from releasegate import bdd, gps

    images, gt, videos = bdd.load_tracking_dir(a.labels, limit=a.limit, seed=a.seed)
    if a.info:
        images = gps.frame_gps(images, a.info)
        videos = videos.merge(gps.video_gps(images), on="video", how="left")
    if a.estimate_lighting:
        from releasegate import lighting

        est = lighting.estimate_timeofday(images, a.estimate_lighting)
        images = lighting.apply_estimate(images, est)
        videos = videos.drop(columns="timeofday").merge(
            bdd.summarize_videos(images)[["video", "timeofday"]], on="video"
        )
    bdd.save_dataset(a.out, images, gt, videos)
    print(f"{len(videos)} clips, {len(images):,} frames, {len(gt):,} boxes -> {a.out}")
    print(images["timeofday"].value_counts().to_string())


def cmd_infer(a):
    from releasegate import bdd, infer

    images, _, _ = bdd.load_dataset(a.data)
    if a.limit:
        keep = images["video"].drop_duplicates().head(a.limit)
        images = images[images["video"].isin(keep)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds = infer.run_yolo(
        a.weights,
        images,
        a.images,
        model_name=a.name or out.name,
        imgsz=a.imgsz,
        conf=a.conf,
        device=a.device,
        batch=a.batch,
    )
    preds.to_parquet(out / "preds.parquet", index=False)
    print(f"{len(preds):,} boxes -> {out / 'preds.parquet'}")


def cmd_match(a):
    from releasegate import bdd
    from releasegate.match import match

    _, gt, _ = bdd.load_dataset(a.data)
    run = Path(a.run)
    preds = pd.read_parquet(run / "preds.parquet")
    gm, pm = match(gt, preds, iou_thr=a.iou)
    gm.to_parquet(run / "gt_match.parquet", index=False)
    pm.to_parquet(run / "pred_match.parquet", index=False)
    print(f"matched {len(pm):,} predictions against {len(gm):,} boxes")


def cmd_gate(a):
    from releasegate import bdd, report, scenarios
    from releasegate.gate import load_config, run_gate

    images, gt, videos = bdd.load_dataset(a.data)
    cfg = load_config(a.config)
    if a.n_boot:
        cfg = replace(cfg, n_boot=a.n_boot)
    sets = [scenarios.load_set(p) for pattern in a.set or [] for p in glob.glob(pattern)]
    runs = {"current": _load_run(a.current), "candidate": _load_run(a.candidate)}
    result = run_gate(images, gt, runs["current"], runs["candidate"], cfg, sets=sets)
    paths = report.write_report(result, gt, images, videos, runs, a.out)
    print(f"\n  DECISION: {result.decision}\n")
    cols = ["check", "current", "candidate", "diff", "diff_lo", "diff_hi", "verdict"]
    print(result.checks[cols].round(4).to_string(index=False))
    print(f"\nsummary: {paths['summary']}\ndetails: {paths['details']}")


def cmd_sql(a):
    from releasegate import db

    if not a.name:
        for k, v in db.list_queries().items():
            print(f"{k:22s} {v}")
        return
    con = db.connect(a.report)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(db.run(con, a.name).to_string(index=False))


def cmd_scenario(a):
    from releasegate import bdd, scenarios
    from releasegate.scenarios import tag_objects
    from releasegate.tracks import object_table, track_table

    images, gt, _ = bdd.load_dataset(a.data)
    gm = gt.loc[~gt["ignore"], ["gt_id", "image_id", "cls"]].assign(score=float("nan"))
    obj = object_table(gt, gm, images, 0.5)
    obj = tag_objects(obj, track_table(obj))
    s = scenarios.rule_set(obj, a.name, a.where, notes=a.notes or "")
    s = scenarios.save_set(s, a.out)
    print(
        f"{s['name']} v{s['version']}: {s['n_frames']:,} frames in {s['n_clips']} clips -> {a.out}"
    )


def cmd_mine(a):
    from releasegate import bdd, mining, scenarios

    images, gt, _ = bdd.load_dataset(a.data)
    if a.action == "embed":
        frames = mining.frame_paths(images, a.images, stride=a.stride)
        emb = mining.ClipEmbedder(a.model).images(frames["path"].tolist())
        mining.save_embeddings(a.emb, frames["image_id"], emb)
        print(f"embedded {len(frames):,} frames -> {a.emb}")
        return
    ids, vecs = mining.load_embeddings(a.emb)
    q = mining.ClipEmbedder(a.model).text([a.query])[0]
    hits = mining.search(ids, vecs, q, k=a.k)
    if a.action == "check":
        pos = images.query(a.label)["image_id"]
        print(f"query: {a.query!r}   labels: {a.label!r}")
        print(f"base rate among embedded frames: {pd.Series(ids).isin(set(pos)).mean():.3f}")
        print(mining.precision_at_k(hits["image_id"], pos))
        return
    frames = scenarios.expand_window(hits["image_id"], images, a.stride)
    s = scenarios.make_set(
        a.name,
        frames,
        "clip",
        a.query,
        notes=f"top {a.k} of {len(ids)} embedded frames, +-{a.stride} frames",
    )
    s = scenarios.save_set(s, a.out)
    hits.to_csv(Path(a.out).with_suffix(".hits.csv"), index=False)
    print(
        f"{s['name']} v{s['version']}: {s['n_frames']:,} frames in {s['n_clips']} clips -> {a.out}"
    )
    print("Look at the top hits before using this set to gate anything.")


def cmd_demo(a):
    from releasegate import bdd, report, sim
    from releasegate.gate import ModelRun, load_config, run_gate
    from releasegate.match import match

    out = Path(a.out)
    d = sim.simulate(sim.SimConfig(n_videos=a.clips), seed=a.seed)
    bdd.save_dataset(out / "data", d.images, d.gt, d.videos)
    runs = {}
    for name, preds in d.preds.items():
        rd = out / "runs" / name
        rd.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(rd / "preds.parquet", index=False)
        gm, pm = match(d.gt, preds)
        gm.to_parquet(rd / "gt_match.parquet", index=False)
        pm.to_parquet(rd / "pred_match.parquet", index=False)
        runs[name] = ModelRun(name, gm, pm)
    print(f"simulated {a.clips} clips, {len(d.images):,} frames, {len(d.gt):,} boxes")

    cfg = load_config()
    for cand in ("candidate", "candidate_good"):
        pair = {"current": runs["current"], "candidate": runs[cand]}
        r = run_gate(d.images, d.gt, pair["current"], pair["candidate"], cfg)
        rep = out / "reports" / f"current_vs_{cand}"
        report.write_report(r, d.gt, d.images, d.videos, pair, rep)
        print(f"current vs {cand}: {r.decision}  -> {rep / 'summary.md'}")


def cmd_study(a):
    """Each part caches its tables as CSV next to the report, so one part can be
    rerun without redoing the others (the whole thing is ~45 min on two cores)."""
    from releasegate import report, studies

    out = Path(a.out)
    cache = out.parent / "studies"
    cache.mkdir(parents=True, exist_ok=True)
    parts = set(a.parts.split(","))

    if "unit" in parts:
        print("resampling unit study...", flush=True)
        studies.resampling_unit_study(reps=a.reps_null, workers=a.workers).to_csv(
            cache / "resampling_unit.csv", index=False
        )
    if "gate" in parts:
        print("gate error rates, 200 clips...", flush=True)
        summary, checks = studies.gate_error_rates(reps=a.reps, workers=a.workers)
        summary.to_csv(cache / "gate_summary.csv", index=False)
        checks.to_csv(cache / "gate_checks.csv", index=False)
    if "size" in parts:
        print("gate error rates, 600 clips...", flush=True)
        summary, _ = studies.gate_error_rates(
            reps=a.reps_large,
            n_videos=600,
            workers=a.workers,
            scenarios=["No change", "Subtle night regression"],
        )
        summary.to_csv(cache / "size_summary.csv", index=False)

    def table(name, digits=2):
        f = cache / f"{name}.csv"
        return report.md_table(pd.read_csv(f), digits=digits) if f.exists() else "(not run)"

    def checks_table():
        f = cache / "gate_checks.csv"
        if not f.exists():
            return "(not run)"
        cols = [
            "scenario",
            "check",
            "true_diff",
            "margin",
            "truly_worse",
            "PASS",
            "FAIL",
            "INCONCLUSIVE",
        ]
        return report.md_table(pd.read_csv(f)[cols], digits=3)

    text = f"""# Simulation studies

Generated by `releasegate study`. The correct decision for each scenario comes
from computing every check on a 2,000-clip eval set from the same simulator.

## Does the gate make the right call? (200-clip eval sets)

Share of simulated eval sets ending in each decision.

{table("gate_summary")}

### First version, for comparison

Before FAIL required the interval to be below zero, the gate held an unchanged
model a quarter of the time:

{table("gate_summary_v1")}

## Bigger eval set (600 clips)

The pretrained detectors never saw BDD100K, so its training clips are fair
evaluation data too. Three times the clips makes every interval about 1.7x
narrower.

{table("size_summary")}

## Per check (200 clips)

`true_diff` is candidate minus current on the large eval set; `truly_worse` means
that is beyond the check's margin.

{checks_table()}

## Why resample clips, not boxes

Two identical models, pedestrian recall difference, 90% interval. A false alarm is
an interval that excludes zero.

{table("resampling_unit", digits=3)}
"""
    out.write_text(text)
    print(f"-> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="releasegate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="BDD100K tracking labels -> parquet")
    s.add_argument("--labels", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--info", help="BDD100K info folder, for GPS and speed")
    s.add_argument(
        "--estimate-lighting",
        metavar="IMAGES",
        help="fill missing time of day from frame brightness",
    )
    s.add_argument("--limit", type=int, help="random subset of clips")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("infer", help="run a YOLO model over the frames")
    s.add_argument("--data", required=True)
    s.add_argument("--images", required=True, help="root folder of the frames")
    s.add_argument("--weights", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--name")
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--conf", type=float, default=0.01)
    s.add_argument("--device")
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--limit", type=int, help="only the first N clips (for a quick test)")
    s.set_defaults(fn=cmd_infer)

    s = sub.add_parser("match", help="match predictions to labels")
    s.add_argument("--data", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--iou", type=float, default=0.5)
    s.set_defaults(fn=cmd_match)

    s = sub.add_parser("gate", help="compare two models and decide")
    s.add_argument("--data", required=True)
    s.add_argument("--current", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.add_argument("--set", action="append", help="scenario set JSON (glob ok), repeatable")
    s.add_argument("--n-boot", type=int)
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("sql", help="run a saved query against a gate report")
    s.add_argument("name", nargs="?")
    s.add_argument("--report")
    s.set_defaults(fn=cmd_sql)

    s = sub.add_parser("scenario", help="build a rule-based scenario set")
    s.add_argument("--data", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--where", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--notes")
    s.set_defaults(fn=cmd_scenario)

    s = sub.add_parser("mine", help="CLIP scenario mining: embed, check, search")
    s.add_argument("action", choices=["embed", "check", "search"])
    s.add_argument("--data", required=True)
    s.add_argument("--emb", required=True, help=".npz with frame embeddings")
    s.add_argument("--images")
    s.add_argument("--query")
    s.add_argument("--label", help="pandas query over images for `check`")
    s.add_argument("--name")
    s.add_argument("--out")
    s.add_argument("--k", type=int, default=200)
    s.add_argument("--stride", type=int, default=5)
    s.add_argument("--model", default="openai/clip-vit-base-patch32")
    s.set_defaults(fn=cmd_mine)

    s = sub.add_parser("demo", help="whole pipeline on simulated clips")
    s.add_argument("--out", default="demo")
    s.add_argument("--clips", type=int, default=200)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("study", help="simulation studies of the gate itself")
    s.add_argument("--out", default="reports/studies.md")
    s.add_argument("--reps", type=int, default=40)
    s.add_argument("--reps-null", type=int, default=200)
    s.add_argument("--reps-large", type=int, default=20, help="reps for the 600-clip runs")
    s.add_argument("--parts", default="unit,gate,size", help="any of unit,gate,size")
    s.add_argument("--workers", type=int, default=2)
    s.set_defaults(fn=cmd_study)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
