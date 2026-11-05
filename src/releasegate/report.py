"""Write the gate's outputs: parquet tables, figures, and two markdown files.

summary.md   one page for the people who decide. Verdict first, then why,
             then what to look at. No jargon past "recall".
details.md   everything an engineer needs to check the verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from releasegate import db, plots
from releasegate.gate import GateResult
from releasegate.metrics import class_table

PCT_METRICS = {"frame_recall", "detected_within_1s", "late_in_path", "map50"}


def _fmt(metric: str, v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    if metric in PCT_METRICS:
        return f"{100 * v:.1f}%"
    return f"{v:.2f}"


def _fmt_diff(metric: str, v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    if metric in PCT_METRICS:
        return f"{100 * v:+.1f} pts"
    return f"{v:+.2f}"


def save_tables(result: GateResult, images, videos, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([o.assign(model=r) for r, o in result.objects.items()]).to_parquet(
        out / "objects.parquet", index=False
    )
    pd.concat([t.assign(model=r) for r, t in result.tracks.items()]).to_parquet(
        out / "tracks.parquet", index=False
    )
    result.checks.to_parquet(out / "checks.parquet", index=False)
    result.checks.to_csv(out / "checks.csv", index=False)
    images.to_parquet(out / "images.parquet", index=False)
    videos.to_parquet(out / "videos.parquet", index=False)


def md_table(df: pd.DataFrame, index: bool = False, digits: int = 3) -> str:
    """Markdown table without pulling in tabulate."""
    if index:
        df = df.reset_index()
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _lower(name: str) -> str:
    """Lower-case a check name for use mid-sentence, leaving acronyms alone."""
    first, _, rest = name.partition(" ")
    if any(ch.isupper() for ch in first[1:]):
        return name
    return first.lower() + (" " + rest if rest else "")


def _sentence_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def headline(result: GateResult, names: dict) -> str:
    c = result.checks
    # improvement interval lower end > 0 means clearly better
    worst_case = np.where(c["higher_is_better"], c["diff_lo"], -c["diff_hi"])
    better = c[worst_case > 0]
    fails = c[(c["verdict"] == "FAIL") & c["blocking"]]
    unsure = c[(c["verdict"] == "INCONCLUSIVE") & c["blocking"]]

    parts = []
    if len(better):
        bits = [f"{_lower(r.check)} ({_fmt_diff(r.metric, r.diff)})" for r in better.itertuples()]
        who = "The candidate"
        if names["candidate"] != "candidate":
            who += f" ({names['candidate']})"
        parts.append(f"{who} is clearly better on {_sentence_list(bits[:3])}.")
    if len(fails):
        bits = [
            f"{_lower(r.check)} ({_fmt(r.metric, r.current)} -> {_fmt(r.metric, r.candidate)})"
            for r in fails.itertuples()
        ]
        parts.append(
            f"It is worse than allowed on {_sentence_list(bits)}. "
            "These checks block the release, so the overall gains don't make up for them."
        )
    if len(unsure):
        bits = [_lower(r.check) for r in unsure.itertuples()]
        parts.append(
            f"The eval set is too small to tell on {_sentence_list(bits)}: "
            "the interval is wider than the allowed margin. Adding clips of that kind "
            "is the fix, not lowering the bar."
        )
    if result.decision == "SHIP" and not len(fails) and not len(unsure):
        parts.append("No blocking check shows a regression beyond its margin.")
    return " ".join(parts)


def check_table_md(c: pd.DataFrame) -> str:
    lines = [
        "| Check | Current | Candidate | Change (90% CI) | Allowed | Verdict |",
        "|---|---:|---:|---|---:|---|",
    ]
    for r in c.itertuples():
        ci = f"{_fmt_diff(r.metric, r.diff)} ({_fmt_diff(r.metric, r.diff_lo)} to {_fmt_diff(r.metric, r.diff_hi)})"
        allowed = _fmt_diff(r.metric, -r.margin if r.higher_is_better else r.margin)
        flag = "" if r.blocking else " (info)"
        lines.append(
            f"| {r.check} | {_fmt(r.metric, r.current)} | {_fmt(r.metric, r.candidate)} "
            f"| {ci} | {allowed} | **{r.verdict}**{flag} |"
        )
    return "\n".join(lines)


def write_summary(result: GateResult, names: dict, out: Path, review: dict) -> Path:
    n_clips = len(result.clips)
    o = result.objects["current"]
    n_ped_tracks = int((result.tracks["current"]["cls"] == "pedestrian").sum())
    text = f"""# Release review: {names["candidate"]} vs {names["current"]}

## Decision: {result.decision}

{headline(result, names)}

Evaluated on {n_clips} driving clips ({o["image_id"].nunique():,} frames, {len(o):,} labelled boxes,
{n_ped_tracks:,} pedestrians followed over time). Score thresholds: current
{result.config.thresholds["current"]}, candidate {result.config.thresholds["candidate"]}.
Intervals are 90% and resample whole clips.

{check_table_md(result.checks)}

![gate checks](figures/gate_checks.png)

## Where to look

- **Clips with the most pedestrians lost:** {", ".join(review["clips"]) or "none"}
- **In-path pedestrians the candidate saw too late to stop for, but the current model didn't:** {review["late"]}
- **Weakest slices** (exploratory, see details.md): {review["slices"] or "none below the current model"}

## What this doesn't cover

- 2D camera boxes only. No lidar/radar, no 3D position, no tracker.
- Stopping-distance checks use a flat-road, straight-path approximation (and an
  assumed focal length where the dataset has no calibration). They rank models;
  they are not a safety case.
- Scenario margins were set by me, not derived from a system-level budget.
"""
    path = out / "summary.md"
    path.write_text(text)
    return path


def write_details(
    result: GateResult, gt, runs: dict, images, out: Path, slices: pd.DataFrame
) -> Path:
    sections = ["# Details\n"]

    sections.append("## Per-class results at the deployment threshold\n")
    for role, run in runs.items():
        if run.pred_match is None:
            continue
        t = class_table(run.gt_match, run.pred_match, len(images), result.config.thresholds[role])
        t = t[["cls", "n_gt", "ap50", "precision", "recall", "f1", "fp_per_image"]]
        sections.append(f"**{role}: {run.name}**\n\n" + md_table(t) + "\n")

    o = result.objects["current"]
    comp = (
        o[o["cls"] == "pedestrian"]
        .groupby("timeofday")
        .agg(
            boxes=("gt_id", "size"),
            median_distance_m=("distance_m", "median"),
            share_small=("small", "mean"),
            share_occluded=("occluded", "mean"),
        )
    )
    sections.append(
        "## What the pedestrians look like in each condition\n\n"
        "Before blaming lighting for a night gap, check whether night pedestrians are "
        "simply further away or more occluded in this eval set.\n\n"
        + md_table(comp, index=True)
        + "\n"
    )

    sections.append(
        "## Biggest recall drops by slice (exploratory)\n\n"
        "Paired on the same boxes, slices with at least 200 boxes and 5 clips. Many slices "
        "are scanned, so some will look bad by chance; promote a slice to a gate check "
        "before acting on it.\n\n" + md_table(slices.head(10), digits=4) + "\n"
    )

    tr = {r: t[t["cls"] == "pedestrian"] for r, t in result.tracks.items()}
    rows = []
    for r, t in tr.items():
        e = t[t["detected_within_1s"].notna()]
        rows.append(
            {
                "model": r,
                "pedestrians (eligible)": len(e),
                "never detected": int(e["first_det_frame"].isna().sum()),
                "median s to first detection": e["ttfd_s"].median(),
                "90th pct s": e["ttfd_s"].quantile(0.9),
                "dropouts / 100 det. frames": 100
                * t["dropouts"].sum()
                / max(t["n_det_frames"].sum(), 1),
            }
        )
    sections.append("## Pedestrians over time\n\n" + md_table(pd.DataFrame(rows), digits=2) + "\n")
    sections.append("![time to first detection](figures/time_to_detection.png)\n")
    sections.append("![recall by distance](figures/recall_by_distance.png)\n")
    if (out / "figures" / "clip_map.png").exists():
        sections.append("## By location\n\n![map](figures/clip_map.png)\n")

    path = out / "details.md"
    path.write_text("\n".join(sections))
    return path


def write_report(result: GateResult, gt, images, videos, runs: dict, out_dir) -> dict:
    out = Path(out_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    save_tables(result, images, videos, out)

    plots.gate_checks(result.checks, figs / "gate_checks.png")
    plots.time_to_detection(result.tracks, figs / "time_to_detection.png")
    plots.recall_by_distance(result.objects, figs / "recall_by_distance.png")
    plots.clip_map(result.objects, videos, figs / "clip_map.png")

    con = db.connect(out)
    clips = db.run(con, "failure_clips").head(3)
    late = db.run(con, "late_pedestrians")
    slices = db.run(con, "biggest_regressions")
    worst = slices[slices["diff"] < 0].head(3)
    review = {
        "clips": [
            f"`{v}`"
            for v in clips.loc[clips["lost_by_candidate"] > clips["gained_by_candidate"], "video"]
        ],
        "late": f"{len(late)} (listed by `releasegate sql late_pedestrians`)",
        "slices": "; ".join(
            f"{r.cls} / {r.timeofday} / {r.weather} / {r.size} ({100 * r.diff:+.1f} pts)"
            for r in worst.itertuples()
        ),
    }
    names = {role: run.name for role, run in runs.items()}
    return {
        "summary": write_summary(result, names, out, review),
        "details": write_details(result, gt, runs, images, out, slices),
    }
