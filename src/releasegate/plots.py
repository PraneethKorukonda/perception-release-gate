"""Static figures for the report (matplotlib, PNG).

Colours: current model is always blue, candidate always orange, in every
figure. Verdicts are written as text next to each row, never shown by colour
alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CURRENT = "#2a78d6"
CANDIDATE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w, h, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE, squeeze=False)
    for ax in axes.ravel():
        _style(ax)
    return fig, axes.ravel()


def gate_checks(checks: pd.DataFrame, path: Path) -> None:
    """One row per check: improvement with its interval, and the allowed margin."""
    c = checks.iloc[::-1].reset_index(drop=True)
    sign = np.where(c["higher_is_better"], 1.0, -1.0)
    # put everything on "points of improvement" so right is always better
    scale = np.where(c["metric"].isin(["dropouts_per_100", "fp_per_frame"]), 1.0, 100.0)
    est = sign * c["diff"] * scale
    lo = np.minimum(sign * c["diff_lo"], sign * c["diff_hi"]) * scale
    hi = np.maximum(sign * c["diff_lo"], sign * c["diff_hi"]) * scale
    margin = -c["margin"] * scale

    fig, (ax,) = _fig(9, 0.45 * len(c) + 1.2)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(c))
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.hlines(y, lo, hi, color=INK, linewidth=2)
    ax.plot(est, y, "o", color=INK, markersize=6)
    ax.plot(margin, y, "|", color=CANDIDATE, markersize=14, markeredgewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels(c["check"], color=INK, fontsize=9)
    ax.set_xlabel(
        "candidate vs current  (right = better; points, or raw units for rates per frame/100)"
    )
    xmax = np.nanmax(np.abs(np.r_[lo, hi, margin])) * 1.15
    ax.set_xlim(-xmax, xmax)
    for yi, v in zip(y, c["verdict"], strict=True):
        ax.text(
            xmax * 1.02,
            yi,
            v,
            va="center",
            fontsize=9,
            color=INK,
            fontweight="bold" if v == "FAIL" else "normal",
            clip_on=False,
        )
    ax.set_title(
        "Gate checks: estimate, 90% interval, and allowed margin (orange tick)",
        loc="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def time_to_detection(tracks: dict, path: Path, cls: str = "pedestrian") -> None:
    """Share of objects detected within t seconds of appearing, day vs night."""
    fig, axes = _fig(9, 3.4, ncols=2)
    t_grid = np.linspace(0, 3, 61)
    for ax, (label, cond) in zip(axes, [("Day and dusk", False), ("Night", True)], strict=True):
        for role, color in (("current", CURRENT), ("candidate", CANDIDATE)):
            t = tracks[role]
            t = t[(t["cls"] == cls) & t["detected_within_1s"].notna() & (t["night"] == cond)]
            ttfd = t["ttfd_s"].fillna(np.inf).to_numpy()
            share = [(ttfd <= g).mean() for g in t_grid]
            ax.plot(t_grid, share, color=color, linewidth=2, label=role)
        ax.set_title(f"{label}  (n = {len(t):,})", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("seconds since the pedestrian appeared")
        ax.set_ylim(0.5, 1.0)
        ax.axvline(1.0, color=GRID, linewidth=1, linestyle="--")
    axes[0].set_ylabel("share detected")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def recall_by_distance(objects: dict, path: Path, cls: str = "pedestrian") -> None:
    bins = [0, 10, 20, 30, 40, 50, 70, 200]
    labels = ["<10", "10-20", "20-30", "30-40", "40-50", "50-70", "70+"]
    fig, axes = _fig(9, 3.4, ncols=2)
    for ax, (label, cond) in zip(axes, [("Day and dusk", False), ("Night", True)], strict=True):
        x = np.arange(len(labels))
        for k, (role, color) in enumerate((("current", CURRENT), ("candidate", CANDIDATE))):
            o = objects[role]
            o = o[(o["cls"] == cls) & (o["night"] == cond) & o["distance_m"].notna()]
            b = pd.cut(o["distance_m"], bins, labels=labels)
            r = o.groupby(b, observed=False)["det"].mean().reindex(labels)
            ax.bar(x + (k - 0.5) * 0.38, r.to_numpy(), width=0.36, color=color, label=role)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("estimated distance (m)")
        ax.set_ylim(0, 1)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("recall")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def clip_map(objects: dict, videos: pd.DataFrame, path: Path, cls: str = "pedestrian") -> bool:
    """Per-clip change in pedestrian recall, on a lat/lon scatter per city."""
    if "lat" not in videos or videos["lat"].isna().all():
        return False
    rows = []
    for role in ("current", "candidate"):
        o = objects[role]
        rows.append(o[o["cls"] == cls].groupby("video")["det"].mean().rename(role))
    d = pd.concat(rows, axis=1).join(videos.set_index("video")[["lat", "lon"]]).dropna()
    d["diff"] = 100 * (d["candidate"] - d["current"])
    # group clips into regions that are far apart (BDD is mostly NYC and the Bay Area)
    d["region"] = np.where(d["lon"] < -100, "San Francisco Bay Area", "New York area")
    regions = sorted(d["region"].unique())
    fig, axes = _fig(4.6 * len(regions), 4.2, ncols=len(regions))
    lim = max(5.0, np.nanpercentile(np.abs(d["diff"]), 95))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#e34948", "#f0efec", "#2a78d6"]
    )
    for ax, reg in zip(axes, regions, strict=True):
        g = d[d["region"] == reg]
        sc = ax.scatter(
            g["lon"],
            g["lat"],
            c=g["diff"],
            cmap=cmap,
            vmin=-lim,
            vmax=lim,
            s=30,
            edgecolors=SURFACE,
            linewidths=1,
        )
        ax.set_title(f"{reg}  ({len(g)} clips)", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("longitude")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_ylabel("latitude")
    cb = fig.colorbar(sc, ax=list(axes), shrink=0.8)
    cb.set_label("pedestrian recall change, candidate - current (pts)", color=INK_2, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
