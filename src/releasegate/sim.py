"""Synthetic driving clips with detectors whose behaviour I control.

Two reasons this exists:

1. BDD100K can't be redistributed, and running YOLO needs a GPU-ish machine.
   The simulator writes tables with exactly the same columns as the real
   ingest + inference steps, so the whole pipeline, the dashboard and CI run
   without either.

2. Grading the gate. Here I know whether the candidate model is really worse
   at night pedestrians, because I wrote that into it. So I can check how
   often the gate ships a real regression, and how often it blocks a model
   that's actually fine (studies.py).

The world is deliberately simple:

  * a camera 1.3 m high with f = 1000 px looks down the road; the ego car
    drives at a speed that depends on the scene
  * objects spawn at 12-70 m, in lanes (vehicles) or on the sidewalk
    (pedestrians, some of whom cross), and move relative to the camera
  * the pinhole model turns metres into a box, so boxes grow as the car
    approaches, exactly the geometry safety.py inverts
  * a box is occluded if nearer boxes cover a quarter of it

Each model gives every box a latent detection strength

    z = base[class] + 1.4 * log(height / 60 px) - 1.0 * occluded
        + time-of-day + weather + model-specific terms
        + clip effect + track effect + AR(1) frame noise

and outputs score = sigmoid(z). The clip and track effects are partly shared
between models (a hard clip is hard for everyone). The AR(1) noise is what
produces flicker: a model with noisier frame-to-frame scores drops objects
more often even if its average recall is the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit

from releasegate.bdd import finish_gt, summarize_videos

IMG_W, IMG_H = 1280, 720
FOCAL = 1000.0
HORIZON_Y = 330.0
CAM_HEIGHT = 1.3
CONF_FLOOR = 0.01

GEOMETRY = {  # height, width in metres
    "car": (1.5, 1.8),
    "truck": (3.0, 2.5),
    "bus": (3.2, 2.6),
    "pedestrian": (1.7, 0.6),
    "bicycle": (1.0, 0.7),
    "motorcycle": (1.1, 0.8),
}
BASE_LOGIT = {
    "car": 1.4,
    "truck": 1.0,
    "bus": 1.1,
    "pedestrian": 0.9,
    "bicycle": 0.1,
    "motorcycle": 0.3,
}
SPAWN_RATE = {  # new objects per frame, by scene
    "city street": {
        "car": 0.28,
        "pedestrian": 0.22,
        "truck": 0.04,
        "bus": 0.02,
        "bicycle": 0.04,
        "motorcycle": 0.02,
    },
    "residential": {"car": 0.14, "pedestrian": 0.12, "truck": 0.02, "bicycle": 0.03},
    "highway": {"car": 0.40, "truck": 0.10, "bus": 0.02, "motorcycle": 0.02},
}
EGO_SPEED = {"city street": 9.0, "residential": 6.0, "highway": 24.0}
CITIES = {"new_york": (40.735, -73.990), "bay_area": (37.800, -122.270)}


@dataclass(frozen=True)
class ModelSpec:
    offset: float = 0.0
    pedestrian: float = 0.0
    night: float = 0.0
    small_pedestrian_night: float = 0.0  # pedestrians under 45 px at night
    frame_sd: float = 0.8  # AR(1) noise on the score, the flicker knob
    fp_per_frame: float = 2.0


@dataclass(frozen=True)
class SimConfig:
    n_videos: int = 200
    frames_per_video: int = 100
    fps: float = 5.0
    timeofday: dict = field(
        default_factory=lambda: {"daytime": 0.55, "night": 0.35, "dawn/dusk": 0.10}
    )
    weather: dict = field(default_factory=lambda: {"clear": 0.65, "overcast": 0.15, "rainy": 0.20})
    scene: dict = field(
        default_factory=lambda: {"city street": 0.60, "residential": 0.15, "highway": 0.25}
    )
    time_effect: dict = field(
        default_factory=lambda: {"daytime": 0.0, "night": -0.5, "dawn/dusk": -0.2}
    )
    weather_effect: dict = field(
        default_factory=lambda: {"clear": 0.0, "overcast": -0.05, "rainy": -0.3}
    )
    height_slope: float = 1.4
    occluded_effect: float = -1.0
    truncated_effect: float = -0.5
    sd_video: float = 0.5
    sd_video_model: float = 0.2
    sd_track: float = 0.6
    sd_track_model: float = 0.25
    ar_rho: float = 0.6
    p_crossing: float = 0.2
    p_missing_gps: float = 0.1
    models: dict = field(
        default_factory=lambda: {
            "current": ModelSpec(),
            # better on average, worse on far pedestrians at night, noisier frame to frame
            "candidate": ModelSpec(offset=0.5, small_pedestrian_night=-1.6, frame_sd=1.1),
            # a plain improvement
            "candidate_good": ModelSpec(offset=0.3, pedestrian=0.15),
        }
    )


@dataclass
class SimData:
    images: pd.DataFrame
    gt: pd.DataFrame
    videos: pd.DataFrame
    preds: dict
    latent: dict  # model -> gt_match-shaped table from the generator itself


def _draw(rng, probs: dict, n: int):
    keys = list(probs)
    p = np.array([probs[k] for k in keys], float)
    return np.array(keys, dtype=object)[rng.choice(len(keys), size=n, p=p / p.sum())]


def _spawn(rng, cls, scene, v_ego, k0, n_frames):
    """Trajectory of one object: arrays of (frame_index, distance, lateral)."""
    d0 = rng.uniform(12, 70)
    if cls == "pedestrian":
        side = rng.choice([-1, 1])
        if rng.random() < 0.15:
            # crossing: often steps out from behind parked cars, so closer
            x0, vx, d0 = side * rng.uniform(2.5, 5), -side * 1.4, rng.uniform(10, 40)
        else:
            x0, vx = side * rng.uniform(3, 9), 0.0
        v_rel = v_ego
    elif cls == "bicycle":
        x0 = rng.choice([-1, 1]) * rng.uniform(2, 4)
        vx, v_rel = 0.0, v_ego - 5.0
    else:
        lanes = [-3.6, 0.0, 3.6]
        x0 = rng.choice(lanes) + rng.normal(0, 0.3)
        vx = 0.0
        if scene != "highway" and x0 < -2:  # oncoming lane
            v_rel = v_ego + rng.uniform(5, 12)
        else:
            v_rel = rng.normal(0, 1.5 if abs(x0) < 1 else 3.0)
    k = np.arange(k0, n_frames)
    t = (k - k0) / 5.0
    d = d0 - v_rel * t
    x = x0 + vx * t
    # objects also leave for reasons the camera can't see (turns, parking)
    life = rng.geometric(1 / 30)
    keep = (d > 3) & (d < 100) & (np.arange(len(k)) < life)
    # stop at the first frame it leaves the valid range
    if not keep.all():
        stop = np.argmin(keep)
        k, d, x = k[:stop], d[:stop], x[:stop]
    return k, d, x


def _boxes(cls, d, x):
    H, W = GEOMETRY[cls]
    h = FOCAL * H / d
    w = FOCAL * W / d
    bottom = HORIZON_Y + FOCAL * CAM_HEIGHT / d
    cx = IMG_W / 2 + FOCAL * x / d
    return cx - w / 2, bottom - h, cx + w / 2, bottom


def _occlusion(frame_pos: np.ndarray, dist: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """A box is occluded if nearer boxes cover more than 25% of it."""
    occ = np.zeros(len(dist), bool)
    order = np.argsort(frame_pos, kind="mergesort")
    starts = np.r_[0, np.nonzero(np.diff(frame_pos[order]))[0] + 1, len(order)]
    for a, b in zip(starts[:-1], starts[1:], strict=False):
        idx = order[a:b]
        if len(idx) < 2:
            continue
        bx = boxes[idx]
        ix = np.clip(
            np.minimum(bx[:, None, 2], bx[None, :, 2]) - np.maximum(bx[:, None, 0], bx[None, :, 0]),
            0,
            None,
        )
        iy = np.clip(
            np.minimum(bx[:, None, 3], bx[None, :, 3]) - np.maximum(bx[:, None, 1], bx[None, :, 1]),
            0,
            None,
        )
        nearer = dist[idx][None, :] < dist[idx][:, None]
        own = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
        occ[idx] = (ix * iy * nearer).sum(1) > 0.25 * own
    return occ


def simulate(cfg: SimConfig = SimConfig(), seed: int = 0) -> SimData:
    rng = np.random.default_rng(seed)
    nv, nf = cfg.n_videos, cfg.frames_per_video
    vids = np.array([f"sim{seed:03d}-{i:04d}" for i in range(nv)])
    tod = _draw(rng, cfg.timeofday, nv)
    wx = _draw(rng, cfg.weather, nv)
    scene = _draw(rng, cfg.scene, nv)
    city = _draw(rng, {c: 1 for c in CITIES}, nv)
    v_ego = np.array([EGO_SPEED[s] for s in scene]) * np.exp(rng.normal(0, 0.15, nv))

    # ---- frames -------------------------------------------------------------
    fr_video = np.repeat(np.arange(nv), nf)
    fr_k = np.tile(np.arange(nf), nv)
    image_id = np.array(
        [f"{vids[v]}/{vids[v]}-{k + 1:07d}.jpg" for v, k in zip(fr_video, fr_k, strict=False)]
    )
    heading = rng.uniform(0, 2 * np.pi, nv)
    lat0 = np.array([CITIES[c][0] for c in city]) + rng.normal(0, 0.03, nv)
    lon0 = np.array([CITIES[c][1] for c in city]) + rng.normal(0, 0.03, nv)
    t_s = fr_k / cfg.fps
    travelled = v_ego[fr_video] * t_s
    lat = lat0[fr_video] + travelled * np.cos(heading[fr_video]) / 111_000
    lon = lon0[fr_video] + travelled * np.sin(heading[fr_video]) / (
        111_000 * np.cos(np.radians(lat0[fr_video]))
    )
    speed = v_ego[fr_video] + rng.normal(0, 0.3, len(fr_k))
    no_gps = rng.random(nv) < cfg.p_missing_gps
    for arr in (lat, lon, speed):
        arr[no_gps[fr_video]] = np.nan
    images = pd.DataFrame(
        {
            "image_id": image_id,
            "video": vids[fr_video],
            "frame_index": fr_k,
            "timeofday": tod[fr_video],
            "weather": wx[fr_video],
            "scene": scene[fr_video],
            "width": IMG_W,
            "height": IMG_H,
            "lat": lat,
            "lon": lon,
            "speed_mps": speed,
            "location": city[fr_video],
        }
    )

    # ---- objects ------------------------------------------------------------
    rows = []  # (video, track, cls, k, d, x)
    track_n = 0
    for v in range(nv):
        for cls, rate in SPAWN_RATE[scene[v]].items():
            # some objects are already in view at frame 0 (left-censored)
            starts = np.r_[
                np.zeros(rng.poisson(rate * 6), int), np.nonzero(rng.random(nf) < rate)[0]
            ]
            for k0 in starts:
                k, d, x = _spawn(rng, cls, scene[v], v_ego[v], int(k0), nf)
                if len(k) == 0:
                    continue
                rows.append(
                    (
                        np.full(len(k), v),
                        np.full(len(k), track_n),
                        np.full(len(k), cls, dtype=object),
                        k,
                        d,
                        x,
                    )
                )
                track_n += 1
    ov, ot, oc, ok, od, ox = (np.concatenate([r[i] for r in rows]) for i in range(6))
    x1, y1, x2, y2 = np.empty(len(ov)), np.empty(len(ov)), np.empty(len(ov)), np.empty(len(ov))
    for cls in GEOMETRY:
        m = oc == cls
        x1[m], y1[m], x2[m], y2[m] = _boxes(cls, od[m], ox[m])
    full_area = (x2 - x1) * (y2 - y1)
    cx1, cy1 = np.clip(x1, 0, IMG_W), np.clip(y1, 0, IMG_H)
    cx2, cy2 = np.clip(x2, 0, IMG_W), np.clip(y2, 0, IMG_H)
    vis_area = np.clip(cx2 - cx1, 0, None) * np.clip(cy2 - cy1, 0, None)
    keep = (vis_area > 0.3 * full_area) & ((cy2 - cy1) >= 8)
    truncated = vis_area < 0.95 * full_area

    ov, ot, oc, ok, od, ox = ov[keep], ot[keep], oc[keep], ok[keep], od[keep], ox[keep]
    box = np.c_[cx1, cy1, cx2, cy2][keep]
    truncated = truncated[keep]
    frame_pos = ov * nf + ok
    occluded = _occlusion(frame_pos, od, box)

    gt = finish_gt(
        pd.DataFrame(
            {
                "image_id": image_id[frame_pos],
                "video": vids[ov],
                "frame_index": ok,
                "track_id": [f"{vids[v]}:{t}" for v, t in zip(ov, ot, strict=False)],
                "cls": oc,
                "label": oc,
                "ignore": False,
                "x1": box[:, 0],
                "y1": box[:, 1],
                "x2": box[:, 2],
                "y2": box[:, 3],
                "occluded": occluded,
                "truncated": truncated,
            }
        )
    )
    # finish_gt keeps row order, so the arrays above still line up with gt.

    # ---- models -------------------------------------------------------------
    h = box[:, 3] - box[:, 1]
    log_h = np.log(h / 60.0)
    is_ped = oc == "pedestrian"
    is_night = tod[ov] == "night"
    fixed = (
        np.array([BASE_LOGIT[c] for c in oc])
        + cfg.height_slope * log_h
        + cfg.occluded_effect * occluded
        + cfg.truncated_effect * truncated
        + np.array([cfg.time_effect[t] for t in tod])[ov]
        + np.array([cfg.weather_effect[w] for w in wx])[ov]
    )
    u_video = cfg.sd_video * rng.standard_normal(nv)
    u_track = cfg.sd_track * rng.standard_normal(track_n)
    p_bad = expit(-3.2 - 1.3 * log_h)

    order = np.lexsort((ok, ot))  # by track, then frame
    new_track = np.r_[True, ot[order][1:] != ot[order][:-1]]

    preds, latent = {}, {}
    next_id = 0
    for name, m in cfg.models.items():
        eta = (
            fixed
            + m.offset
            + m.pedestrian * is_ped
            + m.night * is_night
            + m.small_pedestrian_night * (is_ped & is_night & (h < 45))
            + u_video[ov]
            + cfg.sd_video_model * rng.standard_normal(nv)[ov]
            + u_track[ot]
            + cfg.sd_track_model * rng.standard_normal(track_n)[ot]
        )
        # AR(1) frame noise along each track
        shocks = rng.standard_normal(len(order))
        e = np.empty(len(order))
        rho, s = cfg.ar_rho, np.sqrt(1 - cfg.ar_rho**2)
        for i in range(len(order)):
            e[i] = shocks[i] if new_track[i] else rho * e[i - 1] + s * shocks[i]
        noise = np.empty(len(order))
        noise[order] = e
        score = expit(eta + m.frame_sd * noise)
        bad = rng.random(len(score)) < p_bad
        emitted = score >= CONF_FLOOR

        w, hh = box[:, 2] - box[:, 0], h
        jit = rng.uniform(-0.04, 0.04, (len(score), 4))
        pb = box + jit * np.c_[w, hh, w, hh]
        cxm, cym = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
        shrunk = np.c_[cxm - 0.15 * w, cym - 0.15 * hh, cxm + 0.15 * w, cym + 0.15 * hh]
        pb = np.where(bad[:, None], shrunk, pb)

        real = pd.DataFrame(
            {
                "image_id": image_id[frame_pos],
                "cls": oc,
                "score": score,
                "x1": pb[:, 0],
                "y1": pb[:, 1],
                "x2": pb[:, 2],
                "y2": pb[:, 3],
            }
        )[emitted]

        n_fp = rng.poisson(m.fp_per_frame, len(image_id))
        fp_frame = np.repeat(np.arange(len(image_id)), n_fp)
        fh = np.exp(rng.uniform(np.log(12), np.log(90), len(fp_frame)))
        fw = fh * rng.uniform(0.4, 1.6, len(fp_frame))
        fx = rng.uniform(0, IMG_W - fw)
        fy = rng.uniform(HORIZON_Y - 60, IMG_H - fh)
        fake = pd.DataFrame(
            {
                "image_id": image_id[fp_frame],
                "cls": _draw(
                    rng, {"car": 4, "pedestrian": 3, "truck": 1, "bicycle": 1}, len(fp_frame)
                ),
                "score": expit(rng.normal(-1.8, 1.2, len(fp_frame))),
                "x1": fx,
                "y1": fy,
                "x2": fx + fw,
                "y2": fy + fh,
            }
        )
        p = pd.concat([real, fake[fake["score"] >= CONF_FLOOR]], ignore_index=True)
        p.insert(0, "pred_id", np.arange(next_id, next_id + len(p), dtype=np.int64))
        p.insert(1, "model", name)
        next_id += len(p)
        preds[name] = p

        latent[name] = pd.DataFrame(
            {
                "gt_id": gt["gt_id"].to_numpy(),
                "image_id": gt["image_id"].to_numpy(),
                "cls": oc,
                "pred_id": -1,
                "score": np.where(emitted & ~bad, score, np.nan),
                "iou": np.nan,
            }
        )

    videos = summarize_videos(images).merge(
        images.groupby("video")
        .agg(lat=("lat", "median"), lon=("lon", "median"), speed_mps=("speed_mps", "median"))
        .reset_index(),
        on="video",
    )
    videos["city"] = city
    return SimData(images, gt, videos, preds, latent)
