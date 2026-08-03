"""How far can motion alone follow one player? Two leads, measured.

`scripts/follow_player.py` chains lanes from a hand-verified seed and shows the
chain holds a player perfectly — for 12.4 s, and then stops. This asks *why* it
stops and whether two structural fixes extend it without the halo sliding onto
somebody else.

Both leads come from looking at the frames at the break rather than at the gate:

1. **Overlapping births.** ByteTrack's most common way of losing a long lane is
   not a dropout — it is a *duplicate box*. A second detection lands on the same
   player, is given a fresh id, and the old lane dies a frame or two later. On
   the 30 fps U14G run that is how **9.9% of lanes lasting >=10 s end**. The
   linker cannot join those pairs by construction: ``score_pair`` requires
   ``b.first_frame > a.last_frame``, so a successor born *before* its
   predecessor died is rejected as "overlapping in time". The pair is the easiest
   possible link — same player, same frame, IoU>0.5 — and it is the one case the
   gate refuses.

2. **Camera pan.** This is a Veo camera that pans constantly: the median frame
   carries 1.35 px of global image motion and the 99th percentile carries 10.8
   px/frame (~325 px/s). Every motion model in the stack — ByteTrack's Kalman and
   this linker's velocity extrapolation — measures velocity in *image* pixels, so
   a standing player has apparent velocity and a running one has the wrong
   velocity. Extrapolating a lane forward across a gap therefore aims at where
   the player would be if the camera had held still. Subtracting the global pan
   first costs one median per frame and makes the extrapolation mean what it
   says.

Nothing here needs a model, a gallery, or the ball. Run:

    python slurm/eval_follow_leads.py --run runs/saints-u14g-full-30fps \
        --seed-track 14185
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from soccer_vision.tracking.link import LinkConfig, _find, link_tracks


# ---------------------------------------------------------------- camera pan

def estimate_pan(tracks: dict, *, min_lanes: int = 5) -> dict[int, tuple[float, float]]:
    """Per-frame global image motion, as the median box displacement.

    The median over every lane alive in both frames is a robust camera estimate:
    players move in every direction and cancel, the camera moves them all the
    same way. It needs no features, no homography and no extra decode pass — the
    boxes are already loaded. Frames seen by fewer than ``min_lanes`` lanes get
    no estimate and are treated as zero motion.
    """
    dx = collections.defaultdict(list)
    dy = collections.defaultdict(list)
    for samples in tracks.values():
        for a, b in zip(samples, samples[1:]):
            if b["frame"] - a["frame"] != 1:
                continue
            ab, bb = a["bbox"], b["bbox"]
            dx[b["frame"]].append(((bb[0] + bb[2]) - (ab[0] + ab[2])) / 2.0)
            dy[b["frame"]].append(bb[3] - ab[3])
    return {
        f: (float(np.median(v)), float(np.median(dy[f])))
        for f, v in dx.items()
        if len(v) >= min_lanes
    }


def cumulative_pan(pan: dict[int, tuple[float, float]], max_frame: int) -> np.ndarray:
    """``(N, 2)`` cumulative camera offset per frame, so world = image - offset."""
    out = np.zeros((max_frame + 2, 2), dtype=np.float64)
    acc = np.zeros(2)
    for f in range(1, max_frame + 2):
        acc += pan.get(f, (0.0, 0.0))
        out[f] = acc
    return out


def stabilise(tracks: dict, offset: np.ndarray) -> dict:
    """Copy of ``tracks`` with the camera's cumulative motion subtracted."""
    out = {}
    for tid, samples in tracks.items():
        rows = []
        for s in samples:
            f = s["frame"]
            ox, oy = offset[f] if f < len(offset) else offset[-1]
            b = s["bbox"]
            rows.append({"frame": f,
                         "bbox": [b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy]})
        out[tid] = rows
    return out


# ------------------------------------------------------- overlapping handoff

def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_player_box(a, b, *, max_centre_frac: float, max_scale: float) -> float:
    """Agreement score for "these two boxes are the same player", or 0.

    **IoU is the wrong test here** and the reason a plain IoU gate misses the
    handoffs it exists to catch. When RF-DETR mints a duplicate box the two boxes
    sit on the same player but disagree on *extent* — the real handoff that
    breaks Morgan's chain pairs a 30x59 box with a 51x83 one, same player, IoU
    0.40. IoU punishes that scale disagreement twice over. Centre distance
    measured in box heights does not, and a separate ratio test keeps a player
    from being merged with someone at a different depth.
    """
    ha, hb = a[3] - a[1], b[3] - b[1]
    if ha <= 0 or hb <= 0:
        return 0.0
    ratio = ha / hb if ha < hb else hb / ha
    if ratio < 1.0 / max_scale:
        return 0.0
    ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    d = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])) / ((ha + hb) / 2.0)
    if d > max_centre_frac:
        return 0.0
    return (1.0 - d / max_centre_frac) * ratio


def merge_overlapping_handoffs(tracks: dict, teams: dict, fps: float, *,
                               max_overlap_s: float = 0.5,
                               max_centre_frac: float = 0.4,
                               max_scale: float = 2.0) -> dict[str, str]:
    """``{successor: predecessor}`` for lanes that are one player under two ids.

    A successor qualifies when it is born no more than ``max_overlap_s`` before
    its predecessor dies *and* their boxes agree at the overlap, by
    :func:`_same_player_box`.

    **The kit gate is not optional here**, which cost a run to learn. Two boxes
    can sit on top of each other because they are one player under two ids, or
    because two players are in contact — a tackle, a shoulder-to-shoulder chase —
    and geometry cannot tell those apart. Without this gate 16.3% of the merges
    on the U14G run joined lanes the team classifier had put in *different kits*,
    and one of them let a 62 s white-kit lane be stitched to Morgan's black one.
    Lanes with no kit are left alone rather than merged on geometry alone.
    """
    births = collections.defaultdict(list)
    for tid, s in tracks.items():
        births[s[0]["frame"]].append(tid)
    by_frame = {tid: {s["frame"]: s["bbox"] for s in samples}
                for tid, samples in tracks.items()}

    win = int(round(max_overlap_s * fps))
    claimed: dict[str, str] = {}
    used_pred: set[str] = set()
    pairs = []
    for tid, samples in tracks.items():
        death, dbox = samples[-1]["frame"], samples[-1]["bbox"]
        for df in range(-win, 1):
            for cand in births.get(death + df, ()):
                if cand == tid:
                    continue
                if teams.get(tid) != teams.get(cand) or teams.get(tid) is None:
                    continue
                cbox = by_frame[cand].get(death + df)
                if cbox is None:
                    continue
                # compare at the birth frame, where both lanes have a box
                pbox = by_frame[tid].get(death + df, dbox)
                v = _same_player_box(pbox, cbox, max_centre_frac=max_centre_frac,
                                     max_scale=max_scale)
                if v > 0:
                    pairs.append((v, tid, cand))
    for v, pred, succ in sorted(pairs, key=lambda p: -p[0]):
        if succ in claimed or pred in used_pred:
            continue
        claimed[succ] = pred
        used_pred.add(pred)
    return claimed


def apply_merges(parent: dict[str, str], merges: dict[str, str]) -> dict[str, str]:
    parent = dict(parent)
    for succ, pred in merges.items():
        if succ not in parent or pred not in parent:
            continue
        ra, rb = _find(parent, pred), _find(parent, succ)
        if ra != rb:
            parent[rb] = ra
    return parent


# ------------------------------------------------------------------ scoring

def chain_rows(tracks: dict, parent: dict[str, str], seed: str):
    root = _find(parent, seed)
    rows = []
    for tid in parent:
        if _find(parent, tid) != root:
            continue
        for s in tracks.get(tid, []):
            rows.append((s["frame"], s["bbox"], tid))
    rows.sort(key=lambda r: r[0])
    seen, out = set(), []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        out.append(r)
    return out


def score(rows, fps: float) -> dict:
    if not rows:
        return {}
    jumps, prev = [], None
    lanes = set()
    for f, b, tid in rows:
        foot = ((b[0] + b[2]) / 2.0, b[3])
        lanes.add(tid)
        if prev is not None and prev[2] != tid:
            jumps.append(float(np.hypot(foot[0] - prev[1][0], foot[1] - prev[1][1])))
        prev = (f, foot, tid)
    span = (rows[-1][0] - rows[0][0]) / fps
    return {
        "span_s": round(span, 1),
        "lanes": len(lanes),
        "frames": len(rows),
        "fill": round(len(rows) / (span * fps + 1), 2) if span > 0 else 1.0,
        "jump_med": round(float(np.median(jumps)), 1) if jumps else 0.0,
        "jump_p90": round(float(np.percentile(jumps, 90)), 1) if jumps else 0.0,
        "jumps_over_60px": sum(1 for j in jumps if j > 60),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--seed-track", required=True)
    ap.add_argument("--max-gap-s", type=float, default=2.0)
    ap.add_argument("--max-dist-px", type=float, default=150.0)
    ap.add_argument("--seed-min-s", type=float, default=20.0)
    args = ap.parse_args()

    doc = json.loads((args.run / "tracks.json").read_text())
    tracks, fps = doc["tracks"], float(doc["fps"])
    maxf = max(s[-1]["frame"] for s in tracks.values())

    pan = estimate_pan(tracks)
    mags = np.abs([p[0] for p in pan.values()])
    print(f"camera pan: median |dx| {np.median(mags):.2f} px/frame, "
          f"p99 {np.percentile(mags, 99):.2f}, "
          f"{100 * (mags > 3).mean():.0f}% of frames over 3 px/frame")
    stable = stabilise(tracks, cumulative_pan(pan, maxf))

    merges = merge_overlapping_handoffs(tracks, fps)
    print(f"overlapping-birth handoffs found: {len(merges)}")

    # A population of seeds, so the answer isn't n=1. Lanes over 20 s are the
    # ones long enough that following them is worth anything, and long enough
    # that we can be fairly sure each is a single player to begin with.
    seeds = sorted(
        (tid for tid, s in tracks.items()
         if (s[-1]["frame"] - s[0]["frame"]) / fps >= args.seed_min_s),
        key=lambda t: -(tracks[t][-1]["frame"] - tracks[t][0]["frame"]),
    )
    print(f"seed population: {len(seeds)} lanes >= {args.seed_min_s:.0f}s")

    cfg = LinkConfig(max_gap_s=args.max_gap_s, max_dist_px=args.max_dist_px)
    variants = {
        "baseline": (tracks, False),
        "+ pan compensation": (stable, False),
        "+ overlap handoff": (tracks, True),
        "+ both": (stable, True),
    }
    print(f"\n{'variant':>22} {'chains':>8} | seed {args.seed_track}: "
          f"{'span_s':>7} {'lanes':>6} {'fill':>5} {'jumpP90':>8} {'>60px':>6} | "
          f"population: {'medSpan':>8} {'p90Span':>8} {'>60px/chain':>12}")
    for name, (src, use_merge) in variants.items():
        res = link_tracks({**doc, "tracks": src}, cfg)
        parent = apply_merges(res.parent, merges) if use_merge else res.parent
        # Score against the *original* boxes: pan-compensated pixels are a
        # linking space, not something to measure a halo's smoothness in.
        s = score(chain_rows(tracks, parent, args.seed_track), fps)
        spans, bad = [], []
        seen_roots = set()
        for seed in seeds:
            root = _find(parent, seed)
            if root in seen_roots:
                continue
            seen_roots.add(root)
            ps = score(chain_rows(tracks, parent, seed), fps)
            spans.append(ps["span_s"])
            bad.append(ps["jumps_over_60px"])
        n_chains = len({_find(parent, t) for t in parent})
        print(f"{name:>22} {n_chains:8d} | {'':>10} {s['span_s']:7.1f} {s['lanes']:6d} "
              f"{s['fill']:5.2f} {s['jump_p90']:8.1f} {s['jumps_over_60px']:6d} | "
              f"{'':>12} {np.median(spans):8.1f} {np.percentile(spans, 90):8.1f} "
              f"{np.mean(bad):12.2f}")


if __name__ == "__main__":
    main()
