"""Follow one player through a match by motion, and render the halo you get.

This exists because a player reel built from *names* is not a reel of a player.
`identify` names each ByteTrack lane independently, so "Morgan" resolves to
hundreds of lanes — on `runs/saints-u14g-full`, 275 of them, **two to four alive
at once on 18.4% of the frames she is named at**. There is one Morgan on the
pitch, so every one of those frames has another child under her name, and a reel
that unions them halos whoever the gallery guessed. That is the jumping halo, and
no clip-selection change fixes it.

So this tool asks the other question: *pick one lane that is definitely her, and
see how far motion continuity alone carries it.* No ball, no gallery, no OCR —
`soccer_vision.tracking.link` chains lanes end-to-start on extrapolated velocity
and kit, and we render the chain as a single continuous window of the match with
exactly one halo on screen. Whatever the halo does is what the tracking does.

The HUD names the lane currently being followed and marks every handoff, because
a halo that slides onto a different player looks identical to one that stays put
unless you can see the lane id change under it.

Usage:
    # Chain from a known-good lane and report, without rendering:
    python scripts/follow_player.py --run runs/<id> --seed-track 14185 --dry-run

    # Render a continuous window around the seed:
    python scripts/follow_player.py --run runs/<id> --seed-track 14185 \
        --start-s 1040 --dur-s 60 --out morgan_follow.mp4
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np

from soccer_vision.tracking.link import (LinkConfig, _find, link_tracks,
                                         merge_duplicate_lanes)

HALO_COLOUR = (0, 215, 255)      # BGR amber — the one halo on screen
SWITCH_COLOUR = (0, 100, 255)    # redder, for the second after a lane handoff
HUD_BG = (24, 24, 24)


def chain_for(result, seed: str) -> list[str]:
    """Every lane in the seed's connected chain."""
    root = _find(result.parent, seed)
    return [tid for tid in result.parent if _find(result.parent, tid) == root]


def chain_samples(tracks: dict, chain: list[str]) -> list[tuple[int, list, str]]:
    """``(frame, bbox, track_id)`` over the whole chain, sorted, one per frame.

    Lanes in a chain are disjoint in time by construction (linking joins
    end-to-start and never merges overlapping lanes), so a duplicate frame means
    something is wrong upstream; the earlier lane wins and it is counted.
    """
    rows: list[tuple[int, list, str]] = []
    for tid in chain:
        for s in tracks.get(tid, []):
            rows.append((int(s["frame"]), s["bbox"], tid))
    rows.sort(key=lambda r: r[0])
    out, seen = [], set()
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        out.append(r)
    return out


def describe_chain(rows, fps: float, run_frames: int | None = None) -> dict:
    """Smoothness of a followed chain: coverage, handoffs, and the jump at each.

    A "jump" is how far the halo teleports across a lane handoff, in px of foot
    position. A clean continuation is a few px; a hundred px at 30 fps is the
    halo landing on a different person.
    """
    if not rows:
        return {}
    lanes_in_order, jumps, gaps = [], [], []
    prev_f, prev_foot, prev_tid = None, None, None
    for f, bbox, tid in rows:
        foot = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
        if prev_tid is not None and tid != prev_tid:
            jumps.append({
                "at_s": round(f / fps, 2), "from": prev_tid, "to": tid,
                "gap_s": round((f - prev_f) / fps, 2),
                "jump_px": round(float(np.hypot(foot[0] - prev_foot[0],
                                                foot[1] - prev_foot[1])), 1),
            })
        if prev_f is not None and f - prev_f > 1:
            gaps.append((f - prev_f) / fps)
        if not lanes_in_order or lanes_in_order[-1] != tid:
            lanes_in_order.append(tid)
        prev_f, prev_foot, prev_tid = f, foot, tid

    span_s = (rows[-1][0] - rows[0][0]) / fps
    return {
        "lanes": len(set(t for _, _, t in rows)),
        "handoffs": len(jumps),
        "first_s": round(rows[0][0] / fps, 1),
        "last_s": round(rows[-1][0] / fps, 1),
        "span_s": round(span_s, 1),
        "frames": len(rows),
        "fill": round(len(rows) / (span_s * fps + 1), 3) if span_s > 0 else 1.0,
        "gap_s_total": round(sum(gaps), 1),
        "gap_s_max": round(max(gaps), 2) if gaps else 0.0,
        "jump_px_median": round(float(np.median([j["jump_px"] for j in jumps])), 1) if jumps else 0.0,
        "jump_px_max": round(max(j["jump_px"] for j in jumps), 1) if jumps else 0.0,
        "big_jumps": [j for j in jumps if j["jump_px"] > 60],
    }


def box_at(rows, frame: int, hold_frames: int) -> tuple[list, str] | None:
    """Chain box for ``frame``: the most recent sample within ``hold_frames``.

    Held, not interpolated. Interpolating across a dropout is what draws a halo
    on empty grass — the player left, the box coasted, and the viewer sees a ring
    round nobody. Holding for a beat and then dropping tells the truth.
    """
    frames = rows["frames"]
    i = bisect_right(frames, frame) - 1
    if i < 0:
        return None
    if frame - frames[i] > hold_frames:
        return None
    return rows["boxes"][i], rows["tids"][i]


def render(video: Path, out: Path, rows: list, *, fps: float, start_s: float,
           dur_s: float, hold_frames: int, seed: str) -> Path:
    idx = {"frames": [r[0] for r in rows], "boxes": [r[1] for r in rows],
           "tids": [r[2] for r in rows]}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0 = int(round(start_s * vfps))
    n = int(round(dur_s * vfps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), vfps, (w, h))
    last_tid, switch_at, n_drawn = None, -999, 0
    try:
        for k in range(n):
            ok, img = cap.read()
            if not ok:
                break
            fno = f0 + k
            hit = box_at(idx, fno, hold_frames)
            label = "LOST"
            if hit is not None:
                bbox, tid = hit
                if last_tid is not None and tid != last_tid:
                    switch_at = k
                last_tid = tid
                fresh = (k - switch_at) < int(vfps)   # flag the handoff for a second
                colour = SWITCH_COLOUR if fresh else HALO_COLOUR
                cx, cy = (bbox[0] + bbox[2]) / 2.0, bbox[3]
                rx = max(12.0, (bbox[2] - bbox[0]) * 0.6)
                cv2.ellipse(img, (int(cx), int(cy)), (int(rx), int(rx * 0.35)),
                            0, -30, 210, colour, 3, cv2.LINE_AA)
                label = f"lane {tid}" + ("  <- HANDOFF" if fresh else "")
                n_drawn += 1
            cv2.rectangle(img, (10, 10), (620, 96), HUD_BG, -1)
            cv2.putText(img, f"t={fno / vfps:7.1f}s   seed lane {seed}", (22, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
            cv2.putText(img, label, (22, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        HALO_COLOUR if hit else (120, 120, 200), 2, cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()
        cap.release()
    print(f"  rendered {out}  ({n_drawn}/{n} frames haloed)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--seed-track", required=True, help="lane id known to be the player")
    p.add_argument("--tracks", default="tracks.json")
    p.add_argument("--max-gap-s", type=float, default=2.0)
    p.add_argument("--max-dist-px", type=float, default=150.0)
    p.add_argument("--no-kit", action="store_true", help="drop the kit-agreement gate")
    p.add_argument("--no-dedup", action="store_true",
                   help="skip the duplicate-id pre-pass (see merge_duplicate_lanes)")
    p.add_argument("--start-s", type=float)
    p.add_argument("--dur-s", type=float, default=60.0)
    p.add_argument("--hold-s", type=float, default=0.25,
                   help="how long a box is held after the last sample before the halo drops")
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    doc = json.loads((args.run / args.tracks).read_text())
    fps = float(doc["fps"])
    seed = args.seed_track

    if not args.no_dedup:
        doc, dstats = merge_duplicate_lanes(doc)
        print(f"  dedup: {dstats['lanes_before']} -> {dstats['lanes_after']} lanes "
              f"({dstats['merges']} duplicate-id handoffs merged)")
        seed = dstats["alias"].get(seed, seed)
        if seed != args.seed_track:
            print(f"  seed lane {args.seed_track} now carried by lane {seed}")

    cfg = LinkConfig(max_gap_s=args.max_gap_s, max_dist_px=args.max_dist_px,
                     require_kit=not args.no_kit)
    print(f"linking {len(doc['tracks'])} lanes (gap<={cfg.max_gap_s}s, "
          f"dist<={cfg.max_dist_px}px, kit={'on' if cfg.require_kit else 'off'})...")
    result = link_tracks(doc, cfg)
    print(f"  {result.stats}")

    if seed not in result.parent:
        raise SystemExit(f"seed lane {seed} was merged away by dedup; re-run with "
                         f"--no-dedup to find the lane it belongs to")
    chain = chain_for(result, seed)
    rows = chain_samples(doc["tracks"], chain)
    stats = describe_chain(rows, fps)
    print(f"\nchain containing lane {args.seed_track}:")
    for k, v in stats.items():
        if k != "big_jumps":
            print(f"  {k:16} {v}")
    for j in stats.get("big_jumps", [])[:12]:
        print(f"  JUMP {j['jump_px']:6.0f}px at {j['at_s']}s  {j['from']} -> {j['to']} "
              f"(gap {j['gap_s']}s)")

    if args.dry_run:
        return
    start = args.start_s if args.start_s is not None else stats["first_s"]
    out = args.out or (args.run / f"follow_{args.seed_track}_{int(start)}s.mp4")
    render(args.run / "broadcast_proxy.mp4", out, rows, fps=fps, start_s=start,
           dur_s=args.dur_s, hold_frames=int(round(args.hold_s * fps)),
           seed=args.seed_track)


if __name__ == "__main__":
    main()
