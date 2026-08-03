"""What does refusing to name short lanes cost, and what does it buy?

Re-id names a one-frame lane as readily as a one-minute one — ``embed_tracks``
samples whatever crops exist and ``match_track`` returns the nearest of our
eleven either way. On ``runs/saints-u14g-full-30fps`` that is not a rare edge
case: 81% of its naming decisions were made on lanes shorter than a second, and
``propagate_names`` then seeds whole chains from them.

This simulates the gate at several thresholds **from saved artefacts only** — no
GPU, no model, no re-detection — in the order the pipeline actually runs it:
gate the pre-link names by the evidence lane's own length, re-run name
propagation over the saved link result, then recount what a reel could cut.

Three columns matter and they pull against each other:

- **named on-ball** — what a `--team` reel could cut. The cost side.
- **one player** — what her reel actually cuts. The number that matters.
- **collide%** — frames where one player's name is alive on two lanes at once.
  At least one of those is wrong, so this is a *lower bound* on naming error and
  the only wrongness proxy available without hand labels. It is not accuracy.

Needs a run that has been identified *and* linked, so ``jerseys.prelink.json``,
``tracks.unlinked.json`` and ``track_links.json`` all sit beside the linked pair.

Usage:
    python slurm/eval_min_lane_length.py --run runs/saints-u14g-full-30fps \
        --team black --player "Morgan Lobey"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from soccer_vision.events.on_ball import select_on_ball_spans
from soccer_vision.tracking.link import Link, LinkResult, propagate_names


def _spans_s(tracks: dict, fps: float) -> dict[str, float]:
    return {t: (max(x["frame"] for x in s) - min(x["frame"] for x in s)) / fps
            for t, s in tracks["tracks"].items() if s}


def _rebuild_result(unlinked: dict, links_doc: dict) -> LinkResult:
    """Reconstruct the LinkResult from ``track_links.json`` (it stores the edges)."""
    parent = {t: t for t in unlinked["tracks"]}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    links = []
    for e in links_doc["links"]:
        a, b = str(e["a"]), str(e["b"])
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
        links.append(Link(a, b, e.get("gap_s", 0.0), e.get("dist_px", 0.0),
                          e.get("naive_dist_px", 0.0), e.get("speed_px_s", 0.0)))
    return LinkResult(parent=parent, links=links)


def _collision_rate(named: set[int], tracks: dict, jt: dict) -> float:
    """Share of tracked frames where one name is alive on two lanes at once."""
    frames: dict[int, Counter] = {}
    for t in named:
        name = jt[str(t)]["name"]
        for s in tracks["tracks"][str(t)]:
            frames.setdefault(s["frame"], Counter())[name] += 1
    if not frames:
        return 0.0
    return sum(1 for c in frames.values() if max(c.values()) > 1) / len(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--team", default="black", help="our kit colour this match")
    ap.add_argument("--player", default=None, help="one player to report separately")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    args = ap.parse_args()

    run = Path(args.run)
    linked = json.loads((run / "tracks.json").read_text())
    unlinked = json.loads((run / "tracks.unlinked.json").read_text())
    ball = json.loads((run / "ball_track.json").read_text())
    pre = json.loads((run / "jerseys.prelink.json").read_text())
    result = _rebuild_result(unlinked, json.loads((run / "track_links.json").read_text()))

    span_u = _spans_s(unlinked, unlinked["fps"])
    teams = linked.get("teams") or {}
    ours = {int(t) for t in linked["tracks"] if teams.get(str(t)) == args.team}
    ceiling = select_on_ball_spans(ball, linked, ours)
    who = (args.player or "").strip().lower()

    print(f"== {run}")
    print(f"ceiling ({args.team} kit, no identity): {len(ceiling)} spans, "
          f"{sum(s.end_s - s.start_s for s in ceiling):.0f}s")
    head = (f"{'gate':>6} {'seeds':>6} {'named lanes':>11} {'named on-ball':>13} "
            f"{'sec':>6} {'collide%':>9}")
    if who:
        head += f" | {args.player} spans / sec"
    print(head)

    for thresh in args.thresholds:
        gated = {"tracks": {t: dict(r) for t, r in pre["tracks"].items()}}
        seeds = 0
        for tid, rec in gated["tracks"].items():
            if not rec.get("name"):
                continue
            if span_u.get(tid, 0.0) < thresh:
                rec.update(name=None, jersey=None, source=None, similarity=None)
            else:
                seeds += 1

        doc, _ = propagate_names(gated, result)
        jt = doc["tracks"]
        named, mine = set(), set()
        for tid in linked["tracks"]:          # a chain's root id is its linked lane id
            if int(tid) not in ours:
                continue
            name = (jt.get(tid) or {}).get("name")
            if not name:
                continue
            named.add(int(tid))
            if who and name.strip().lower() == who:
                mine.add(int(tid))

        on_ball = select_on_ball_spans(ball, linked, named) if named else []
        row = (f"{thresh:5.1f}s {seeds:6d} {len(named):11d} {len(on_ball):13d} "
               f"{sum(s.end_s - s.start_s for s in on_ball):6.0f} "
               f"{_collision_rate(named, linked, jt) * 100:8.1f}%")
        if who:
            hers = select_on_ball_spans(ball, linked, mine) if mine else []
            row += f" | {len(hers):3d} / {sum(s.end_s - s.start_s for s in hers):.0f}s"
        print(row)


if __name__ == "__main__":
    main()
