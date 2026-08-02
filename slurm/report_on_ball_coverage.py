"""What does the pipeline see of our team's touches, and how much of it is named?

`CLAUDE.md`'s *Identity coverage* section is the number that should govern where
effort goes: on the 5 fps run, 1,065 s of Saints on-ball time was detected,
tracked and turned into clean spans, and **9.4% of it carried a name**. The other
91% was thrown away for want of a label, not for want of football.

This recomputes that from saved artefacts only — no GPU, no model — so it can be
re-run after any change to the ball track, the tracker, the linker or naming, and
splits it three ways that each answer a different question:

- **detected** — spans over *every* lane of our kit. The ceiling: touches the
  pipeline saw at all.
- **named** — spans over lanes carrying any roster name. What a `--team` reel
  could cut.
- **one player** — spans for a single `--player`. What her reel actually cuts,
  and the number that has been disappointing.

``--compare-ball`` reruns the whole thing against a second ball track (normally
``ball_track.raw.json``, kept by `scripts/smooth_saved_ball_track.py`) so the
effect of gating the ball flicker on *segment selection* is visible, rather than
only on the ball-jump statistics that motivated the gate. A false ball position
does two things here and they pull opposite ways: it invents proximity where
there was none, and it destroys proximity that was real.

Usage:
    python slurm/report_on_ball_coverage.py --run /path/to/runs/<match_id> \
        --team black --player "Morgan Lobey" --compare-ball ball_track.raw.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soccer_vision.events.on_ball import select_on_ball_spans


def _summarise(spans) -> tuple[int, float]:
    return len(spans), sum(s.end_s - s.start_s for s in spans)


def _lane_sets(tracks: dict, jerseys: dict | None, kit: str | None,
               player: str | None) -> dict[str, set[int]]:
    teams = tracks.get("teams") or {}
    all_ids = {int(t) for t in tracks["tracks"]}
    ours = {int(t) for t in tracks["tracks"]
            if kit is None or teams.get(str(t)) == kit}

    sets = {"detected (our kit)": ours}
    if jerseys:
        jt = jerseys.get("tracks", {})
        named = {int(t) for t in ours if jt.get(str(t), {}).get("name")}
        sets["named (any roster name)"] = named
        if player:
            key = player.strip().lower()
            sets[f"one player ({player})"] = {
                int(t) for t in ours
                if (jt.get(str(t), {}).get("name") or "").strip().lower() == key
            }
    assert ours <= all_ids
    return sets


def report(run: Path, tracks_name: str, jerseys_name: str, ball_name: str,
           kit: str | None, player: str | None, dist: float, min_span: float,
           label: str) -> dict[str, tuple[int, float]]:
    tracks = json.loads((run / tracks_name).read_text())
    ball = json.loads((run / ball_name).read_text())
    jp = run / jerseys_name
    jerseys = json.loads(jp.read_text()) if jp.exists() else None

    vis = sum(1 for s in ball["samples"] if s.get("visible"))
    print(f"\n--- {label}: {ball_name} "
          f"({vis}/{len(ball['samples'])} visible, "
          f"{'gated' if ball.get('smoothed') else 'RAW'}) ---")
    print(f"{'selection':<32} {'spans':>7} {'touch_s':>9} {'share':>7} "
          f"{'reel_min':>9} {'signal':>7}")

    out = {}
    base = None
    for name, ids in _lane_sets(tracks, jerseys, kit, player).items():
        spans = select_on_ball_spans(ball, tracks, ids,
                                     max_ball_dist_px=dist, min_span_s=min_span)
        n, secs = _summarise(spans)
        out[name] = (n, secs)
        if base is None:
            base = secs or 1.0
        reel_s = _reel_length(spans, pre_s=6.0, post_s=5.0, merge_gap_s=2.0)
        print(f"{name:<32} {n:7d} {secs:9.0f} {100 * secs / base:6.1f}% "
              f"{reel_s / 60:9.1f} {100 * secs / reel_s if reel_s else 0:6.1f}%")
    return out


def _reel_length(spans, *, pre_s: float, post_s: float, merge_gap_s: float) -> float:
    """Minutes of footage `reel` would emit for these spans, after merging.

    Worth printing next to the touch seconds because **the two are barely
    related**. `reel` pads every span by 6 s before and 5 s after and merges
    windows within 2 s, so a run of brief touches becomes minutes of footage: on
    the 5 fps run Morgan's 58 spans hold **41 s** of her actually being on the
    ball and produced an **8.7-minute** reel. The complaint that a reel "jumps to
    8 minutes" is a padding policy, not a detection result, and shortening it is
    a different problem from finding the right moments.
    """
    windows = []
    for s in sorted(spans, key=lambda s: s.start_s):
        a, b = max(0.0, s.start_s - pre_s), s.end_s + post_s
        if windows and a - windows[-1][1] <= merge_gap_s:
            windows[-1][1] = max(windows[-1][1], b)
        else:
            windows.append([a, b])
    return sum(b - a for a, b in windows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--tracks", default="tracks.json")
    ap.add_argument("--jerseys", default="jerseys.json")
    ap.add_argument("--ball", default="ball_track.json")
    ap.add_argument("--compare-ball", default=None,
                    help="second ball track to rerun against, e.g. ball_track.raw.json")
    ap.add_argument("--team", default=None)
    ap.add_argument("--player", default=None)
    ap.add_argument("--on-ball-dist", type=float, default=90.0)
    ap.add_argument("--on-ball-min-span", type=float, default=0.4)
    args = ap.parse_args()

    a = report(args.run, args.tracks, args.jerseys, args.ball, args.team,
               args.player, args.on_ball_dist, args.on_ball_min_span, "current")
    if args.compare_ball and (args.run / args.compare_ball).exists():
        b = report(args.run, args.tracks, args.jerseys, args.compare_ball,
                   args.team, args.player, args.on_ball_dist,
                   args.on_ball_min_span, "comparison")
        print(f"\n{'selection':<32} {'gated s':>9} {'raw s':>9} {'delta':>9}")
        for k in a:
            if k in b:
                print(f"{k:<32} {a[k][1]:9.0f} {b[k][1]:9.0f} "
                      f"{a[k][1] - b[k][1]:+9.0f}")
        print("\nA raw ball invents proximity where there was none and destroys "
              "proximity that was real; the net is not the interesting part, "
              "which clips changed is.")


if __name__ == "__main__":
    main()
