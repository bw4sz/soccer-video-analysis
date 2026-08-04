"""Census the on-ball selection before/after the span-duration and halo fixes.

Both fixes change *what gets cut*, so the reels have to be re-rendered — and a
census is much cheaper than 13 renders for finding out whether the change is the
one intended. Loads the run's artefacts once and walks the roster in memory;
`tracks.json` is 336 MB and re-reading it per player dominates everything else.

Reports per player: spans and touch seconds under the old rule (a span of 2+
samples was exempt from ``--on-ball-min-span``) and under the new one, plus how
many merged clip windows carried anchor lanes that were alive at the same time —
the halo hole, where two lanes of one name both fed the spotlight and the
per-frame dedupe picked between them arbitrarily.

    python slurm/census_span_fixes.py [run_dir] [profile] [kit]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccer_vision.events.on_ball import (  # noqa: E402
    _sample_step_frames,
    select_on_ball_spans,
    spans_to_events,
)
from soccer_vision.identify.resolve import tracks_for  # noqa: E402
from soccer_vision.profiles.loader import load_profile  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1
           else "runs/saints-u14g-full-30fps")
PROFILE = (sys.argv[2] if len(sys.argv) > 2
           else "examples/profiles/saints-u14g.yaml")
KIT = sys.argv[3] if len(sys.argv) > 3 else "black"
PRE, POST, MERGE_GAP, MIN_SPAN = 6.0, 5.0, 2.0, 0.4


def old_rule_spans(ball, tracks, ids, *, min_span_s=MIN_SPAN):
    """Re-implement the pre-fix filter: 2+ samples bypassed the duration floor."""
    spans = select_on_ball_spans(ball, tracks, ids, min_span_s=0.0)
    fps = float(ball.get("fps") or tracks.get("fps") or 30.0)
    return [s for s in spans
            if not (s.end_s - s.start_s < min_span_s and s.n_samples < 2)], fps


def main() -> None:
    ball = json.loads((RUN / "ball_track.json").read_text())
    tracks = json.loads((RUN / "tracks.json").read_text())
    jerseys = json.loads((RUN / "jerseys.json").read_text())
    profile = load_profile(PROFILE)
    fps = float(ball.get("fps") or tracks.get("fps") or 30.0)
    step = _sample_step_frames(ball, fps)
    print(f"run={RUN}  fps={fps:.2f}  sample step={step} frame(s) "
          f"({step / fps:.3f}s per sample)\n")

    lane_frames = {int(t): (int(s[0]["frame"]), int(s[-1]["frame"]))
                   for t, s in tracks.get("tracks", {}).items() if s}

    def overlapping_anchor_windows(events):
        """Merged windows whose anchor lanes are alive simultaneously."""
        from soccer_vision.cli.extract import _merge_windows
        n = 0
        for w in _merge_windows(events, pre_s=PRE, post_s=POST,
                                merge_gap_s=MERGE_GAP):
            ids = [int(t) for t in w["track_ids"] if int(t) in lane_frames]
            if any(lane_frames[a][0] <= lane_frames[b][1]
                   and lane_frames[b][0] <= lane_frames[a][1]
                   for i, a in enumerate(ids) for b in ids[i + 1:]):
                n += 1
        return n

    hdr = (f"{'player':<24}{'old spans':>10}{'new spans':>10}{'old s':>8}"
           f"{'new s':>8}{'dropped':>9}{'overlap win':>12}")
    print(hdr)
    print("-" * len(hdr))

    totals = [0, 0, 0.0, 0.0, 0]
    for entry in profile["roster"]:
        name = entry["name"]
        ids = tracks_for(jerseys, name=name, profile=profile)
        if not ids:
            print(f"{name:<24}{'—':>10}{'—':>10}{'—':>8}{'—':>8}"
                  f"{'—':>9}{'—':>12}   (no lane carries this name)")
            continue
        old, _ = old_rule_spans(ball, tracks, set(ids))
        new = select_on_ball_spans(ball, tracks, set(ids), min_span_s=MIN_SPAN)
        old_s = sum(s.end_s - s.start_s for s in old)
        new_s = sum(s.end_s - s.start_s for s in new)
        ov = overlapping_anchor_windows(
            spans_to_events(old, track_teams=tracks.get("teams"), team=KIT))
        print(f"{name:<24}{len(old):>10}{len(new):>10}{old_s:>8.0f}{new_s:>8.0f}"
              f"{len(old) - len(new):>9}{ov:>12}")
        totals[0] += len(old)
        totals[1] += len(new)
        totals[2] += old_s
        totals[3] += new_s
        totals[4] += ov

    print("-" * len(hdr))
    print(f"{'TOTAL':<24}{totals[0]:>10}{totals[1]:>10}{totals[2]:>8.0f}"
          f"{totals[3]:>8.0f}{totals[0] - totals[1]:>9}{totals[4]:>12}")

    # The team reel needs no identity, so it is the control on the same change.
    teams = tracks.get("teams") or {}
    kit_ids = {int(t) for t in tracks.get("tracks", {}) if teams.get(str(t)) == KIT}
    old, _ = old_rule_spans(ball, tracks, kit_ids)
    new = select_on_ball_spans(ball, tracks, kit_ids, min_span_s=MIN_SPAN)
    print(f"\n{KIT} kit (no identity): {len(old)} -> {len(new)} spans, "
          f"{sum(s.end_s - s.start_s for s in old):.0f}s -> "
          f"{sum(s.end_s - s.start_s for s in new):.0f}s")


if __name__ == "__main__":
    main()
