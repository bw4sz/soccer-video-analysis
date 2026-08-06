#!/usr/bin/env python
"""How big would this tracklet project be? Answer before rendering an hour of clips.

`enroll --dump-tracklets` prints its task count, but only after it has committed
to the run — and rendering 51 windows of 1080p takes about two hours. Annotation
scope is a person's afternoon, so it deserves a number first: how many clips to
watch, how many decisions to make, and how they split by kit.

Reads saved artefacts only — no video, no GPU, seconds to run.

    python slurm/size_tracklet_project.py --run runs/saints-u14g-full-30fps \\
        --at 2400 --window 20 --n-windows 9 --all-lanes --heldout-mode only

    # Sweep the knob that actually moves the cost:
    python slurm/size_tracklet_project.py --run runs/<match> --at 945 \\
        --n-windows 45 --all-lanes --sweep-min-lane-seconds
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soccer_vision import heldout as H  # noqa: E402
from soccer_vision.annotate.tracklets import choose_windows  # noqa: E402
from soccer_vision.clips.halo import load_track_boxes  # noqa: E402


def load_run(run: Path):
    """``(fps, teams, samples)``. Read once — ``tracks.json`` on a 30 fps match is
    3.1M samples, and the sweep below would otherwise re-parse it per row."""
    doc = json.loads((run / "tracks.json").read_text())
    fps = float(doc.get("fps") or 30.0)
    teams = {int(k): v for k, v in (doc.get("teams") or {}).items()}
    return fps, teams, load_track_boxes(run / "tracks.json")


def size(run: Path, loaded, *, at, window, n_windows, max_lanes, all_lanes, team,
         min_track_frames, mode, registry) -> dict:
    fps, teams, samples = loaded

    windows = choose_windows(
        samples, fps=fps, window_s=window, n_windows=n_windows,
        min_track_frames=min_track_frames, max_lanes=max_lanes,
        teams=teams, team=team, start_s=at, all_lanes=all_lanes,
    )
    windows, _ = H.filter_frames(
        windows, mode, key=run, registry=registry,
        frame_of=lambda w: (w["start_frame"] + w["end_frame"]) // 2)

    lanes = [lane for w in windows for lane in w["lanes"]]
    kit = Counter(teams.get(lane["track_id"], "unclassified") for lane in lanes)
    stretches = {w["start_frame"] for w in windows}
    return {
        "fps": fps,
        "tasks": len(windows),
        "stretches": len(stretches),
        "lanes": len(lanes),
        "kit": kit,
        "watch_min": len(windows) * window / 60.0,
        "play_min": len(stretches) * window / 60.0,
        "max_pages": max((w.get("n_pages", 1) for w in windows), default=0),
    }


def report(label: str, s: dict) -> None:
    kit = "  ".join(f"{k}:{n}" for k, n in s["kit"].most_common())
    print(f"{label:<26} {s['tasks']:>4} tasks  {s['lanes']:>5} decisions  "
          f"{s['watch_min']:>5.0f} min of clips  ({s['play_min']:.0f} min of play, "
          f"up to {s['max_pages']} passes)   {kit}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--at", type=float, default=None)
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--n-windows", type=int, default=9)
    ap.add_argument("--max-lanes", type=int, default=16)
    ap.add_argument("--all-lanes", action="store_true")
    ap.add_argument("--team", default=None)
    ap.add_argument("--min-lane-seconds", type=float, default=1.0,
                    help="Lane floor, in seconds — the same quantity "
                         "`identify --min-lane-seconds` gates on (default: 1.0)")
    ap.add_argument("--heldout-mode", choices=H.MODES, default=H.EXCLUDE)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--sweep-min-lane-seconds", action="store_true",
                    help="Show the cost curve — this is the knob that moves it")
    args = ap.parse_args()

    registry = H.load_registry(args.heldout)
    print(registry.describe_coverage(args.run))
    loaded = load_run(args.run)
    fps = loaded[0]

    def run_at(min_lane_s):
        return size(args.run, loaded,
                    at=args.at, window=args.window, n_windows=args.n_windows,
                    max_lanes=args.max_lanes, all_lanes=args.all_lanes, team=args.team,
                    min_track_frames=max(1, int(round(min_lane_s * fps))),
                    mode=args.heldout_mode, registry=registry)

    floors = [0.5, 1.0, 2.0, 3.0, 5.0] if args.sweep_min_lane_seconds else [args.min_lane_seconds]
    print()
    for floor in floors:
        report(f"lanes >= {floor:.1f}s", run_at(floor))
    print("\nA decision is one dropdown. A task is one 20 s clip to watch — under "
          "--all-lanes the same footage repeats once per page, which is why "
          "'min of clips' exceeds 'min of play'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
