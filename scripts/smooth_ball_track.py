"""Re-smooth an already-processed run's ball_track.json in place.

The flicker gate is a pure post-process over saved detections, so an existing
run can be fixed in seconds rather than by re-running ~2 h of detection. The
original file is copied to `ball_track.raw.json` before anything is written,
and every rejected detection is preserved inside the new file under
`raw_pixel_x` / `raw_pixel_y` as well -- so this is reversible twice over.

    python scripts/smooth_ball_track.py runs/saints-u14g-full-30fps
    python scripts/smooth_ball_track.py runs/*/ --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from soccer_vision.tracking.ball_smooth import (  # noqa: E402
    MAX_DEV_PX, MAX_FILL_S, WINDOW_S, smooth_samples,
)

PHYS_PX_PER_FRAME = 70.0   # 2x a 30 m/s ball; above this it is not the ball


def step_stats(samples: list[dict], fps: float) -> tuple[float, float, float]:
    """(p95 step, % unphysical, coverage) over adjacent-frame pairs."""
    pts = [(s["timestamp_s"], s["pixel_x"], s["pixel_y"]) for s in samples
           if s.get("visible") and s.get("pixel_x") is not None]
    cov = len(pts) / max(1, len(samples))
    if len(pts) < 3:
        return float("nan"), float("nan"), cov
    a = np.array(pts, dtype=float)
    d = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2]))
    adjacent = np.diff(a[:, 0]) < 1.5 / max(fps, 1e-6)
    d = d[adjacent]
    if not len(d):
        return float("nan"), float("nan"), cov
    return float(np.percentile(d, 95)), float((d > PHYS_PX_PER_FRAME).mean() * 100), cov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--window-s", type=float, default=WINDOW_S)
    ap.add_argument("--max-dev-px", type=float, default=MAX_DEV_PX)
    ap.add_argument("--max-fill-s", type=float, default=MAX_FILL_S)
    ap.add_argument("--force", action="store_true",
                    help="re-smooth a track already marked smoothed")
    args = ap.parse_args()

    print(f"{'run':32s} {'p95 step':>18s} {'unphysical':>18s} {'coverage':>16s}")
    print("-" * 88)
    for run in args.runs:
        path = run / "ball_track.json"
        if not path.exists():
            print(f"{run.name:32s}  no ball_track.json — skipped")
            continue

        track = json.loads(path.read_text())
        samples = track.get("samples", [])
        fps = float(track.get("sample_fps") or track.get("fps") or 30.0)

        if track.get("smoothed") and not args.force:
            print(f"{run.name:32s}  already smoothed — skipped (--force to redo)")
            continue

        before = step_stats(samples, fps)
        out = smooth_samples(samples, window_s=args.window_s,
                             max_dev_px=args.max_dev_px, max_fill_s=args.max_fill_s)
        after = step_stats(out, fps)

        print(f"{run.name:32s} {before[0]:8.1f} ->{after[0]:7.1f} "
              f"{before[1]:9.1f}% ->{after[1]:6.1f}% "
              f"{before[2]:7.1%} ->{after[2]:6.1%}")

        if args.dry_run:
            continue
        backup = run / "ball_track.raw.json"
        if not backup.exists():
            shutil.copy2(path, backup)
        track["samples"] = out
        track["smoothed"] = True
        path.write_text(json.dumps(track))

    if args.dry_run:
        print("\n(dry run — nothing written)")


if __name__ == "__main__":
    main()
