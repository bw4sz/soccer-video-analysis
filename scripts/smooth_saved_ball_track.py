"""Apply the ball median gate to a `ball_track.json` that `process` wrote raw.

`process` gates the ball flicker before writing (`cli/process.py`), so a fresh
run needs nothing. Runs made **before that landed** carry the raw detector
output, and there is no reason to re-`process` a 60-minute match — a ~2 hour GPU
job — to get a filter that runs on the saved file in seconds.

The original is kept as `ball_track.raw.json` and never modified. Rejected
detections survive in-place under `raw_pixel_x` / `raw_pixel_y`, so nothing the
detector reported is lost.

Usage:
    python scripts/smooth_saved_ball_track.py --run /path/to/runs/<match_id>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from soccer_vision.tracking.ball_smooth import smooth_ball_track


def _steps(samples: list[dict]) -> np.ndarray:
    """Frame-to-frame pixel steps over consecutive visible samples."""
    pts = [(s["frame"], s["pixel_x"], s["pixel_y"]) for s in samples if s.get("visible")]
    out = []
    for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
        if f1 - f0 <= 0:
            continue
        out.append(float(np.hypot(x1 - x0, y1 - y0)) / (f1 - f0))
    return np.asarray(out) if out else np.zeros(1)


def describe(samples: list[dict], label: str) -> None:
    vis = sum(1 for s in samples if s.get("visible"))
    st = _steps(samples)
    print(f"  {label:<10} visible {vis:6d}/{len(samples)} ({100 * vis / len(samples):5.1f}%)  "
          f"step median {np.median(st):6.1f} px  p95 {np.percentile(st, 95):7.1f} px  "
          f">70 px/frame {100 * (st > 70).mean():5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="re-smooth even if the track is already marked smoothed")
    args = ap.parse_args()

    path = args.run / "ball_track.json"
    track = json.loads(path.read_text())
    if track.get("smoothed") and not args.force:
        print(f"{path} is already marked smoothed — nothing to do (--force to redo).")
        return

    print(f"{path}  ({len(track['samples'])} samples at "
          f"{track.get('sample_fps', 0):.1f} fps)")
    describe(track["samples"], "raw")

    backup = args.run / "ball_track.raw.json"
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  original kept: {backup}")

    out = smooth_ball_track(track)
    describe(out["samples"], "smoothed")
    n_out = sum(1 for s in out["samples"] if s.get("outlier"))
    print(f"  {n_out} detections gated as outliers (kept under raw_pixel_x/y)")
    path.write_text(json.dumps(out))
    print(f"  wrote: {path}")


if __name__ == "__main__":
    main()
