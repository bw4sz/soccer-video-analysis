"""Can SAM3 track the ball better than RF-DETR?

RF-DETR on this clip (persisted in runs/sam3-ball/ball_track.json):
  212/300 samples "visible" (70.7%) but median jump 117px, p95 1260px,
  max 1650px on a 1920px-wide frame — i.e. mostly false positives, with
  detections in the trees and sky.

The hybrid (RF-DETR ball + SAM3 players) exists only because SAM *v1* couldn't
isolate a ball from automatic masks. SAM3 is text-promptable, and "referee"
returned exactly 1 stable object, so test "soccer ball" head-to-head on the
SAME frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from soccer_vision.tracking.sam3 import SAM3PlayerTracker

CLIP = "/orange/ewhite/b.weinstein/soccer-video-analysis/data/saints_u11_smoke60.mp4"
RF_TRACK = Path("/orange/ewhite/b.weinstein/soccer-video-analysis/runs/sam3-ball/ball_track.json")
OUT = Path("/orange/ewhite/b.weinstein/soccer-video-analysis/runs/sam3-ball/sam3_ball_track.json")
STRIDE = 6          # same 5fps sampling the pipeline uses
PROMPTS = ["soccer ball", "ball"]


def stats(xs, ys, n_total):
    n = len(xs)
    jumps = [float(np.hypot(xs[i+1]-xs[i], ys[i+1]-ys[i])) for i in range(n-1)]
    if not jumps:
        return f"detected {n}/{n_total}, no jumps"
    return (f"detected {n}/{n_total} ({100*n/n_total:.1f}%)  "
            f"jump median {np.median(jumps):.0f}px  p95 {np.percentile(jumps,95):.0f}px  "
            f"max {max(jumps):.0f}px")


def main():
    cap = cv2.VideoCapture(CLIP)
    frames, fns = [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % STRIDE == 0:
            frames.append(f); fns.append(i)
        i += 1
    cap.release()
    print(f"clip: {len(frames)} sampled frames @ stride {STRIDE}")

    rf = json.loads(RF_TRACK.read_text())["samples"]
    rvis = [s for s in rf if s["visible"]]
    print(f"\n[RF-DETR baseline] {stats([s['pixel_x'] for s in rvis], [s['pixel_y'] for s in rvis], len(rf))}")

    for prompt in PROMPTS:
        print(f"\n[SAM3 prompt {prompt!r}]")
        tr = SAM3PlayerTracker(device="cuda", prompt=prompt, min_score=0.3)
        tr.start()
        xs, ys, per_frame, samples = [], [], [], []
        ids = {}
        for fn, f in zip(fns, frames):
            d = tr.track(f)
            per_frame.append(len(d))
            if len(d):
                # ball = highest-confidence detection
                k = int(np.argmax(d.confidence))
                cx = float((d.xyxy[k][0] + d.xyxy[k][2]) / 2)
                cy = float((d.xyxy[k][1] + d.xyxy[k][3]) / 2)
                xs.append(cx); ys.append(cy)
                tid = int(d.tracker_id[k]); ids[tid] = ids.get(tid, 0) + 1
                samples.append({"frame": fn, "visible": True, "pixel_x": cx,
                                "pixel_y": cy, "confidence": float(d.confidence[k]),
                                "track_id": tid})
            else:
                samples.append({"frame": fn, "visible": False, "pixel_x": None,
                                "pixel_y": None, "confidence": 0.0, "track_id": None})
        print("  " + stats(xs, ys, len(frames)))
        print(f"  objects/frame: median {int(np.median(per_frame))} max {max(per_frame)}")
        top = sorted(ids.items(), key=lambda kv: -kv[1])[:3]
        print(f"  distinct ids: {len(ids)}   top ids (id,frames): {top}")
        if prompt == PROMPTS[0]:
            OUT.write_text(json.dumps({"prompt": prompt, "samples": samples}))
            print(f"  wrote {OUT}")
        tr.close()


if __name__ == "__main__":
    main()
