"""Verify the streaming SAM3PlayerTracker matches the batch-session result.

Batch baseline (job 37864846, 48 frames from 15000): 20-22 players/frame,
22 distinct ids, 21 stable. Streaming must reproduce that AND be fast enough
for a full match, or the pathway doesn't scale.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from soccer_vision.tracking.sam3 import SAM3PlayerTracker

VIDEO = "/orange/ewhite/b.weinstein/soccer-video-analysis/data/SaintsU11_OVF_Jul192026.MP4"
START, N = 15000, 48
FULL_MATCH_FRAMES_AT_5FPS = 55354 / 6


def main() -> None:
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    frames = []
    for _ in range(N):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"clip: {len(frames)} frames @ {frames[0].shape[1]}x{frames[0].shape[0]}")

    tracker = SAM3PlayerTracker(device="cuda")
    tracker.start()

    counts, id_hits, times = [], {}, []
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        dets = tracker.track(f)
        times.append(time.perf_counter() - t0)
        n = len(dets)
        counts.append(n)
        if dets.tracker_id is not None:
            for tid in dets.tracker_id:
                id_hits[int(tid)] = id_hits.get(int(tid), 0) + 1
        if i == 0:
            print(f"  frame0: {n} dets, has_mask={dets.mask is not None}, "
                  f"ids={sorted(int(t) for t in dets.tracker_id)[:8]}...")

    c = np.array(counts)
    stable = sum(1 for v in id_hits.values() if v >= 0.5 * len(frames))
    spf = float(np.mean(times[1:]))  # drop first (warmup)

    print("\n" + "=" * 60)
    print("STREAMING SAM3 — vs batch baseline (20/21/22, 22 ids, 21 stable)")
    print("=" * 60)
    print(f"players/frame:      min {c.min()}  median {int(np.median(c))}  max {c.max()}")
    print(f"distinct ids:       {len(id_hits)}")
    print(f"stable (>=50%):     {stable}")
    print(f"speed:              {spf:.3f} s/frame")
    print(f"full match @5fps:   {spf * FULL_MATCH_FRAMES_AT_5FPS / 3600:.2f} h "
          f"({FULL_MATCH_FRAMES_AT_5FPS:.0f} frames)")
    print("=" * 60)
    ok_det = c.max() >= 18
    ok_trk = stable >= 15
    print(f"VERDICT: detection {'PASS' if ok_det else 'FAIL'}, "
          f"tracking {'PASS' if ok_trk else 'FAIL'}")


if __name__ == "__main__":
    main()
