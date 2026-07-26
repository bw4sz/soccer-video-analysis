"""Where do SAM3's ~20 detections/frame go? Measure raw vs post-filter counts,
and whether the homography branch is the one doing the rejecting.

Hypothesis: compute_homography returns ok=True with a garbage H on this
overhead footage (Hough is known to fail here), so filter_by_homography
projects players off-field and drops them. The smoke run kept only 52
track-frames out of ~6000.
"""
from __future__ import annotations

import cv2
import numpy as np

from soccer_vision.detection.field_filter import (
    filter_by_field_hull, filter_spectators,
)
from soccer_vision.registration.hough import compute_homography
from soccer_vision.tracking.sam3 import SAM3PlayerTracker

VIDEO = "/orange/ewhite/b.weinstein/soccer-video-analysis/data/saints_u11_smoke60.mp4"
STRIDE, N = 6, 30


def main():
    cap = cv2.VideoCapture(VIDEO)
    tracker = SAM3PlayerTracker(device="cuda")
    tracker.start()

    H_cache = None
    print(f"{'i':>4}{'raw':>6}{'hull':>6}{'filtered':>10}{'H?':>5}")
    for i in range(N):
        for _ in range(STRIDE):
            ok, bgr = cap.read()
        if not ok:
            break
        dets = tracker.track(bgr)
        raw = len(dets)
        hull = len(filter_by_field_hull(dets, bgr.shape))

        # mimic process.py: homography recomputed periodically
        if i == 5:
            H_new, okh = compute_homography(bgr)
            print(f"  [compute_homography] ok={okh}  H is None: {H_new is None}")
            if okh:
                H_cache = H_new
                print(f"  H=\n{np.array2string(H_new, precision=3)}")

        filt = len(filter_spectators(dets, H_cache, bgr.shape))
        print(f"{i:>4}{raw:>6}{hull:>6}{filt:>10}{'Y' if H_cache is not None else 'n':>5}")

    cap.release()
    print("\nIf 'filtered' collapses to ~0 exactly when H? flips to Y,")
    print("the broken homography is the cause, not SAM3 detection.")


if __name__ == "__main__":
    main()
