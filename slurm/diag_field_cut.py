"""How many real players did the centred rectangle throw away?

`filter_spectators` used to keep the middle 70% of width *and* height. On our
cameras that shape is wrong: they stand at the touchline, so the near half of the
pitch runs off the bottom edge and a wide frame is one pitch across — only the
top of the frame holds other people's matches. Job 38180242 measured the old cut
discarding 36% of detected people while *keeping* actual spectators on the far
touchline, and every track in runs/saints-u14g-full is clipped into
x in [288, 1632], y <= 918.

This runs the two cuts side by side on frames spread across the whole match (not
one 3-minute smoke clip) and reports what the new one recovers and where it sits
in frame, so the change is measured rather than argued.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/orange/ewhite/b.weinstein/soccer-video-analysis/src")

VIDEO = ("/orange/ewhite/b.weinstein/soccer-video-analysis/data/"
         "wfc-rangers-vs-saints-pcu-cup-2026-07-11.mp4")
N_FRAMES = 120


def main():
    from soccer_vision.detection.field_filter import filter_spectators
    from soccer_vision.detection.rfdetr import ALL_PERSON_CLASS_IDS, RFDETRSoccerDetector

    det = RFDETRSoccerDetector.from_pretrained(device="cuda")
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = total // (N_FRAMES + 1)

    people, old, new = [], [], []
    recovered_feet = []
    for i in range(N_FRAMES):
        fn = step * (i + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = det.predict(frame, conf_threshold=0.3)
        person = dets[np.isin(dets.class_id, list(ALL_PERSON_CLASS_IDS))]
        people.append(len(person))
        # The old centred rectangle, expressed in the new parameters.
        kept_old = filter_spectators(person, frame.shape,
                                     top_frac=0.15, side_frac=0.15, bottom_frac=0.15)
        kept_new = filter_spectators(person, frame.shape)
        old.append(len(kept_old))
        new.append(len(kept_new))

        if len(kept_new) > len(kept_old):
            feet = np.column_stack([
                (kept_new.xyxy[:, 0] + kept_new.xyxy[:, 2]) / 2, kept_new.xyxy[:, 3]])
            gained = (feet[:, 0] < 0.15 * w) | (feet[:, 0] > 0.85 * w) | (feet[:, 1] > 0.85 * h)
            recovered_feet.extend(feet[gained].tolist())

    cap.release()

    people, old, new = (np.array(x, dtype=float) for x in (people, old, new))
    print(f"video : {VIDEO}")
    print(f"frames: {len(people)} spread across the match\n")
    print(f"{'stage':<38s} {'median':>7s} {'mean':>7s} {'min':>5s} {'max':>5s}")
    for label, v in [("RF-DETR person detections", people),
                     ("kept — old centred rectangle", old),
                     ("kept — top-only cut (new default)", new)]:
        print(f"{label:<38s} {np.median(v):7.1f} {v.mean():7.1f} {v.min():5.0f} {v.max():5.0f}")

    drop_old = 100 * (people - old).sum() / max(people.sum(), 1)
    drop_new = 100 * (people - new).sum() / max(people.sum(), 1)
    gain = 100 * (new - old).sum() / max(old.sum(), 1)
    print(f"\nold cut discarded {drop_old:.0f}% of detected people")
    print(f"new cut discards  {drop_new:.0f}%")
    print(f"players recovered: +{gain:.0f}% ({(new - old).sum():.0f} boxes over "
          f"{len(people)} frames)")

    if recovered_feet:
        feet = np.array(recovered_feet)
        print(f"\nwhere the recovered {len(feet)} boxes stand (foot point):")
        print(f"  left edge   (x < {0.15 * w:.0f}): {(feet[:, 0] < 0.15 * w).sum()}")
        print(f"  right edge  (x > {0.85 * w:.0f}): {(feet[:, 0] > 0.85 * w).sum()}")
        print(f"  bottom band (y > {0.85 * h:.0f}): {(feet[:, 1] > 0.85 * h).sum()}")


if __name__ == "__main__":
    main()
