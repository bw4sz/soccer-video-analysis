"""Where do RF-DETR's players go between detection and tracks.json?

The preview of runs/u14g-smoke-rfdetr (scripts/preview_run.py) shows 6-13 boxes
per frame where the eye counts ~12-20 players, plus a ball latched onto
background objects and almost every track stamped "black". Speed profiling on
this same video saw RF-DETR return 22-27 raw detections/frame, so the players are
being *detected* and lost somewhere downstream.

This walks the exact pipeline stages process.py applies, on the same frames, and
counts survivors at each one:

    raw -> class filter (player+GK) -> filter_spectators -> ByteTrack

so the loss can be attributed instead of guessed at. Also reports the ball's
class-0 detections and the kit colour sampled per box, to check the two other
things the preview flagged.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/orange/ewhite/b.weinstein/soccer-video-analysis/src")

VIDEO = "/orange/ewhite/b.weinstein/soccer-video-analysis/data/u14g_smoke180.mp4"
RUN = Path("/orange/ewhite/b.weinstein/soccer-video-analysis/runs/u14g-smoke-rfdetr")
START, N, INTERVAL = 300, 40, 6


def main():
    from soccer_vision.detection.field_filter import filter_spectators
    from soccer_vision.detection.rfdetr import (
        ALL_PERSON_CLASS_IDS,
        BALL_CLASS_IDS,
        PLAYER_CLASS_IDS,
        RFDETRSoccerDetector,
    )
    from soccer_vision.tracking.bytetrack import create_tracker, track_detections
    from soccer_vision.tracking.teams import sample_jersey_bgr
    import supervision as sv

    det = RFDETRSoccerDetector.from_pretrained(device="cuda")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = create_tracker(frame_rate=int(fps))

    stats = {k: [] for k in ("raw", "person", "player_gk", "on_field", "tracked", "ball")}
    kit_bgr = []

    for i in range(N):
        fn = START + i * INTERVAL
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if not ok:
            break

        dets = det.predict(frame, conf_threshold=0.3)
        stats["raw"].append(len(dets))
        if len(dets) == 0:
            for k in ("person", "player_gk", "on_field", "tracked", "ball"):
                stats[k].append(0)
            continue

        stats["ball"].append(int(np.isin(dets.class_id, list(BALL_CLASS_IDS)).sum()))
        person = dets[np.isin(dets.class_id, list(ALL_PERSON_CLASS_IDS))]
        stats["person"].append(len(person))
        stats["player_gk"].append(
            int(np.isin(dets.class_id, list(PLAYER_CLASS_IDS)).sum())
        )

        on_field = filter_spectators(person, frame.shape)
        stats["on_field"].append(len(on_field))

        # process.py merges ball back in before tracking, so mirror that
        ball_dets = dets[~np.isin(dets.class_id, list(ALL_PERSON_CLASS_IDS))]
        merged = sv.Detections.merge([ball_dets, on_field])
        tracked = track_detections(tracker, merged)
        stats["tracked"].append(len(tracked))

        for b in on_field.xyxy:
            c = sample_jersey_bgr(frame, b, None)
            if c is not None:
                kit_bgr.append(c)

    cap.release()

    print(f"video  : {VIDEO}")
    print(f"frames : {N} sampled every {INTERVAL} from {START}\n")
    print(f"{'stage':<34s} {'median':>7s} {'mean':>7s} {'min':>5s} {'max':>5s}")
    order = [("raw", "RF-DETR raw (all 4 classes)"),
             ("ball", "  of which class 0 = ball"),
             ("person", "person classes (1,2,3)"),
             ("player_gk", "player+GK only (1,3)"),
             ("on_field", "after filter_spectators"),
             ("tracked", "after ByteTrack")]
    for k, label in order:
        v = np.array(stats[k], dtype=float)
        print(f"{label:<34s} {np.median(v):7.1f} {v.mean():7.1f} "
              f"{v.min():5.0f} {v.max():5.0f}")

    raw_p = np.median(stats["person"])
    onf = np.median(stats["on_field"])
    trk = np.median(stats["tracked"])
    print(f"\nattribution (medians):")
    print(f"  filter_spectators drops {raw_p - onf:.0f} of {raw_p:.0f} people "
          f"({100 * (raw_p - onf) / max(raw_p, 1):.0f}%)")
    print(f"  ByteTrack then yields   {trk:.0f}")

    # what colours is the kit sampler actually seeing?
    if kit_bgr:
        arr = np.array(kit_bgr)
        print(f"\nkit colour sampled from bbox (no mask), n={len(arr)}:")
        print(f"  mean BGR {arr.mean(axis=0).round(0)}   "
              f"std {arr.std(axis=0).round(0)}")
        lab = cv2.cvtColor(arr.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_BGR2LAB).reshape(-1, 3)
        print(f"  L* range {lab[:, 0].min()}-{lab[:, 0].max()}, "
              f"median {np.median(lab[:, 0]):.0f}  "
              f"(low spread here = both kits collapsing to one cluster)")

    teams = json.loads((RUN / "tracks.json").read_text()).get("teams") or {}
    print(f"\nteams stamped in the run: {Counter(teams.values())}")


if __name__ == "__main__":
    main()
