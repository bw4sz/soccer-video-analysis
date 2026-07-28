"""Why does SAM3 crawl on the U14G Veo footage but not on the U11 match?

Job 38176317 spent 5h without clearing 250 sampled frames (>72 s/frame) where the
U11 full match ran at ~2.5 s/frame. SAM3's cost is per-masklet, so the suspect is
object count: this venue is a multi-pitch complex, and the wide Veo view takes in
other games, sidelines and spectators — every one of them a "soccer player".

Measures objects/frame and s/frame for both detectors on a few frames of each
match, so the comparison is like-for-like.

Usage: python slurm/diag_u14g_detect_cost.py
"""

from __future__ import annotations

import time

import numpy as np

from soccer_vision.io.video import VideoReader

CLIPS = {
    "u14g (Veo, multi-pitch venue)": "data/u14g_sampler600.mp4",
    "u11  (XbotGo, single pitch)": "data/SaintsU11_OVF_Jul192026.MP4",
}
N_FRAMES = 6


def _frames(path: str, n: int) -> list[np.ndarray]:
    reader = VideoReader(path)
    try:
        step = max(1, reader.total_frames // (n + 2))
        return [f for f in (reader.read_frame(step * (i + 1)) for i in range(n))
                if f is not None]
    finally:
        reader.close()


def bench_sam3(frames: list[np.ndarray]) -> tuple[float, float]:
    from soccer_vision.tracking.sam3 import SAM3PlayerTracker

    tracker = SAM3PlayerTracker(device="cuda", prompt="soccer player")
    tracker.start()
    counts, elapsed = [], []
    try:
        for frame in frames:
            t0 = time.time()
            dets = tracker.track(frame)
            elapsed.append(time.time() - t0)
            counts.append(len(dets))
    finally:
        tracker.close()
    return float(np.mean(counts)), float(np.mean(elapsed))


def bench_rfdetr(frames: list[np.ndarray]) -> tuple[float, float]:
    from soccer_vision.detection.rfdetr import RFDETRSoccerDetector

    det = RFDETRSoccerDetector.from_pretrained(device="cuda")
    counts, elapsed = [], []
    for frame in frames:
        t0 = time.time()
        dets = det.predict_players(frame)
        elapsed.append(time.time() - t0)
        counts.append(len(dets))
    return float(np.mean(counts)), float(np.mean(elapsed))


def main():
    for label, path in CLIPS.items():
        print(f"\n=== {label} — {path}")
        frames = _frames(path, N_FRAMES)
        print(f"  {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")
        for name, fn in (("sam3", bench_sam3), ("rfdetr", bench_rfdetr)):
            try:
                objects, secs = fn(frames)
                print(f"  {name:7s} {objects:6.1f} objects/frame   {secs:6.2f} s/frame")
            except Exception as exc:  # a missing model shouldn't hide the other's number
                print(f"  {name:7s} FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
