"""Is the SAM3-vs-RF-DETR speed gap a property of the model or of the video?

We have four speed numbers from three different scripts (jobs 38133841, 38176330,
38186278) and they disagree on the ratio -- 13x, 23x, 26x, 29x. Before reading
that spread as "it depends on the footage", the methodology has to be made
identical, because two artefacts are known to be in there:

  1. **Warm-up.** 38186278 timed 6 frames with no warm-up pass, so the first
     frame's cuDNN autotune landed in the mean. That alone can double a 0.05
     s/frame number, and RF-DETR's own scaling study (38176330) found it flat at
     0.046 s/frame from 5 to 27 objects -- so a real 2x swing between videos
     would contradict a measurement we trust more.
  2. **Which SAM3 sessions counted.** Production runs *two* SAM3 sessions per
     frame (players, then ball); 38186278 timed only the player session, while
     38176330 reported the pair. RF-DETR gets both from one forward pass.

So: same frames, same warm-up, same definition of "a frame of production work",
across every distinct camera we own. Youth clips are sampled in for camera
diversity -- they are the only footage here we did not shoot ourselves.

Usage: python slurm/bench_detector_speed_multivideo.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from soccer_vision.io.video import VideoReader

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "detector_speed_multivideo_bursts" / "results.json"

# One entry per distinct camera/venue. The two U14G files come off the same
# match, so only the sampler (which spans the whole game, and so the whole sun
# arc) is carried.
CLIPS: list[tuple[str, str]] = [
    ("u14g-veo-multipitch", "data/u14g_sampler600.mp4"),
    ("u11-xbotgo", "data/SaintsU11_OVF_Jul192026.MP4"),
    ("saints-16b-sideline", "data/match-saints-16b-pre-mls-next-2026-04-26.mp4"),
]
N_YOUTH = 4          # harvested clips, for cameras we did not shoot
N_BURSTS = 3         # sampling positions spread across the video
BURST_LEN = 15       # CONSECUTIVE frames per position (see note below)
N_WARMUP = 5         # discarded frames at the head of each burst

# Frames must come in **consecutive bursts**, not spread evenly. SAM3 is a
# stateful video tracker: hand it frames a minute apart and every frame is a
# scene cut, so masklets churn and it re-detects from scratch each time --
# which is not what `process` does (it feeds consecutive 5 fps samples, 0.2 s
# apart, where tracking continuity holds). An evenly-spaced sweep measured SAM3
# at 3.07 s/frame against the 1.62 s/frame that job 38176330 got on consecutive
# frames; that gap is the artefact, not a finding. RF-DETR is stateless and
# scored 0.048 either way, which is what makes the artefact attributable.
# Bursts keep the coverage (three points in the match) without the scene cuts,
# and the session is reset per burst -- the same design as the FOOTPASS eval.


def _add_youth_clips() -> None:
    """Sample a few harvested clips so the sweep covers unfamiliar cameras."""
    clips = sorted((ROOT / "data" / "youth_clips" / "clips").glob("*.mp4"))
    if not clips:
        return
    rng = random.Random(0)
    for path in rng.sample(clips, min(N_YOUTH, len(clips))):
        CLIPS.append((f"youth-{path.stem[:11]}", str(path.relative_to(ROOT))))


def _bursts(path: str) -> tuple[list[list[np.ndarray]], tuple[int, int]]:
    """N_BURSTS runs of BURST_LEN consecutive frames, spread across the video.

    Consecutive at the 6-frame stride `process` samples at (5 fps from 30 fps),
    so each burst is what the tracker would really see."""
    reader = VideoReader(str(ROOT / path))
    try:
        stride = max(1, int(round(reader.fps / 5)) if reader.fps else 6)
        span = BURST_LEN * stride
        bursts = []
        for b in range(N_BURSTS):
            start = (reader.total_frames - span) * (b + 1) // (N_BURSTS + 1)
            if start < 0:
                continue
            frames = [f for f in (reader.read_frame(start + i * stride)
                                  for i in range(BURST_LEN)) if f is not None]
            if len(frames) > N_WARMUP:
                bursts.append(frames)
        return bursts, (int(reader.width), int(reader.height))
    finally:
        reader.close()


def bench_sam3(bursts: list[list[np.ndarray]]) -> dict:
    """Time SAM3 the way `process` actually runs it: a player session AND a ball
    session over every frame. `sharing` reuses the loaded weights, which is what
    process does -- so this is two forwards per frame, not two model loads.

    Sessions are restarted per burst, because a burst is a jump in time."""
    from soccer_vision.tracking.sam3 import SAM3PlayerTracker

    counts, t_player, t_ball = [], [], []
    for frames in bursts:
        players = SAM3PlayerTracker(device="cuda", prompt="soccer player")
        ball = SAM3PlayerTracker.sharing(players, prompt="soccer ball")
        players.start()
        ball.start()
        try:
            for i, frame in enumerate(frames):
                t0 = time.time()
                dets = players.track(frame)
                t1 = time.time()
                ball.track(frame)
                t2 = time.time()
                if i < N_WARMUP:      # session still filling its memory bank
                    continue
                counts.append(len(dets))
                t_player.append(t1 - t0)
                t_ball.append(t2 - t1)
        finally:
            players.close()
            ball.close()
    return {
        "objects_per_frame": float(np.mean(counts)),
        "s_per_frame": float(np.mean(t_player) + np.mean(t_ball)),
        "s_per_frame_players": float(np.mean(t_player)),
        "s_per_frame_ball": float(np.mean(t_ball)),
        "n_timed": len(counts),
    }


def bench_rfdetr(bursts: list[list[np.ndarray]]) -> dict:
    """Time RF-DETR the way `process` runs it: ONE predict() per frame returning
    players and ball together, which is then split by class_id."""
    from soccer_vision.detection.rfdetr import RFDETRSoccerDetector

    det = RFDETRSoccerDetector.from_pretrained(device="cuda")
    counts, elapsed = [], []
    for frames in bursts:
        for i, frame in enumerate(frames):
            t0 = time.time()
            dets = det.predict(frame)
            dt = time.time() - t0
            if i < N_WARMUP:      # first call pays cuDNN autotune
                continue
            counts.append(len(dets))
            elapsed.append(dt)
    return {
        "objects_per_frame": float(np.mean(counts)),
        "s_per_frame": float(np.mean(elapsed)),
        "n_timed": len(counts),
    }


def main() -> None:
    _add_youth_clips()
    results = []
    for label, path in CLIPS:
        print(f"\n=== {label} — {path}", flush=True)
        try:
            bursts, (w, h) = _bursts(path)
        except Exception as exc:
            print(f"  unreadable: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not bursts:
            print("  too short for a burst", flush=True)
            continue
        n = sum(len(b) for b in bursts)
        print(f"  {len(bursts)} bursts x ~{n // len(bursts)} consecutive frames, "
              f"{w}x{h}", flush=True)
        row: dict = {"video": label, "path": path, "width": w, "height": h,
                     "n_bursts": len(bursts), "n_frames": n}
        for name, fn in (("rfdetr", bench_rfdetr), ("sam3", bench_sam3)):
            try:
                row[name] = fn(bursts)
                r = row[name]
                print(f"  {name:7s} {r['objects_per_frame']:6.1f} obj/frame   "
                      f"{r['s_per_frame']:6.3f} s/frame  (n={r['n_timed']})",
                      flush=True)
            except Exception as exc:
                # A missing/gated model must not hide the other detector's number.
                row[name] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {name:7s} FAILED: {type(exc).__name__}: {exc}", flush=True)
        if "s_per_frame" in row.get("sam3", {}) and "s_per_frame" in row.get("rfdetr", {}):
            row["ratio"] = row["sam3"]["s_per_frame"] / row["rfdetr"]["s_per_frame"]
            print(f"  ratio   {row['ratio']:6.1f}x", flush=True)
        results.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}", flush=True)

    # A 60-min match at the 5 fps `process` samples at = 18000 detection-frames.
    print("\n--- projected 60-min match (18000 detection-frames, detector only)")
    for row in results:
        if "ratio" not in row:
            continue
        rf = row["rfdetr"]["s_per_frame"] * 18000 / 3600
        s3 = row["sam3"]["s_per_frame"] * 18000 / 3600
        print(f"  {row['video']:24s} rfdetr {rf:5.2f} h   sam3 {s3:5.2f} h   "
              f"{row['ratio']:5.1f}x", flush=True)


if __name__ == "__main__":
    main()
