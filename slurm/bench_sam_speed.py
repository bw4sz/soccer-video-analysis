"""Benchmark SAM automatic-mask-generation speed vs. grid density on this match.

Answers: how much of the 7.7h full-run cost is the 32x32 "segment everything"
grid, and do coarser grids still find every player? Times generate() on a set
of real live-action frames for several (points_per_side, points_per_batch)
configs, reporting s/frame, extrapolated full-run hours, and median player-box
count after the same size/aspect filter process uses.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

VIDEO = "/orange/ewhite/b.weinstein/soccer-video-analysis/data/SaintsU11_OVF_Jul192026.MP4"
# Effective SAM calls for the full match: 55354 frames / detect_interval 6 (5fps).
FULL_RUN_CALLS = 55354 / 6

# Same filter thresholds as src/soccer_vision/detection/sam2.py
MIN_AREA_RATIO = 0.0002
MAX_AREA_RATIO = 0.12

CONFIGS = [
    {"name": "baseline (current)", "points_per_side": 32, "points_per_batch": 64},
    {"name": "grid16 / batch128", "points_per_side": 16, "points_per_batch": 128},
    {"name": "grid12 / batch192", "points_per_side": 12, "points_per_batch": 192},
]

N_WARMUP = 2
N_TIMED = 18


def grab_frames() -> list[np.ndarray]:
    """Sample live-action frames spaced through the middle of the match."""
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    # frames 15000..33000 step 1000 -> 18 frames of mid-match live play
    for fn in range(15000, 15000 + (N_WARMUP + N_TIMED) * 1000, 1000):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def count_players(masks: list[dict], h: int, w: int) -> int:
    n = 0
    for md in masks:
        ys, xs = np.where(md["segmentation"])
        if len(ys) == 0:
            continue
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        area_ratio = (bw * bh) / (h * w)
        if not (MIN_AREA_RATIO < area_ratio < MAX_AREA_RATIO):
            continue
        aspect = bw / (bh + 1e-6)
        if 0.25 < aspect < 1.0:
            n += 1
    return n


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  |  full-run SAM calls: {FULL_RUN_CALLS:.0f}\n")
    frames = grab_frames()
    print(f"grabbed {len(frames)} frames\n")

    sam = sam_model_registry["vit_b"]().to(device)

    print(f"{'config':<22}{'s/frame':>10}{'full-run h':>12}{'med players':>14}{'raw masks':>12}")
    print("-" * 70)
    for cfg in CONFIGS:
        gen = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=cfg["points_per_side"],
            points_per_batch=cfg["points_per_batch"],
            pred_iou_thresh=0.70,
            stability_score_thresh=0.85,
            crop_n_layers=0,
            min_mask_region_area=100,
        )
        # warmup
        for f in frames[:N_WARMUP]:
            gen.generate(f[:, :, ::-1])
        torch.cuda.synchronize() if device == "cuda" else None

        times, player_counts, raw_counts = [], [], []
        for f in frames[N_WARMUP:N_WARMUP + N_TIMED]:
            rgb = f[:, :, ::-1]
            t0 = time.perf_counter()
            masks = gen.generate(rgb)
            torch.cuda.synchronize() if device == "cuda" else None
            times.append(time.perf_counter() - t0)
            h, w = f.shape[:2]
            player_counts.append(count_players(masks, h, w))
            raw_counts.append(len(masks))

        spf = float(np.mean(times))
        full_h = spf * FULL_RUN_CALLS / 3600
        print(f"{cfg['name']:<22}{spf:>10.3f}{full_h:>12.2f}"
              f"{int(np.median(player_counts)):>14}{int(np.median(raw_counts)):>12}")

    if device == "cuda":
        print(f"\npeak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
