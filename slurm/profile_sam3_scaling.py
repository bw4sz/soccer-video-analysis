"""What does SAM3's forward time actually scale with?

Follow-up to job 38176330, which established that 93% of SAM3's per-frame cost
is the model forward (not preprocessing, not mask->box conversion) and that
resolution barely matters (640x360 is only 1.2x faster than 1920x1080 despite
9x fewer pixels — SAM3 resizes internally to a fixed size).

Two candidate drivers remain, which that run could not separate because the
player session's object count barely varied (18-24):

  S1 number of tracked masklets.  SAM3 propagates and mask-decodes per object,
     where RF-DETR decodes a fixed query set in one batched pass. If true,
     forward time is roughly linear in object count, and the ball session's
     0.27s vs the player session's 1.26s is that line, not a session overhead.

  S2 masklet-memory depth.  The video model attends over a memory bank that
     grows every frame in a chunk. Job 38176330 fit 3.9 ms per frame of depth
     with a positive correlation (r=0.41).

This sweeps prompts that yield different object counts over the *same* frames
with the *same* chunk depth, so object count is the only thing moving, and
separately walks chunk_frames to isolate depth.

Usage: python slurm/profile_sam3_scaling.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/orange/ewhite/b.weinstein/soccer-video-analysis/src")

OUT_DIR = Path("/orange/ewhite/b.weinstein/soccer-video-analysis/runs/detector_speed_profile")


def sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def read_frames(video: str, start: int, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def time_prompt(frames, prompt, chunk_frames, min_score=0.3):
    """Fresh session, time each frame's forward pass and count live masklets."""
    import torch

    from soccer_vision.tracking.sam3 import SAM3PlayerTracker, _masks_to_detections

    tr = SAM3PlayerTracker(device="cuda", prompt=prompt, min_score=min_score,
                           chunk_frames=chunk_frames)
    tr.start()
    rows = []
    for i, frame in enumerate(frames):
        if tr._frame_in_chunk >= tr.chunk_frames:
            tr._rotate()
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        inputs = tr.processor(images=rgb, return_tensors="pt")
        pv = inputs["pixel_values"].to(tr.device, dtype=tr._dtype)
        sync()
        t = time.perf_counter()
        with torch.inference_mode():
            out = tr.model(inference_session=tr.session, frame=pv[0])
        sync()
        fwd = time.perf_counter() - t
        # n_raw = masklets the model is actually propagating (pre-threshold);
        # that, not the surviving detection count, is what it pays for.
        n_raw = len(out.object_ids)
        dets = _masks_to_detections(out.object_ids, out.obj_id_to_mask,
                                    getattr(out, "obj_id_to_score", None), h, w, min_score)
        tr._frame_in_chunk += 1
        rows.append({"i": i, "forward": fwd, "n_raw": n_raw, "n_kept": len(dets),
                     "depth": tr._frame_in_chunk - 1})
    tr.close()
    del tr
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="/orange/ewhite/b.weinstein/soccer-video-analysis/"
                                      "data/u14g_smoke180.mp4")
    p.add_argument("--start", type=int, default=1200)
    p.add_argument("--frames", type=int, default=40)
    args = p.parse_args()

    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = read_frames(args.video, args.start, args.frames)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}\n", flush=True)

    results = {}

    # --- S1: object-count sweep, same frames, same depth --------------------
    print("[S1] prompt sweep — object count is the only variable")
    print(f"{'prompt':22s} {'n_masklets':>11s} {'fwd mean':>9s} {'fwd f0':>8s}")
    s1 = {}
    for prompt in ["soccer ball", "referee", "goalkeeper", "soccer player",
                   "person", "person on a sports field"]:
        try:
            rows = time_prompt(frames, prompt, chunk_frames=10_000)
        except RuntimeError as exc:
            print(f"{prompt:22s} ERROR {exc}")
            continue
        n = float(np.median([r["n_raw"] for r in rows]))
        fw = float(np.mean([r["forward"] for r in rows]))
        s1[prompt] = {"n_raw_median": n, "forward_mean": fw,
                      "forward_first": rows[0]["forward"],
                      "n_kept_median": float(np.median([r["n_kept"] for r in rows])),
                      "rows": rows}
        print(f"{prompt:22s} {n:11.1f} {fw:9.3f} {rows[0]['forward']:8.3f}", flush=True)
    results["prompt_sweep"] = s1

    xs = np.array([v["n_raw_median"] for v in s1.values()])
    ys = np.array([v["forward_mean"] for v in s1.values()])
    if len(xs) >= 3 and np.ptp(xs) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * xs + intercept
        r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
        results["object_scaling"] = {"slope_s_per_object": float(slope),
                                     "intercept_s": float(intercept), "r2": float(r2)}
        print(f"\n  forward = {intercept:.3f}s + {slope * 1000:.1f} ms/masklet   "
              f"(R^2={r2:.3f})")
        print(f"  -> 20 players cost {intercept + 20 * slope:.2f}s; "
              f"fixed image+text cost is only {intercept:.2f}s\n", flush=True)

    # --- S2: chunk-depth sweep ----------------------------------------------
    print("[S2] chunk-depth sweep — 'soccer player', varying rotation period")
    s2 = {}
    for cf in [10, 20, 30, 60]:
        rows = time_prompt(frames, "soccer player", chunk_frames=cf)
        fw = float(np.mean([r["forward"] for r in rows]))
        s2[str(cf)] = {"forward_mean": fw, "rows": rows}
        print(f"  chunk_frames={cf:3d}  forward mean {fw:.3f}s", flush=True)
    results["chunk_sweep"] = s2

    # depth slope pooled from the deepest run
    rows = s2["60"]["rows"]
    d = np.array([r["depth"] for r in rows], float)
    f = np.array([r["forward"] for r in rows], float)
    if np.ptp(d) > 0:
        sl, ic = np.polyfit(d, f, 1)
        results["depth_scaling"] = {"slope_s_per_depth": float(sl), "intercept_s": float(ic)}
        print(f"\n  forward = {ic:.3f}s + {sl * 1000:.1f} ms per frame of chunk depth")

    (OUT_DIR / "scaling.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_DIR / 'scaling.json'}")


if __name__ == "__main__":
    main()
