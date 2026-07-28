"""Where does SAM3's per-frame time actually go, and why is it ~23x RF-DETR?

Job 38133841 measured the headline gap on FOOTPASS footage (640x352):
SAM3 0.91 s/frame vs RF-DETR 0.04 s/frame. That says *how much* slower, not
*why*. This script decomposes SAM3's per-frame cost into the three stages the
adapter actually runs (soccer_vision/tracking/sam3.py):

  preproc   Sam3VideoProcessor on the BGR->RGB frame           (CPU)
  forward   Sam3VideoModel(inference_session=..., frame=...)   (GPU)
  postproc  _masks_to_detections: mask -> cpu -> np.where      (GPU->CPU + CPU)

and tests the three hypotheses for the gap:

  H1 memory-bank growth  SAM3's video model attends over a masklet memory bank
     that grows with every frame in a chunk, so cost should rise with
     frame-in-chunk index. Measured by running one long chunk with rotation
     effectively disabled and regressing time on index.

  H2 mask postprocessing  Each of ~20-45 objects yields a full HxW mask that is
     moved to CPU, optionally cv2.resized, then np.where'd. At 1080p that is
     45 x 2.07M elements per frame of pure CPU work RF-DETR never does (it
     returns boxes).

  H3 two sessions  process.py runs a second SAM3 session for "soccer ball"
     (SAM3PlayerTracker.sharing), so a production frame costs 2 forwards, not 1.

Also sweeps resolution, because the production video is 1920x1080 while the
0.91 s/frame figure came from 640x352.

Usage: python slurm/profile_detector_speed.py --video <path> --start <frame>
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
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# SAM3: staged timing, reimplementing track() inline so each stage is separable
# ---------------------------------------------------------------------------


def sam3_staged(tracker, frame_bgr, min_score):
    """One tracker.track() call, timed stage by stage. Mirrors sam3.py::track."""
    import torch

    from soccer_vision.tracking.sam3 import _masks_to_detections

    h, w = frame_bgr.shape[:2]

    if tracker.session is None:
        tracker.start()
    elif tracker._frame_in_chunk >= tracker.chunk_frames:
        t = time.perf_counter()
        tracker._rotate()
        rot = time.perf_counter() - t
    else:
        rot = 0.0

    t = time.perf_counter()
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    inputs = tracker.processor(images=rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(tracker.device, dtype=tracker._dtype)
    sync()
    t_pre = time.perf_counter() - t

    t = time.perf_counter()
    with torch.inference_mode():
        out = tracker.model(inference_session=tracker.session, frame=pixel_values[0])
    sync()
    t_fwd = time.perf_counter() - t

    t = time.perf_counter()
    dets = _masks_to_detections(
        out.object_ids, out.obj_id_to_mask, getattr(out, "obj_id_to_score", None),
        h, w, min_score,
    )
    sync()
    t_post = time.perf_counter() - t

    t = time.perf_counter()
    dets = tracker._assign_global_ids(dets)
    tracker._restitch = False
    tracker._frame_in_chunk += 1
    if dets.mask is not None and len(dets):
        tracker._prev_masks = {int(i): m for i, m in zip(dets.tracker_id, dets.mask)}
    t_id = time.perf_counter() - t

    return {
        "n_obj": len(dets),
        "rotate": rot,
        "preproc": t_pre,
        "forward": t_fwd,
        "postproc": t_post,
        "idmap": t_id,
        "total": rot + t_pre + t_fwd + t_post + t_id,
        "frame_in_chunk": tracker._frame_in_chunk - 1,
    }


def run_sam3(frames, prompt, chunk_frames, min_score=0.3, label=""):
    from soccer_vision.tracking.sam3 import SAM3PlayerTracker

    tr = SAM3PlayerTracker(device="cuda", prompt=prompt, min_score=min_score,
                           chunk_frames=chunk_frames)
    tr.start()
    rows = []
    for i, f in enumerate(frames):
        r = sam3_staged(tr, f, min_score)
        r["i"] = i
        rows.append(r)
        if i % 20 == 0:
            print(f"    [{label}] frame {i:3d}  {r['total']:.3f}s "
                  f"(fwd {r['forward']:.3f} post {r['postproc']:.3f}) "
                  f"n={r['n_obj']}", flush=True)
    tr.close()
    return rows


def run_rfdetr(frames, conf=0.3):
    import torch

    from soccer_vision.detection.rfdetr import RFDETRSoccerDetector

    det = RFDETRSoccerDetector.from_pretrained(device="cuda")
    rows = []
    for i, f in enumerate(frames):
        t = time.perf_counter()
        dets = det.predict(f, conf_threshold=conf)
        sync()
        total = time.perf_counter() - t
        rows.append({"i": i, "total": total, "n_obj": len(dets)})
        if i % 20 == 0:
            print(f"    [rfdetr] frame {i:3d}  {total:.3f}s n={len(dets)}", flush=True)
    del det
    torch.cuda.empty_cache()
    return rows


def summarize(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                  "p90": float(np.percentile(v, 90)), "sum": float(v.sum())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="/orange/ewhite/b.weinstein/soccer-video-analysis/"
                                      "data/u14g_smoke180.mp4")
    p.add_argument("--start", type=int, default=1200)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--chunk-frames", type=int, default=60)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"video: {args.video}  start={args.start}  frames={args.frames}\n", flush=True)

    frames = read_frames(args.video, args.start, args.frames)
    h, w = frames[0].shape[:2]
    print(f"loaded {len(frames)} frames at {w}x{h}\n", flush=True)

    results = {"video": args.video, "resolution": [w, h], "n_frames": len(frames),
               "gpu": torch.cuda.get_device_name(0)}

    # --- A. RF-DETR baseline -------------------------------------------------
    print("[A] RF-DETR baseline (native res)", flush=True)
    rf = run_rfdetr(frames)
    results["rfdetr"] = summarize(rf, ["total"])
    results["rfdetr"]["n_obj_median"] = float(np.median([r["n_obj"] for r in rf]))
    print(f"  -> {results['rfdetr']['total']['mean']:.3f} s/frame\n", flush=True)

    # --- B. SAM3 staged, production chunking ---------------------------------
    print(f"[B] SAM3 'soccer player' staged (chunk_frames={args.chunk_frames})", flush=True)
    s_player = run_sam3(frames, "soccer player", args.chunk_frames, label="player")
    keys = ["total", "preproc", "forward", "postproc", "idmap", "rotate"]
    results["sam3_player"] = summarize(s_player, keys)
    results["sam3_player"]["n_obj_median"] = float(np.median([r["n_obj"] for r in s_player]))
    results["sam3_player"]["per_frame"] = [
        {k: r[k] for k in keys + ["i", "n_obj", "frame_in_chunk"]} for r in s_player
    ]
    m = results["sam3_player"]
    print(f"  -> {m['total']['mean']:.3f} s/frame  "
          f"(pre {m['preproc']['mean']:.3f} | fwd {m['forward']['mean']:.3f} | "
          f"post {m['postproc']['mean']:.3f} | id {m['idmap']['mean']:.3f})\n", flush=True)

    # --- C. H1: memory-bank growth (one long chunk, no rotation) -------------
    print("[C] SAM3 with rotation disabled — does cost grow with chunk depth?", flush=True)
    try:
        s_long = run_sam3(frames, "soccer player", chunk_frames=10_000, label="nochunk")
        results["sam3_no_rotation"] = summarize(s_long, keys)
        results["sam3_no_rotation"]["per_frame"] = [
            {k: r[k] for k in keys + ["i", "n_obj"]} for r in s_long
        ]
        fwd = np.array([r["forward"] for r in s_long])
        idx = np.arange(len(fwd))
        slope, intercept = np.polyfit(idx, fwd, 1)
        results["sam3_no_rotation"]["forward_growth_s_per_frame"] = float(slope)
        results["sam3_no_rotation"]["forward_intercept_s"] = float(intercept)
        print(f"  -> forward time grows {slope * 1000:.2f} ms per frame of chunk depth "
              f"(intercept {intercept:.3f}s)\n", flush=True)
    except RuntimeError as exc:
        print(f"  !! OOM/err with rotation disabled: {exc}\n", flush=True)
        results["sam3_no_rotation"] = {"error": str(exc)}
        torch.cuda.empty_cache()

    # --- D. H3: the second (ball) session ------------------------------------
    print("[D] SAM3 'soccer ball' second session — production runs both", flush=True)
    s_ball = run_sam3(frames[:60], "soccer ball", args.chunk_frames, label="ball")
    results["sam3_ball"] = summarize(s_ball, keys)
    results["sam3_ball"]["n_obj_median"] = float(np.median([r["n_obj"] for r in s_ball]))
    print(f"  -> {results['sam3_ball']['total']['mean']:.3f} s/frame\n", flush=True)

    # --- E. resolution sweep --------------------------------------------------
    print("[E] resolution sweep (postproc scales with HxW, forward should not)", flush=True)
    res_rows = {}
    for scale, name in [(1 / 3, "640x360"), (2 / 3, "1280x720")]:
        small = [cv2.resize(f, (int(w * scale), int(h * scale))) for f in frames[:60]]
        rr = run_sam3(small, "soccer player", args.chunk_frames, label=name)
        res_rows[name] = summarize(rr, keys)
        res_rows[name]["n_obj_median"] = float(np.median([r["n_obj"] for r in rr]))
        print(f"  -> {name}: {res_rows[name]['total']['mean']:.3f} s/frame "
              f"(fwd {res_rows[name]['forward']['mean']:.3f} "
              f"post {res_rows[name]['postproc']['mean']:.3f})", flush=True)
    results["resolution_sweep"] = res_rows

    (OUT_DIR / "profile.json").write_text(json.dumps(results, indent=2))

    # --- report ---------------------------------------------------------------
    rfm = results["rfdetr"]["total"]["mean"]
    spm = results["sam3_player"]["total"]["mean"]
    sbm = results["sam3_ball"]["total"]["mean"]
    print("\n" + "=" * 70)
    print("SUMMARY  (per detection-frame, 1 frame = 1 sampled frame at 5fps)")
    print("=" * 70)
    print(f"  RF-DETR (players+ball, one pass) : {rfm:.3f} s")
    print(f"  SAM3 player session              : {spm:.3f} s")
    print(f"  SAM3 ball session                : {sbm:.3f} s")
    print(f"  SAM3 production total (both)     : {spm + sbm:.3f} s")
    print(f"  slowdown vs RF-DETR              : {(spm + sbm) / rfm:.1f}x")
    print()
    m = results["sam3_player"]
    tot = m["total"]["mean"]
    for k in ["preproc", "forward", "postproc", "idmap", "rotate"]:
        print(f"  {k:9s} {m[k]['mean']:.3f}s  ({100 * m[k]['mean'] / tot:4.1f}%)")
    print(f"\nwrote {OUT_DIR / 'profile.json'}")


if __name__ == "__main__":
    main()
