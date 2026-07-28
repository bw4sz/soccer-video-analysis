"""In-domain player-detection evaluation on FOOTPASS broadcast video.

Detection counts on our own Veo youth footage have **no ground truth at all** —
they are counts, not accuracies. This script produces the missing baseline:
precision/recall against real per-player boxes, on broadcast footage the
detector was trained for.

Ground truth is the FOOTPASS/SN-PCBAS-2026 tactical data (`val_tactical_data.h5`),
whose ROI_* columns are per-frame player boxes in **fullHD** coordinates. The
local videos are 640x352, and upstream's dataloader (vendor/FOOTPASS/utils/
TAAD_Dataset.py:271-274) rescales anisotropically by x/3 and y/3.068181 — the
video was squashed, not letterboxed — so we use exactly those factors.

Two caveats this eval cannot remove, both worth remembering when reading the
output:

1. The GT is Footovision's production tracker output, i.e. strong pseudo-GT,
   not hand-labelled truth.
2. A median GT player is 38x79 px in fullHD -> **13x26 px** on the local 640x352
   video, *smaller* than a player on our Veo footage. So a low score here may be
   resolution rather than domain, and this script cannot separate the two.
   Naively upsampling the frame would not answer it — cubic interpolation adds
   no information. Settling the resolution question needs either the fullHD
   videos (on HuggingFace, never downloaded here) or SAHI-style tiling, where
   each tile spends the model's full input budget on a quarter of the pitch.

The GT counts only the ~22 outfield players + keepers — **not referees, coaches
or crowd** — all of which RF-DETR's player class happily
returns on a broadcast frame. So raw precision is pessimistic by construction; we
also report precision restricted to the field envelope (the bounding box of that
frame's GT, expanded), which is the closer analogue of how the pipeline uses a
field mask downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import h5py
import numpy as np

DATA_ROOT = "/blue/ewhite/b.weinstein/soccer-vision-data/footpass"
VAL_H5 = f"{DATA_ROOT}/data/val_tactical_data.h5"
VIDEO_DIR = f"{DATA_ROOT}/videos"
OUT_DIR = "/orange/ewhite/b.weinstein/soccer-video-analysis/runs/footpass_detector_eval"

# Column indices, per vendor/FOOTPASS/utils/TAAD_Dataset.py:178
FRAME, PLAYER_ID, ROI_X, ROI_Y, ROI_W, ROI_H = 0, 1, 9, 10, 11, 12
# fullHD -> 640x352, anisotropic, matching the upstream dataloader exactly.
SCALE_X, SCALE_Y = 3.0, 3.068181

SCORE_THRS = (0.2, 0.3, 0.5)
IOU_THRS = (0.3, 0.5)
WIDE_SHOT_MIN_GT = 8  # frames with >=8 visible players ~ the tactical wide shot


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------


def load_gt(h5_path: str, key: str) -> dict[int, np.ndarray]:
    """frame -> (N,4) xyxy player boxes in 640x352 video coordinates.

    Rows whose ROI is NaN are players tracked in pitch space but not visible in
    the current camera framing; they are dropped, so the GT for a frame is
    exactly "who is on screen".
    """
    with h5py.File(h5_path, "r") as f:
        a = np.asarray(f[key])
    a = a[~np.isnan(a[:, ROI_X])]
    x0 = a[:, ROI_X] / SCALE_X
    y0 = a[:, ROI_Y] / SCALE_Y
    x1 = (a[:, ROI_X] + a[:, ROI_W]) / SCALE_X
    y1 = (a[:, ROI_Y] + a[:, ROI_H]) / SCALE_Y
    boxes = np.stack([x0, y0, x1, y1], axis=1)
    frames = a[:, FRAME].astype(int)

    gt: dict[int, list] = defaultdict(list)
    for fr, b in zip(frames, boxes):
        gt[int(fr)].append(b)
    return {k: np.array(v, dtype=float) for k, v in gt.items()}


def pick_windows(
    gt: dict[int, np.ndarray], n_windows: int, window_frames: int, seed: int
) -> list[int]:
    """Uniformly spaced window starts over the half, snapped to frames with GT.

    Deliberately *not* filtered to wide shots — replays and close-ups are part
    of a broadcast, and pre-selecting the easy framings would inflate the score.
    The wide-shot subset is reported separately instead.
    """
    frames = np.array(sorted(gt))
    usable = frames[(frames >= frames.min()) & (frames <= frames.max() - window_frames)]
    if len(usable) < n_windows:
        raise SystemExit(f"not enough GT frames in this half ({len(usable)})")
    rng = np.random.default_rng(seed)
    # even coverage of the half, jittered so we don't always land on the same
    # phase of the broadcast's shot rhythm
    edges = np.linspace(0, len(usable) - 1, n_windows + 1).astype(int)
    starts = []
    for i in range(n_windows):
        lo, hi = edges[i], max(edges[i], edges[i + 1] - 1)
        starts.append(int(usable[rng.integers(lo, hi + 1)]))
    return sorted(set(starts))


# --------------------------------------------------------------------------
# frame reading
# --------------------------------------------------------------------------


def read_window(cap, start: int, n: int) -> list[tuple[int, np.ndarray]]:
    """Decode `n` consecutive frames from `start`, labelled by true index.

    We trust OpenCV's reported position rather than assuming the seek landed
    exactly, because a mis-set frame index would silently shift the GT and make
    every detector look broken.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, bgr = cap.read()
        if not ok:
            break
        out.append((idx, bgr))
    return out


# --------------------------------------------------------------------------
# detectors — each returns, per frame, an (M,4) xyxy array + (M,) scores,
# already mapped back to native 640x352 coordinates.
# --------------------------------------------------------------------------


class RFDetrArm:
    name = "rfdetr"

    def __init__(self, conf: float = 0.15):
        sys.path.insert(0, "/orange/ewhite/b.weinstein/soccer-video-analysis/src")
        from soccer_vision.detection.rfdetr import RFDETRSoccerDetector

        self.det = RFDETRSoccerDetector.from_pretrained(device="cuda")
        self.conf = conf

    def start_window(self) -> None:
        pass

    def detect(self, bgr: np.ndarray):
        dets = self.det.predict_players(bgr, conf_threshold=self.conf)
        if len(dets) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        return np.asarray(dets.xyxy, dtype=float), np.asarray(dets.confidence, dtype=float)


# --------------------------------------------------------------------------
# matching + metrics
# --------------------------------------------------------------------------


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    px0, py0, px1, py1 = (pred[:, i][:, None] for i in range(4))
    gx0, gy0, gx1, gy1 = (gt[:, i][None, :] for i in range(4))
    iw = np.clip(np.minimum(px1, gx1) - np.maximum(px0, gx0), 0, None)
    ih = np.clip(np.minimum(py1, gy1) - np.maximum(py0, gy0), 0, None)
    inter = iw * ih
    pa = (px1 - px0) * (py1 - py0)
    ga = (gx1 - gx0) * (gy1 - gy0)
    union = pa + ga - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy_match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou_thr: float):
    """Confidence-ordered greedy assignment; returns (tp, fp, fn)."""
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    ious = iou_matrix(pred, gt)
    order = np.argsort(-scores)
    taken = np.zeros(len(gt), dtype=bool)
    tp = 0
    for i in order:
        cand = np.where(~taken, ious[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            taken[j] = True
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def field_envelope(gt: np.ndarray, pad: float = 0.10) -> tuple[float, float, float, float]:
    """Bounding box of this frame's GT players, padded — a stand-in field mask."""
    x0, y0 = gt[:, 0].min(), gt[:, 1].min()
    x1, y1 = gt[:, 2].max(), gt[:, 3].max()
    w, h = x1 - x0, y1 - y0
    return x0 - pad * w, y0 - pad * h, x1 + pad * w, y1 + pad * h


def inside(boxes: np.ndarray, env) -> np.ndarray:
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    return (cx >= env[0]) & (cx <= env[2]) & (cy >= env[1]) & (cy <= env[3])


# --------------------------------------------------------------------------
# visualisation — the only real check that GT and video frames are aligned
# --------------------------------------------------------------------------


def save_overlay(bgr, gt, pred, scores, path, score_thr=0.3):
    canvas = bgr.copy()
    for b in gt:
        cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
    for b, s in zip(pred, scores):
        if s < score_thr:
            continue
        cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
    cv2.putText(canvas, f"GT {len(gt)} (green) / pred {(scores >= score_thr).sum()} (red)",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))


# --------------------------------------------------------------------------


def run_arm(arm, windows, cap, gt, out_dir: str, tag: str) -> dict:
    per_frame = []
    t0 = time.time()
    n_frames = 0
    saved = 0

    for w_i, start in enumerate(windows):
        frames = read_window(cap, start, args.window_frames)
        arm.start_window()
        for f_i, (idx, bgr) in enumerate(frames):
            g = gt.get(idx)
            if g is None or len(g) == 0:
                continue
            try:
                pred, scores = arm.detect(bgr)
            except RuntimeError as exc:  # e.g. CUDA OOM — record and move on
                print(f"  !! window {w_i} frame {idx}: {exc}")
                break
            n_frames += 1
            env = field_envelope(g)
            per_frame.append(
                {"frame": idx, "gt": g, "pred": pred, "scores": scores,
                 "in_field": inside(pred, env) if len(pred) else np.zeros(0, bool)}
            )
            if f_i == len(frames) // 2 and saved < 4:
                save_overlay(bgr, g, pred, scores,
                             f"{out_dir}/overlay_{tag}_w{w_i:02d}_f{idx}.jpg")
                saved += 1

    dt = time.time() - t0
    print(f"  {tag}: {n_frames} frames in {dt:.0f}s ({dt / max(n_frames, 1):.2f}s/frame)")

    results = {"tag": tag, "n_frames": n_frames, "sec_per_frame": dt / max(n_frames, 1),
               "metrics": {}}
    for score_thr in SCORE_THRS:
        for iou_thr in IOU_THRS:
            for subset in ("all", "wide"):
                tp = fp = fn = 0
                tp_f = fp_f = 0
                n = 0
                for r in per_frame:
                    if subset == "wide" and len(r["gt"]) < WIDE_SHOT_MIN_GT:
                        continue
                    n += 1
                    keep = r["scores"] >= score_thr
                    a, b, c = greedy_match(r["pred"][keep], r["scores"][keep], r["gt"], iou_thr)
                    tp, fp, fn = tp + a, fp + b, fn + c
                    keep_f = keep & r["in_field"]
                    a2, b2, _ = greedy_match(
                        r["pred"][keep_f], r["scores"][keep_f], r["gt"], iou_thr
                    )
                    tp_f, fp_f = tp_f + a2, fp_f + b2
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                results["metrics"][f"s{score_thr}_iou{iou_thr}_{subset}"] = {
                    "n_frames": n, "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                    "precision_in_field": round(tp_f / max(tp_f + fp_f, 1), 4),
                }

    gt_counts = [len(r["gt"]) for r in per_frame]
    pred_counts = [int((r["scores"] >= 0.3).sum()) for r in per_frame]
    results["gt_per_frame_median"] = float(np.median(gt_counts)) if gt_counts else 0.0
    results["pred_per_frame_median"] = float(np.median(pred_counts)) if pred_counts else 0.0
    return results


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    game = args.game.split("_")[1]
    video = f"{VIDEO_DIR}/game_{game}.mp4"

    gt = load_gt(VAL_H5, args.game)
    windows = pick_windows(gt, args.windows, args.window_frames, args.seed)
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 72)
    print("FOOTPASS in-domain player-detection eval")
    print("=" * 72)
    print(f"game:     {args.game}  ({video}, {w}x{h})")
    print(f"GT:       {len(gt)} frames with >=1 visible player")
    print(f"windows:  {len(windows)} x {args.window_frames} consecutive frames")
    print(f"starts:   {windows}")
    med_gt = np.median([len(v) for v in gt.values()])
    print(f"GT players/frame (whole half): median {med_gt:.0f}")
    print()

    # A GT-only overlay first: if these green boxes don't sit on players, the
    # frame indexing is wrong and every number below is meaningless.
    frames = read_window(cap, windows[0], 1)
    if frames:
        idx, bgr = frames[0]
        save_overlay(bgr, gt[idx], np.zeros((0, 4)), np.zeros(0),
                     f"{OUT_DIR}/overlay_GT_ONLY_f{idx}.jpg")
        print(f"alignment check -> {OUT_DIR}/overlay_GT_ONLY_f{idx}.jpg\n")

    all_results = []
    for name in [s.strip() for s in args.arms.split(",") if s.strip()]:
        print(f"--- {name} ---")
        try:
            arm = RFDetrArm()
        except Exception as exc:
            print(f"  !! could not build {name}: {type(exc).__name__}: {exc}")
            continue
        res = run_arm(arm, windows, cap, gt, OUT_DIR, name)
        res["detector"] = name
        all_results.append(res)
        if hasattr(arm, "tracker"):
            arm.tracker.close()
        del arm
        import torch

        torch.cuda.empty_cache()

    cap.release()

    out_json = f"{OUT_DIR}/results_{args.game}.json"
    with open(out_json, "w") as f:
        json.dump({"game": args.game, "windows": windows,
                   "window_frames": args.window_frames, "results": all_results}, f, indent=2)

    print("\n" + "=" * 72)
    print("RESULTS  (IoU 0.5, score 0.3)")
    print("=" * 72)
    print(f"{'arm':<14}{'subset':<7}{'P':>7}{'R':>7}{'F1':>7}{'P(field)':>10}"
          f"{'GT/fr':>8}{'pred/fr':>9}{'s/fr':>7}")
    for r in all_results:
        for subset in ("all", "wide"):
            m = r["metrics"].get(f"s0.3_iou0.5_{subset}")
            if not m:
                continue
            print(f"{r['tag']:<14}{subset:<7}{m['precision']:>7.3f}{m['recall']:>7.3f}"
                  f"{m['f1']:>7.3f}{m['precision_in_field']:>10.3f}"
                  f"{r['gt_per_frame_median']:>8.0f}{r['pred_per_frame_median']:>9.0f}"
                  f"{r['sec_per_frame']:>7.2f}")
    print("=" * 72)
    print(f"full sweep (score x IoU x subset) -> {out_json}")
    print(f"overlays -> {OUT_DIR}/overlay_*.jpg")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="game_24_H1", help="val h5 key")
    p.add_argument("--windows", type=int, default=10)
    p.add_argument("--window-frames", type=int, default=40)
    p.add_argument("--arms", default="rfdetr", help="comma list of detectors")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main()
