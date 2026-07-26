"""Validate SAM3 text-prompt concept segmentation + native tracking on our match.

The user's proposal: prompt SAM3 with "soccer player" and let its video model
detect + track every player across frames — replacing the broken SAM-v1
segment-everything hack (detection/sam2.py) and the bolted-on ByteTrack.

This is the cheap domain-shift gate BEFORE any full run. On a short live-action
clip it reports, per frame, how many players the text prompt catches, and across
the clip how stable the track IDs are (the flicker problem). Saves 3 annotated
frames for eyeballing. Loads fully offline from the cached facebook/sam3 weights.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

VIDEO = "/orange/ewhite/b.weinstein/soccer-video-analysis/data/SaintsU11_OVF_Jul192026.MP4"
OUT_DIR = "/orange/ewhite/b.weinstein/soccer-video-analysis/runs/sam3_validation"
MODEL_ID = "facebook/sam3"
PROMPT = "soccer player"      # fallback "person" if this under-detects
START_FRAME = 15000          # mid-match live play
N_FRAMES = 48                # ~1.6s @ 30fps of consecutive frames -> real tracking test
STRIDE = 1                   # consecutive frames so tracking has continuity


def load_clip() -> list[np.ndarray]:
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    frames = []
    for _ in range(N_FRAMES * STRIDE):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    return frames[::STRIDE]


def to_mask_array(m, h: int, w: int) -> np.ndarray:
    """Coerce a per-object mask (tensor/ndarray, possibly logits) to bool HxW."""
    if isinstance(m, torch.Tensor):
        m = m.detach().float().cpu().numpy()
    m = np.asarray(m).squeeze()
    if m.ndim != 2:
        m = m.reshape(m.shape[-2], m.shape[-1])
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    # scores may be logits (>1) or probs; threshold sensibly either way
    thr = 0.0 if m.max() > 1.0 else 0.5
    return m > thr


def annotate(bgr: np.ndarray, out: "Sam3VideoSegmentationOutput", path: str) -> int:
    h, w = bgr.shape[:2]
    rng = np.random.default_rng(0)
    canvas = bgr.copy()
    n = 0
    for oid in out.object_ids:
        oid = int(oid)
        mask = to_mask_array(out.obj_id_to_mask[oid], h, w)
        if mask.sum() == 0:
            continue
        n += 1
        color = rng.integers(60, 255, size=3).tolist()
        canvas[mask] = (0.5 * canvas[mask] + 0.5 * np.array(color)).astype(np.uint8)
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(canvas, str(oid), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, canvas)
    return n


def main() -> None:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype} prompt={PROMPT!r}")

    frames = load_clip()
    if not frames:
        sys.exit("ERROR: no frames read from video")
    h, w = frames[0].shape[:2]
    rgb = [f[:, :, ::-1] for f in frames]
    print(f"clip: {len(frames)} frames @ {w}x{h}, from frame {START_FRAME}")

    processor = Sam3VideoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = Sam3VideoModel.from_pretrained(
        MODEL_ID, local_files_only=True, dtype=dtype
    ).to(device).eval()

    session = processor.init_video_session(
        video=rgb, inference_device=device, dtype=dtype
    )
    processor.add_text_prompt(session, PROMPT)

    per_frame_counts = []
    id_frame_count: dict[int, int] = {}
    save_at = {0: f"{OUT_DIR}/frame_first.jpg",
               len(frames) // 2: f"{OUT_DIR}/frame_mid.jpg",
               len(frames) - 1: f"{OUT_DIR}/frame_last.jpg"}

    with torch.inference_mode():
        for i in range(len(frames)):
            out = model(inference_session=session, frame_idx=i)
            ids = [int(o) for o in out.object_ids]
            per_frame_counts.append(len(ids))
            for oid in ids:
                id_frame_count[oid] = id_frame_count.get(oid, 0) + 1
            if i in save_at:
                drawn = annotate(frames[i], out, save_at[i])
                print(f"  frame {i}: {len(ids)} players -> {save_at[i]} ({drawn} drawn)")

    counts = np.array(per_frame_counts)
    n_total_ids = len(id_frame_count)
    stable = sum(1 for c in id_frame_count.values() if c >= 0.5 * len(frames))
    print("\n" + "=" * 60)
    print("SAM3 TEXT-PROMPT VALIDATION")
    print("=" * 60)
    print(f"prompt:                  {PROMPT!r}")
    print(f"players/frame:           min {counts.min()}  median {int(np.median(counts))}  max {counts.max()}")
    print(f"distinct track IDs:      {n_total_ids}")
    print(f"stable tracks (>=50% frames): {stable}")
    print(f"annotated frames:        {OUT_DIR}/frame_{{first,mid,last}}.jpg")
    print("=" * 60)
    if counts.max() >= 12:
        print("VERDICT: text prompt finds a full team's worth of players -> SAM3 pathway viable")
    else:
        print(f"VERDICT: under-detecting (max {counts.max()}); try prompt 'person' or lower threshold")


if __name__ == "__main__":
    main()
