"""
detect_actions.py

Sample frames from a match video at a regular interval, resize them to
thumbnails, and tile them into contact sheets for visual review by Claude.

No computer vision heuristics — Claude looks at the sheets and identifies
action candidates directly.

Usage:
    python detect_actions.py --video match.mp4
    python detect_actions.py --video match.mp4 --effort high
    python detect_actions.py --video match.mp4 --effort low --out-dir my_sheets/

Effort levels (for a 60-min / ~108k-frame video at 30fps):
    low    -- every 1000 frames (~33s gaps) →  ~108 thumbs, 4 sheets
    medium -- every  500 frames (~17s gaps) →  ~216 thumbs, 8 sheets   [default]
    high   -- every  250 frames ( ~8s gaps) →  ~432 thumbs, 16 sheets

Outputs:
    sheet_001.jpg, sheet_002.jpg, ...   Contact sheets (30 thumbs each)
    index.json                          Frame number + timestamp for every sample
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Effort presets:  (sample_interval_frames, description)
# ---------------------------------------------------------------------------
EFFORT_PRESETS = {
    "low":    (1000, "~33s gaps — fast, ~4 sheets for a 60-min match"),
    "medium": ( 500, "~17s gaps — default, ~8 sheets for a 60-min match"),
    "high":   ( 250, "~8s gaps  — thorough, ~16 sheets for a 60-min match"),
}

# Veo footage is a fixed ultra-wide panorama: the whole pitch (plus adjacent
# tournament fields) is in frame, so the goal mouth and ball are tiny. Small
# thumbs tiled densely lose that detail twice over — once in the downscale to
# 320px, and again because Claude shrinks any sheet to <=1568px / ~1.15MP. Use
# larger thumbs, few per sheet, so each one survives at readable resolution.
THUMB_W, THUMB_H = 640, 360   # 2x linear over the old 320x180 — goal setup legible
THUMBS_PER_SHEET = 6          # 2 columns × 3 rows → 1280x1080 sheet (minimal downscale)
SHEET_COLS = 2


def sample_frames(video_path: str, interval: int, out_dir: Path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_min = total / fps / 60
    print(f"  {total} frames @ {fps:.2f} fps ({duration_min:.1f} min)")

    frame_numbers = list(range(0, total, interval))
    print(f"  Sampling {len(frame_numbers)} frames (every {interval} frames)")

    index = []
    thumbs = []

    for fn in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue

        ts = fn / fps
        m, s = divmod(ts, 60)
        hms = f"{int(m):02d}:{s:04.1f}"

        thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
        # Label: frame number and timestamp
        cv2.putText(thumb, f"F{fn}", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(thumb, hms, (6, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        thumbs.append(thumb)
        index.append({"frame": fn, "timestamp_s": round(ts, 2), "timestamp_hms": hms})

    cap.release()

    # Tile into sheets
    blank = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    rows_per_sheet = THUMBS_PER_SHEET // SHEET_COLS
    n_sheets = math.ceil(len(thumbs) / THUMBS_PER_SHEET)

    sheet_paths = []
    for s_idx in range(n_sheets):
        chunk = thumbs[s_idx * THUMBS_PER_SHEET:(s_idx + 1) * THUMBS_PER_SHEET]
        # Pad the final partial sheet up to a full grid so every row has content
        # (hstack on an empty row slice would otherwise raise).
        while len(chunk) < THUMBS_PER_SHEET:
            chunk.append(blank)
        rows = [np.hstack(chunk[r * SHEET_COLS:(r + 1) * SHEET_COLS])
                for r in range(rows_per_sheet)]
        sheet = np.vstack(rows)
        path = out_dir / f"sheet_{s_idx + 1:03d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sheet_paths.append(path)
        print(f"  Saved {path.name}")

    return index, sheet_paths, fps


def main():
    args = parse_args()

    interval, desc = EFFORT_PRESETS[args.effort]
    if args.sample_interval:
        interval = args.sample_interval
        desc = f"custom ({interval} frames)"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Effort: {args.effort} — {desc}")
    print(f"Video:  {args.video}")

    index, sheet_paths, fps = sample_frames(args.video, interval, out_dir)

    meta = {
        "video": args.video,
        "fps": fps,
        "effort": args.effort,
        "sample_interval": interval,
        "total_samples": len(index),
        "sheets": [str(p) for p in sheet_paths],
        "frames": index,
    }
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {len(index)} samples → {len(sheet_paths)} sheets")
    print(f"Index: {index_path}")
    print(f"\nNext: show the sheet(s) to Claude and ask it to identify goal kicks.")
    print(f"Then run: python extract_clips.py --video {args.video} "
          f"--index {index_path} --timestamps <F1> <F2> ...")


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample a match video into contact sheets for Claude review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:8s} {v[1]}" for k, v in EFFORT_PRESETS.items()),
    )
    p.add_argument("--video", required=True)
    p.add_argument("--effort", choices=list(EFFORT_PRESETS), default="medium")
    p.add_argument("--sample-interval", type=int, default=None,
                   help="Override effort: sample every N frames")
    p.add_argument("--out-dir", default="sheets",
                   help="Directory for output sheets and index (default: sheets/)")
    return p.parse_args()


if __name__ == "__main__":
    main()
