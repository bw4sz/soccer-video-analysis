"""Contact sheet generation for human/Claude review."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from soccer_vision.io.video import VideoReader

THUMB_W, THUMB_H = 640, 360
THUMBS_PER_SHEET = 6
SHEET_COLS = 2


def build_contact_sheet(
    video_path: str | Path,
    frame_data: list[dict],
    out_dir: str | Path,
) -> list[Path]:
    """Build contact sheets from frame data.

    Each frame_data dict: {"frame": int, "timestamp_s": float, ...}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with VideoReader(video_path) as reader:
        thumbs = []
        blank = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)

        for d in frame_data:
            fn = d["frame"]
            frame = reader.read_frame(fn)
            if frame is None:
                thumbs.append(blank.copy())
                continue

            thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
            ts = d["timestamp_s"]
            m, s = divmod(ts, 60)
            hms = f"{int(m):02d}:{s:04.1f}"
            label = d.get("label", "")

            cv2.putText(thumb, f"F{fn}", (6, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(thumb, f"{hms} {label}", (6, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            if d.get("ball_px") and d.get("ball_py"):
                bx = int(d["ball_px"] * THUMB_W / frame.shape[1])
                by = int(d["ball_py"] * THUMB_H / frame.shape[0])
                cv2.circle(thumb, (bx, by), 8, (0, 0, 255), 2)

            thumbs.append(thumb)

    rows_per_sheet = THUMBS_PER_SHEET // SHEET_COLS
    n_sheets = max(1, math.ceil(len(thumbs) / THUMBS_PER_SHEET))
    sheet_paths = []

    for s_idx in range(n_sheets):
        chunk = thumbs[s_idx * THUMBS_PER_SHEET : (s_idx + 1) * THUMBS_PER_SHEET]
        while len(chunk) < THUMBS_PER_SHEET:
            chunk.append(blank.copy())
        rows = [
            np.hstack(chunk[r * SHEET_COLS : (r + 1) * SHEET_COLS])
            for r in range(rows_per_sheet)
        ]
        sheet = np.vstack(rows)
        path = out_dir / f"sheet_{s_idx + 1:03d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sheet_paths.append(path)

    return sheet_paths


def sample_and_build_sheets(
    video_path: str | Path,
    out_dir: str | Path,
    interval: int = 500,
) -> tuple[list[dict], list[Path]]:
    """Sample frames at interval and build contact sheets."""
    out_dir = Path(out_dir)
    index = []

    with VideoReader(video_path) as reader:
        for fn, _ in reader.sample_frames(interval):
            ts = fn / reader.fps
            m, s = divmod(ts, 60)
            index.append({
                "frame": fn,
                "timestamp_s": round(ts, 2),
                "timestamp_hms": f"{int(m):02d}:{s:04.1f}",
            })

    sheets = build_contact_sheet(video_path, index, out_dir)
    return index, sheets
