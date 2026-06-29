"""Video I/O helpers: loading, sampling, ffmpeg wrappers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Thin wrapper around cv2.VideoCapture with metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps

    def read_frame(self, frame_no: int) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        return frame if ret else None

    def sample_frames(self, interval: int, start: int = 0, end: int | None = None):
        """Yield (frame_no, frame) at the given interval."""
        end = end or self.total_frames
        for fn in range(start, end, interval):
            frame = self.read_frame(fn)
            if frame is not None:
                yield fn, frame

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def ffmpeg_run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print("ffmpeg stderr:", result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")
    return result


def ffmpeg_extract_clip(
    video: str | Path,
    start_s: float,
    duration_s: float,
    out_path: str | Path,
    reencode: bool = True,
):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0, start_s)),
        "-i", str(video),
        "-t", str(duration_s),
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(out_path))
    ffmpeg_run(cmd)


def ffmpeg_concat(clip_paths: list[Path], out_path: str | Path):
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_file = f.name

    try:
        ffmpeg_run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            str(out_path),
        ])
    finally:
        os.unlink(list_file)
