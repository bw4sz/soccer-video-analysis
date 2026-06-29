"""Optional KpSFR subprocess adapter for field registration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def extract_frames_for_registration(
    video_path: str | Path,
    fps_target: float,
    out_dir: Path,
) -> tuple[list[tuple[int, float, str]], float]:
    """Extract frames at fps_target. Returns (entries, native_fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(round(native_fps / fps_target)))

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for fn in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        ts = fn / native_fps
        img_path = out_dir / f"frame_{fn:08d}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        entries.append((fn, ts, str(img_path)))
    cap.release()
    return entries, native_fps


def run_kpsfr(
    kpsfr_repo: Path,
    image_paths: list[str],
    out_dir: Path,
) -> dict[str, list | None]:
    """Run KpSFR inference. Returns {image_path: H_as_list_or_None}."""
    ckpt = kpsfr_repo / "checkpoint" / "kpsfr_finetuned.pth"
    if not ckpt.exists():
        ckpt = kpsfr_repo / "checkpoint" / "kpsfr.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"KpSFR checkpoint not found at {ckpt.parent}/")

    param_file = out_dir / "kpsfr_params.txt"
    param_lines = [
        f"--ckpt_path {ckpt}",
        "--train_stage 1",
        "--sfp_finetuned True",
        f"--target_image {' '.join(image_paths[:50])}",
        "--name register_out",
    ]
    param_file.write_text("\n".join(param_lines))

    inference_script = kpsfr_repo / "inference.py"
    result = subprocess.run(
        [sys.executable, str(inference_script), str(param_file)],
        cwd=str(kpsfr_repo),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"KpSFR failed: {result.stderr[-2000:]}")

    h_dir = kpsfr_repo / "checkpoint" / "register_out"
    homographies = {}
    for img_path in image_paths:
        stem = Path(img_path).stem
        h_file = h_dir / f"{stem}_homography.npy"
        if h_file.exists():
            H = np.load(str(h_file))
            homographies[img_path] = H.tolist()
        else:
            homographies[img_path] = None
    return homographies
