"""Virtual broadcast: follow-cam proxy generation from wide-angle footage.

Normalizes raw wide-angle single-camera footage into a 16:9 broadcast-style
proxy so downstream SoccerNet models see broadcast framing.

Algorithm:
  1. Decode video; downsample frames to 1080p for inference.
  2. Run RF-DETR at detect_fps for ball + players.
  3. Compute action centroid: ball position; fallback to player cluster centroid.
  4. Smooth crop window with exponential moving average.
  5. Render full-resolution crop to 16:9 proxy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from soccer_vision.detection.rfdetr import BALL_CLASS_IDS, PLAYER_CLASS_IDS, RFDETRSoccerDetector


@dataclass
class BroadcastConfig:
    output_width: int = 1920
    output_height: int = 1080
    smooth_window_s: float = 0.4
    min_zoom: float = 1.0
    max_zoom: float = 2.5
    detect_fps: float = 5.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> BroadcastConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        bc = data.get("broadcast", {})
        kwargs = {}
        if "output_resolution" in bc:
            kwargs["output_width"] = bc["output_resolution"][0]
            kwargs["output_height"] = bc["output_resolution"][1]
        for key in ("smooth_window_s", "min_zoom", "max_zoom", "detect_fps"):
            if key in bc:
                kwargs[key] = bc[key]
        return cls(**kwargs)


def _compute_action_centroid(
    detections,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float] | None:
    """Find action centroid: ball position, or player cluster center."""
    if len(detections) == 0:
        return None

    ball_mask = np.isin(detections.class_id, list(BALL_CLASS_IDS))
    if ball_mask.any():
        ball_boxes = detections.xyxy[ball_mask]
        cx = float(np.mean((ball_boxes[:, 0] + ball_boxes[:, 2]) / 2))
        cy = float(np.mean((ball_boxes[:, 1] + ball_boxes[:, 3]) / 2))
        return cx, cy

    player_mask = np.isin(detections.class_id, list(PLAYER_CLASS_IDS))
    if player_mask.any():
        player_boxes = detections.xyxy[player_mask]
        cx = float(np.mean((player_boxes[:, 0] + player_boxes[:, 2]) / 2))
        cy = float(np.mean((player_boxes[:, 1] + player_boxes[:, 3]) / 2))
        return cx, cy

    return None


def _smooth_position(
    current: tuple[float, float],
    target: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    """Exponential moving average smoothing."""
    return (
        current[0] + alpha * (target[0] - current[0]),
        current[1] + alpha * (target[1] - current[1]),
    )


def _compute_crop_rect(
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Compute clamped crop rectangle (x, y, w, h)."""
    x = int(center_x - crop_w / 2)
    y = int(center_y - crop_h / 2)
    x = max(0, min(x, frame_w - crop_w))
    y = max(0, min(y, frame_h - crop_h))
    return x, y, crop_w, crop_h


def generate_broadcast_proxy(
    video_path: str | Path,
    output_path: str | Path,
    config: BroadcastConfig | None = None,
    detector: RFDETRSoccerDetector | None = None,
    metadata_path: str | Path | None = None,
) -> Path:
    """Generate a broadcast-style 16:9 proxy from wide-angle footage.

    Returns the path to the generated proxy video.
    """
    config = config or BroadcastConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detect_interval = max(1, int(round(native_fps / config.detect_fps)))

    # Default zoom: fit output aspect in source frame
    aspect = config.output_width / config.output_height
    crop_w = frame_w
    crop_h = int(crop_w / aspect)
    if crop_h > frame_h:
        crop_h = frame_h
        crop_w = int(crop_h * aspect)

    # Smoothing alpha based on frame rate and smooth_window_s
    alpha = min(1.0, 1.0 / max(1, config.smooth_window_s * native_fps))

    # Load detector lazily
    if detector is None:
        detector = RFDETRSoccerDetector.from_pretrained()

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        native_fps,
        (config.output_width, config.output_height),
    )

    smooth_center = (frame_w / 2, frame_h / 2)
    crop_metadata = []

    for fn in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection at detect_fps interval
        if fn % detect_interval == 0:
            detections = detector.predict(frame)
            centroid = _compute_action_centroid(detections, frame_w, frame_h)
            if centroid is not None:
                smooth_center = _smooth_position(smooth_center, centroid, alpha)

        x, y, cw, ch = _compute_crop_rect(
            smooth_center[0], smooth_center[1],
            crop_w, crop_h, frame_w, frame_h,
        )
        cropped = frame[y : y + ch, x : x + cw]
        resized = cv2.resize(cropped, (config.output_width, config.output_height))
        writer.write(resized)

        crop_metadata.append({
            "frame": fn,
            "crop_x": x,
            "crop_y": y,
            "crop_w": cw,
            "crop_h": ch,
        })

        if fn % 500 == 0:
            pct = fn / total_frames * 100
            print(f"  Broadcast proxy: {pct:.0f}% ({fn}/{total_frames})")

    cap.release()
    writer.release()

    if metadata_path:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump({"frames": crop_metadata, "config": {
                "output_width": config.output_width,
                "output_height": config.output_height,
                "source_width": frame_w,
                "source_height": frame_h,
            }}, f)

    print(f"  Broadcast proxy saved: {output_path}")
    return output_path
