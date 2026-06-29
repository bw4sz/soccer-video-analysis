"""Ball-specific detection with RF-DETR fallback chain."""

from __future__ import annotations

import numpy as np

from soccer_vision.detection.rfdetr import RFDETRSoccerDetector


def detect_ball_position(
    frame: np.ndarray,
    detector: RFDETRSoccerDetector,
    conf_threshold: float = 0.2,
) -> tuple[float, float, float] | None:
    """Return (cx, cy, confidence) of the highest-confidence ball, or None."""
    dets = detector.predict_ball(frame, conf_threshold=conf_threshold)
    if len(dets) == 0:
        return None

    best_idx = int(np.argmax(dets.confidence))
    x1, y1, x2, y2 = dets.xyxy[best_idx]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return float(cx), float(cy), float(dets.confidence[best_idx])


def detect_ball_in_sequence(
    frames: list[tuple[int, np.ndarray]],
    detector: RFDETRSoccerDetector,
    conf_threshold: float = 0.2,
) -> list[tuple[int, float, float, float]]:
    """Run ball detection on a sequence of (frame_no, frame) pairs.

    Returns list of (frame_no, cx, cy, confidence) for frames where ball was found.
    """
    results = []
    for fn, frame in frames:
        pos = detect_ball_position(frame, detector, conf_threshold)
        if pos is not None:
            results.append((fn, *pos))
    return results
