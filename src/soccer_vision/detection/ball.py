"""Ball-specific detection with RF-DETR fallback chain."""

from __future__ import annotations

import numpy as np

from soccer_vision.detection.rfdetr import RFDETRSoccerDetector


def ball_position_from(dets) -> tuple[float, float, float] | None:
    """(cx, cy, confidence) of the best ball in an already-computed detection set.

    Split out from :func:`detect_ball_position` so a caller that has *already*
    run the detector on this frame — `process` detects people and the ball in one
    forward — can pick the ball out without paying a second pass.
    """
    if dets is None or len(dets) == 0 or dets.confidence is None:
        return None

    best_idx = int(np.argmax(dets.confidence))
    x1, y1, x2, y2 = dets.xyxy[best_idx]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return float(cx), float(cy), float(dets.confidence[best_idx])


def detect_ball_position(
    frame: np.ndarray,
    detector: RFDETRSoccerDetector,
    conf_threshold: float = 0.2,
) -> tuple[float, float, float] | None:
    """Return (cx, cy, confidence) of the highest-confidence ball, or None.

    For callers that only want the ball (``trim-empty``). If you also need the
    people, use :meth:`RFDETRSoccerDetector.predict_split` and pass its ball
    detections to :func:`ball_position_from`.
    """
    return ball_position_from(detector.predict_ball(frame, conf_threshold=conf_threshold))


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
