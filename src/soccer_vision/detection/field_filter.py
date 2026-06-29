"""Filter detections to on-field players using homography or convex hull."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from soccer_vision.registration.hough import FIELD_H_M, FIELD_W_M


def _box_foot_point(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-center of each bounding box (foot position)."""
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = xyxy[:, 3]  # bottom edge
    return np.column_stack([cx, cy])


def filter_by_homography(
    detections: sv.Detections,
    H: np.ndarray,
    field_w: float = FIELD_W_M,
    field_h: float = FIELD_H_M,
    margin_m: float = 3.0,
) -> sv.Detections:
    """Keep only detections whose foot-point projects inside the field (with margin)."""
    if len(detections) == 0 or H is None:
        return detections

    feet = _box_foot_point(detections.xyxy)
    pts = feet.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    inside = (
        (projected[:, 0] >= -margin_m)
        & (projected[:, 0] <= field_w + margin_m)
        & (projected[:, 1] >= -margin_m)
        & (projected[:, 1] <= field_h + margin_m)
    )
    return detections[inside]


def filter_by_field_hull(
    detections: sv.Detections,
    frame_shape: tuple[int, int],
    field_fraction: float = 0.7,
) -> sv.Detections:
    """Fallback when homography fails: keep detections in the central field region.

    Assumes the field occupies roughly the center of the wide-angle frame.
    The field_fraction controls how much of the frame width/height is "field".
    """
    if len(detections) == 0:
        return detections

    h, w = frame_shape[:2]
    margin_x = w * (1 - field_fraction) / 2
    margin_y = h * (1 - field_fraction) / 2

    feet = _box_foot_point(detections.xyxy)
    inside = (
        (feet[:, 0] >= margin_x)
        & (feet[:, 0] <= w - margin_x)
        & (feet[:, 1] >= margin_y)
        & (feet[:, 1] <= h - margin_y)
    )
    return detections[inside]


def filter_spectators(
    detections: sv.Detections,
    H: np.ndarray | None,
    frame_shape: tuple[int, int],
    field_w: float = FIELD_W_M,
    field_h: float = FIELD_H_M,
) -> sv.Detections:
    """Filter spectator detections using homography if available, else hull fallback."""
    if H is not None:
        return filter_by_homography(detections, H, field_w, field_h)
    return filter_by_field_hull(detections, frame_shape)


