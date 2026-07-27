"""Filter detections down to on-field players.

A person prompt finds people anywhere in frame — coaches, subs, parents on the
touchline — so detections need a notion of "on the field" before they become
tracks.

This used to project each foot-point through a homography and keep the ones
landing inside the pitch rectangle. That branch is gone with field registration
(see :mod:`soccer_vision.pitch`): it never worked on this footage, and it failed
*destructively* rather than visibly. Measured in job 37877533, ``compute_homography``
returned ``ok=True`` with a garbage matrix, surviving detections went 20 → 0 and
stayed there, leaving 52 track-frames where ~6000 were expected. The guard added
to catch that — fall back to the hull when the homography rejects essentially
everyone — meant the hull was doing the work in practice, with the homography
contributing only the risk of a bad-but-not-bad-enough matrix quietly dropping
real players.

So the geometric hull is now the only path. It is crude (it assumes the field
occupies the middle of a wide-angle frame) and it is the honest state of the art
here until turf-mask segmentation lands, which would give a real field polygon
without needing lines or metres.
"""

from __future__ import annotations

import supervision as sv


def _box_foot_point(xyxy):
    """Bottom-center of each bounding box (foot position)."""
    import numpy as np

    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = xyxy[:, 3]  # bottom edge
    return np.column_stack([cx, cy])


def filter_by_field_hull(
    detections: sv.Detections,
    frame_shape: tuple[int, int],
    field_fraction: float = 0.7,
) -> sv.Detections:
    """Keep detections whose foot-point sits in the central field region.

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
    frame_shape: tuple[int, int],
    field_fraction: float = 0.7,
) -> sv.Detections:
    """Drop off-field detections (spectators, coaches, subs)."""
    return filter_by_field_hull(detections, frame_shape, field_fraction)
