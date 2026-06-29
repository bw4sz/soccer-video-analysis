"""ByteTrack wrapper via supervision for multi-object tracking."""

from __future__ import annotations

import supervision as sv


def create_tracker(
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
    frame_rate: int = 30,
) -> sv.ByteTrack:
    return sv.ByteTrack(
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        frame_rate=frame_rate,
    )


def track_detections(
    tracker: sv.ByteTrack,
    detections: sv.Detections,
) -> sv.Detections:
    """Update tracker with new detections, return tracked detections with IDs."""
    return tracker.update_with_detections(detections)
