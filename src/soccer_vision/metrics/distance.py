"""Distance covered per player from tracked field positions."""

from __future__ import annotations

import math


def distance_covered(
    track_positions: list[tuple[float, float]],
) -> float:
    """Sum of Euclidean distances between consecutive field positions (metres)."""
    total = 0.0
    for i in range(1, len(track_positions)):
        dx = track_positions[i][0] - track_positions[i - 1][0]
        dy = track_positions[i][1] - track_positions[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def distance_per_player(
    tracks: dict[int, list[tuple[float, float]]],
) -> dict[int, float]:
    """Compute distance for each track_id → list of field (x, y) positions."""
    return {tid: distance_covered(positions) for tid, positions in tracks.items()}


def detect_sprints(
    positions: list[tuple[float, float]],
    fps: float,
    speed_threshold_ms: float = 5.0,
    min_duration_s: float = 1.0,
) -> list[tuple[int, int]]:
    """Find sprint segments where speed > threshold for >= min_duration.

    Returns list of (start_idx, end_idx) pairs.
    """
    min_frames = int(min_duration_s * fps)
    sprints = []
    sprint_start = None

    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        speed = math.sqrt(dx * dx + dy * dy) * fps

        if speed > speed_threshold_ms:
            if sprint_start is None:
                sprint_start = i - 1
        else:
            if sprint_start is not None and i - sprint_start >= min_frames:
                sprints.append((sprint_start, i - 1))
            sprint_start = None

    if sprint_start is not None and len(positions) - sprint_start >= min_frames:
        sprints.append((sprint_start, len(positions) - 1))

    return sprints
