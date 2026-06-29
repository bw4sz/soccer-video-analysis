"""2D histogram heatmaps of player field positions."""

from __future__ import annotations

import numpy as np

from soccer_vision.registration.hough import FIELD_H_M, FIELD_W_M


def compute_heatmap(
    positions: list[tuple[float, float]],
    field_w: float = FIELD_W_M,
    field_h: float = FIELD_H_M,
    bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D histogram from field positions.

    Returns (histogram, x_edges, y_edges).
    """
    if not positions:
        return np.zeros((bins, bins)), np.array([]), np.array([])

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    hist, x_edges, y_edges = np.histogram2d(
        xs, ys,
        bins=bins,
        range=[[0, field_w], [0, field_h]],
    )
    return hist, x_edges, y_edges


def heatmap_per_player(
    tracks: dict[int, list[tuple[float, float]]],
    **kwargs,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute heatmap for each tracked player."""
    return {tid: compute_heatmap(positions, **kwargs) for tid, positions in tracks.items()}
