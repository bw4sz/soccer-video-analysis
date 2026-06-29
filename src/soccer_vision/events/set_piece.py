"""Set-piece heuristics: goal kick, corner kick, throw-in detection."""

from __future__ import annotations

import math

from soccer_vision.registration.hough import BOX_DEPTH_M, FIELD_H_M, FIELD_W_M


def in_goal_zone(
    fx: float,
    fy: float,
    field_w: float = FIELD_W_M,
    field_h: float = FIELD_H_M,
    box_depth: float = BOX_DEPTH_M,
) -> str | None:
    """Return 'left', 'right', or None based on field position."""
    if 0 <= fy <= field_h:
        if 0 <= fx <= box_depth:
            return "left"
        if field_w - box_depth <= fx <= field_w:
            return "right"
    return None


def in_corner_zone(
    fx: float,
    fy: float,
    field_w: float = FIELD_W_M,
    field_h: float = FIELD_H_M,
    corner_radius: float = 2.0,
) -> bool:
    """Check if position is in a corner arc."""
    corners = [(0, 0), (field_w, 0), (0, field_h), (field_w, field_h)]
    return any(
        math.sqrt((fx - cx) ** 2 + (fy - cy) ** 2) <= corner_radius
        for cx, cy in corners
    )


def near_touchline(
    fx: float,
    fy: float,
    field_w: float = FIELD_W_M,
    field_h: float = FIELD_H_M,
    margin: float = 2.0,
) -> bool:
    """Check if position is near a touchline (for throw-in detection)."""
    return fy <= margin or fy >= field_h - margin


def is_stationary(positions: list[tuple[float, float]], thresh_px: float) -> bool:
    """Return True if all positions are within thresh_px of each other."""
    if len(positions) < 2:
        return False
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return math.sqrt(dx * dx + dy * dy) < thresh_px


def detect_goal_kicks(
    ball_positions: list[dict],
    *,
    stationary_frames: int = 3,
    stationary_px: float = 40.0,
    dedup_window_s: float = 5.0,
) -> list[dict]:
    """Detect goal kicks from a sequence of ball position records.

    Each record: {"frame": int, "timestamp_s": float, "field_x": float, "field_y": float,
                  "pixel_x": float, "pixel_y": float, "confidence": float}

    Returns list of detected goal kick events.
    """
    recent_px: list[tuple[float, float]] = []
    candidates = []

    for pos in ball_positions:
        recent_px.append((pos["pixel_x"], pos["pixel_y"]))
        if len(recent_px) > stationary_frames + 2:
            recent_px.pop(0)

        if len(recent_px) < stationary_frames:
            continue

        window = recent_px[-stationary_frames:]
        if not is_stationary(window, stationary_px):
            continue

        fx, fy = pos.get("field_x"), pos.get("field_y")
        if fx is None or fy is None:
            continue

        zone = in_goal_zone(fx, fy)
        if zone is None:
            continue

        ts = pos["timestamp_s"]
        if candidates and abs(ts - candidates[-1]["timestamp_s"]) < dedup_window_s:
            continue

        candidates.append({
            "label": "goal_kick",
            "frame": pos["frame"],
            "timestamp_s": ts,
            "position_ms": int(ts * 1000),
            "confidence": pos.get("confidence", 0.5),
            "goal_zone": zone,
            "field_x": fx,
            "field_y": fy,
        })

    return candidates


def detect_corner_kicks(
    ball_positions: list[dict],
    *,
    stationary_frames: int = 3,
    stationary_px: float = 40.0,
    dedup_window_s: float = 5.0,
) -> list[dict]:
    """Detect corner kicks from ball positions near corner arcs."""
    recent_px: list[tuple[float, float]] = []
    candidates = []

    for pos in ball_positions:
        recent_px.append((pos["pixel_x"], pos["pixel_y"]))
        if len(recent_px) > stationary_frames + 2:
            recent_px.pop(0)

        if len(recent_px) < stationary_frames:
            continue

        window = recent_px[-stationary_frames:]
        if not is_stationary(window, stationary_px):
            continue

        fx, fy = pos.get("field_x"), pos.get("field_y")
        if fx is None or fy is None:
            continue

        if not in_corner_zone(fx, fy):
            continue

        ts = pos["timestamp_s"]
        if candidates and abs(ts - candidates[-1]["timestamp_s"]) < dedup_window_s:
            continue

        candidates.append({
            "label": "corner_kick",
            "frame": pos["frame"],
            "timestamp_s": ts,
            "position_ms": int(ts * 1000),
            "confidence": pos.get("confidence", 0.5),
            "field_x": fx,
            "field_y": fy,
        })

    return candidates


def detect_throw_ins(
    ball_positions: list[dict],
    *,
    stationary_frames: int = 3,
    stationary_px: float = 40.0,
    dedup_window_s: float = 5.0,
) -> list[dict]:
    """Detect throw-ins from ball positions near touchlines."""
    recent_px: list[tuple[float, float]] = []
    candidates = []

    for pos in ball_positions:
        recent_px.append((pos["pixel_x"], pos["pixel_y"]))
        if len(recent_px) > stationary_frames + 2:
            recent_px.pop(0)

        if len(recent_px) < stationary_frames:
            continue

        window = recent_px[-stationary_frames:]
        if not is_stationary(window, stationary_px):
            continue

        fx, fy = pos.get("field_x"), pos.get("field_y")
        if fx is None or fy is None:
            continue

        if not near_touchline(fx, fy):
            continue

        # Exclude corners (those are corner kicks)
        if in_corner_zone(fx, fy):
            continue

        ts = pos["timestamp_s"]
        if candidates and abs(ts - candidates[-1]["timestamp_s"]) < dedup_window_s:
            continue

        candidates.append({
            "label": "throw_in",
            "frame": pos["frame"],
            "timestamp_s": ts,
            "position_ms": int(ts * 1000),
            "confidence": pos.get("confidence", 0.4),
            "field_x": fx,
            "field_y": fy,
        })

    return candidates


def detect_all_set_pieces(
    ball_positions: list[dict],
    **kwargs,
) -> list[dict]:
    """Run all set-piece heuristics and return merged, time-sorted events."""
    events = []
    events.extend(detect_goal_kicks(ball_positions, **kwargs))
    events.extend(detect_corner_kicks(ball_positions, **kwargs))
    events.extend(detect_throw_ins(ball_positions, **kwargs))
    events.sort(key=lambda e: e["timestamp_s"])
    return events
