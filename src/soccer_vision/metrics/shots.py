"""Shot detection from ball trajectory toward goal mouth."""

from __future__ import annotations

import math

from soccer_vision.registration.hough import FIELD_H_M, FIELD_W_M

# Goal mouth: centered on each goal line, ~5m wide for 7v7
GOAL_MOUTH_WIDTH_M = 5.0


def is_shot_toward_goal(
    ball_positions: list[tuple[float, float]],
    fps: float,
    speed_threshold_ms: float = 8.0,
    field_w: float = FIELD_W_M,
    field_h: float = FIELD_H_M,
    goal_width: float = GOAL_MOUTH_WIDTH_M,
) -> bool:
    """Check if the ball trajectory heads toward a goal mouth at speed."""
    if len(ball_positions) < 2:
        return False

    p0 = ball_positions[-2]
    p1 = ball_positions[-1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    speed = math.sqrt(dx * dx + dy * dy) * fps

    if speed < speed_threshold_ms:
        return False

    goal_y_center = field_h / 2
    goal_y_min = goal_y_center - goal_width / 2
    goal_y_max = goal_y_center + goal_width / 2

    # Moving toward left goal (x decreasing toward 0)
    if dx < 0 and p1[0] < field_w * 0.3:
        if goal_y_min <= p1[1] <= goal_y_max:
            return True

    # Moving toward right goal (x increasing toward field_w)
    if dx > 0 and p1[0] > field_w * 0.7:
        if goal_y_min <= p1[1] <= goal_y_max:
            return True

    return False


def detect_shots_from_events(events: list[dict]) -> list[dict]:
    """Filter spotting model 'shot' events."""
    return [e for e in events if e.get("label") == "shot"]
