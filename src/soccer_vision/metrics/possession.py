"""Possession percentage from ball proximity to players."""

from __future__ import annotations

import math


def assign_possession_frame(
    ball_pos: tuple[float, float],
    player_positions: dict[int, tuple[float, float]],
    player_teams: dict[int, str],
    max_distance_m: float = 3.0,
) -> str | None:
    """Assign ball possession to a team for a single frame.

    Returns team name, or None if ball is not near any player.
    """
    nearest_id = None
    nearest_dist = float("inf")

    for pid, pos in player_positions.items():
        dx = ball_pos[0] - pos[0]
        dy = ball_pos[1] - pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_id = pid

    if nearest_id is not None and nearest_dist <= max_distance_m:
        return player_teams.get(nearest_id)
    return None


def compute_possession(
    frames: list[dict],
    max_distance_m: float = 3.0,
) -> dict[str, float]:
    """Compute possession percentages from frame-level data.

    Each frame dict: {"ball": (x, y), "players": {id: (x, y)}, "teams": {id: team_name}}

    Returns {team_name: percentage}.
    """
    counts: dict[str, int] = {}
    total = 0

    for frame in frames:
        ball = frame.get("ball")
        if ball is None:
            continue
        team = assign_possession_frame(
            ball, frame["players"], frame["teams"], max_distance_m
        )
        if team is not None:
            counts[team] = counts.get(team, 0) + 1
            total += 1

    if total == 0:
        return {}
    return {team: round(count / total * 100, 1) for team, count in counts.items()}
