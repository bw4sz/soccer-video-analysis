"""Associate events with the nearest player track and their team.

Given events (from any :mod:`soccer_vision.events.sources` source) and the
per-frame player positions collected during tracking, tag each event with the
``track_id`` of the closest player and their ``team`` colour. This is what turns
an event stream into something the clip-selection CLI can filter by player or by
team.

The nearest-player choice mirrors
:func:`soccer_vision.metrics.possession.assign_possession_frame`, but returns the
track id (a v1 stand-in for player identity) rather than only a team.

Everything here is **pixel space**. There used to be a field-metres path tried
first, which was actively harmful: the metres came from a homography that failed
on this footage, so a bogus sub-threshold match beat the working pixel fallback
and named the wrong player. Field registration is gone (see
:mod:`soccer_vision.pitch`), and so is that branch.
"""

from __future__ import annotations

import math


def _nearest_track(
    point: tuple[float, float],
    players: dict[int, tuple[float, float]],
    max_distance: float,
) -> int | None:
    nearest_id, nearest_dist = None, float("inf")
    for tid, pos in players.items():
        d = math.hypot(point[0] - pos[0], point[1] - pos[1])
        if d < nearest_dist:
            nearest_id, nearest_dist = tid, d
    return nearest_id if nearest_dist <= max_distance else None


def _pixel_point(event: dict) -> tuple[float, float] | None:
    x, y = event.get("pixel_x"), event.get("pixel_y")
    return None if x is None or y is None else (x, y)


def _players_at(
    frame_players: dict[int, dict],
    frame: int,
    search_frames: int,
) -> dict[int, tuple[float, float]]:
    """Player pixel positions at ``frame`` (or the nearest frame within a window)."""
    xk, yk = "pixel_x", "pixel_y"

    candidates = [frame] + [
        frame + off
        for step in range(1, search_frames + 1)
        for off in (step, -step)
    ]
    for fn in candidates:
        players = frame_players.get(fn)
        if not players:
            continue
        out = {
            tid: (p[xk], p[yk])
            for tid, p in players.items()
            if p.get(xk) is not None and p.get(yk) is not None
        }
        if out:
            return out
    return {}


def associate_events(
    events: list[dict],
    frame_players: dict[int, dict],
    team_clf=None,
    *,
    max_distance_pixel: float = 120.0,
    search_frames: int = 3,
) -> list[dict]:
    """Tag each event in place with ``track_id`` and ``team``.

    ``frame_players`` maps frame number → ``{track_id: {pixel_x, pixel_y}}``.
    ``team_clf`` is a fitted
    :class:`soccer_vision.tracking.teams.TeamClassifier` (optional).
    """
    for event in events:
        event.setdefault("track_id", None)
        event.setdefault("team", None)
        if "frame" not in event:
            continue

        point = _pixel_point(event)
        if point is None:
            continue
        players = _players_at(frame_players, event["frame"], search_frames)
        if not players:
            continue
        tid = _nearest_track(point, players, max_distance_pixel)
        if tid is None:
            continue

        event["track_id"] = int(tid)
        if team_clf is not None:
            event["team"] = team_clf.predict(tid)

    return events


def stamp_event_positions(
    events: list[dict],
    ball_positions: list[dict],
    *,
    search_frames: int = 15,
) -> list[dict]:
    """Give each event a spatial anchor so it can be associated to a player.

    Some engines emit only ``frame``/``label``/``confidence`` — no coordinates —
    leaving :func:`associate_events` nothing to search from, so it tagged
    nothing: 0/8 events had a ``track_id`` in job 37878296, which is why no event
    could ever receive a team.

    For ball-driven events the ball *is* the event location, and the nearest
    player to it is the one performing the action, so we stamp the ball's pixel
    position at (or near) the event frame.
    """
    if not ball_positions:
        return events

    by_frame = {int(b["frame"]): b for b in ball_positions if b.get("frame") is not None}
    if not by_frame:
        return events
    frames = sorted(by_frame)

    for event in events:
        if event.get("pixel_x") is not None:
            continue  # already anchored in pixel space
        fn = event.get("frame")
        if fn is None:
            continue
        fn = int(fn)
        ball = by_frame.get(fn)
        if ball is None:
            near = [f for f in frames if abs(f - fn) <= search_frames]
            if near:
                ball = by_frame[min(near, key=lambda f: abs(f - fn))]
        if ball is None:
            continue
        if ball.get("pixel_x") is not None and ball.get("pixel_y") is not None:
            event["pixel_x"] = ball["pixel_x"]
            event["pixel_y"] = ball["pixel_y"]
    return events
