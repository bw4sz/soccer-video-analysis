"""Select the spans where a target player is *on the ball*.

"Actions by number 6" means the moments #6 is the player acting on the ball —
i.e. #6's track is the closest player to the ball and close enough for it to be
their touch. This is computed in pixel space from the persisted ball track and
player tracks, so it needs neither the (degenerate on overhead footage) field
homography nor the set-piece event detector — both of which are unreliable here.

A player is usually fragmented across several track lanes over a match, and a
jersey number resolves to a *set* of lanes (see identify/resolve.py). At any one
frame the on-ball lane is a single specific id, so each returned span records the
lane that was actually on the ball — which is also the lane the halo should spot.

:func:`spans_to_events` renders the spans as ordinary event dicts so `extract`
and `reel` can cut them through exactly the same path as detector events. This
is what backs the ``--player`` fallback: the set-piece detector fires a handful
of times a match, so asking for one player's clips and getting nothing back is
the common case, and proximity to the ball is a far denser signal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OnBallSpan:
    track_id: int          # the lane that got closest to the ball in this span
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    n_samples: int         # on-ball samples supporting the span
    min_dist_px: float     # closest the player got to the ball (px)
    # Every target lane on the ball during the span, in order of first
    # appearance. A player fragments across lanes mid-dribble, so a single
    # continuous touch can span a handoff — splitting there would cut one action
    # into two clips, so the span stays whole and carries all its lanes. The
    # halo follows all of them; ``track_id`` is just the representative.
    track_ids: tuple[int, ...] = ()


def _foot_point(bbox) -> tuple[float, float]:
    """Bottom-centre of a bbox — where the player meets the ground."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def select_on_ball_spans(
    ball_track: dict,
    tracks: dict,
    target_ids: set[int],
    *,
    max_ball_dist_px: float = 90.0,
    max_gap_s: float = 0.8,
    min_span_s: float = 0.4,
) -> list[OnBallSpan]:
    """Frames where a ``target_ids`` lane is near the ball.

    ``ball_track`` / ``tracks`` are the parsed ``ball_track.json`` /
    ``tracks.json`` from a run. For each frame the ball is visible, a frame
    counts when a ``target_ids`` lane's foot point is within
    ``max_ball_dist_px`` of the ball — the target is *involved* near the ball,
    which captures both possession and defending/pressing that no event label
    covers. The distance is deliberately tight: on this footage 200px let in
    fly-bys where the player wasn't really in the play, so the default is close
    enough to read as an actual touch/challenge. Consecutive samples (bridging
    gaps up to ``max_gap_s``) merge into spans; spans shorter than ``min_span_s``
    are dropped as incidental.
    """
    fps = ball_track.get("fps") or tracks.get("fps") or 30.0

    # index every target lane's bbox by frame, once
    boxes_by_frame: dict[int, list[tuple[int, list]]] = {}
    for tid_s, samples in tracks.get("tracks", {}).items():
        tid = int(tid_s)
        if tid not in target_ids:
            continue
        for s in samples:
            boxes_by_frame.setdefault(int(s["frame"]), []).append((tid, s["bbox"]))

    # per-frame: the target lane closest to the ball, if within range
    hits: list[tuple[int, int, float]] = []  # (frame, target_tid, dist)
    for bs in ball_track.get("samples", []):
        if not bs.get("visible"):
            continue
        bx, by = bs["pixel_x"], bs["pixel_y"]
        players = boxes_by_frame.get(int(bs["frame"]))
        if not players:
            continue
        best_tid, best_d = None, float("inf")
        for tid, bbox in players:
            fx, fy = _foot_point(bbox)
            d = ((fx - bx) ** 2 + (fy - by) ** 2) ** 0.5
            if d < best_d:
                best_tid, best_d = tid, d
        if best_tid is not None and best_d <= max_ball_dist_px:
            hits.append((int(bs["frame"]), best_tid, best_d))

    if not hits:
        return []

    # merge consecutive hits (allowing short gaps) into spans
    gap_frames = max_gap_s * fps
    spans: list[OnBallSpan] = []
    cur = None
    for frame, tid, dist in hits:
        if cur and frame - cur["last"] <= gap_frames:
            cur["last"] = frame
            cur["n"] += 1
            # Tag the span with the lane that got closest to the ball. Compare
            # before updating min_d, or every sample ties its own new minimum.
            if dist < cur["min_d"]:
                cur["min_d"], cur["tid"] = dist, tid
            if tid not in cur["ids"]:
                cur["ids"].append(tid)
        else:
            if cur:
                spans.append(cur)
            cur = {"tid": tid, "first": frame, "last": frame, "n": 1,
                   "min_d": dist, "ids": [tid]}
    if cur:
        spans.append(cur)

    out = []
    for s in spans:
        start_s, end_s = s["first"] / fps, s["last"] / fps
        if end_s - start_s < min_span_s and s["n"] < 2:
            continue
        out.append(OnBallSpan(
            track_id=s["tid"], start_frame=s["first"], end_frame=s["last"],
            start_s=round(start_s, 2), end_s=round(end_s, 2),
            n_samples=s["n"], min_dist_px=round(s["min_d"], 1),
            track_ids=tuple(s["ids"]),
        ))
    return out


ON_BALL_LABEL = "on_ball"


def spans_to_events(
    spans: list[OnBallSpan],
    *,
    track_teams: dict | None = None,
    team: str | None = None,
) -> list[dict]:
    """Render on-ball spans as event dicts for the clip pipeline.

    The result is shaped like a detector event — ``label`` / ``frame`` /
    ``timestamp_s`` / ``track_id`` / ``team`` — so ``extract`` and ``reel`` cut
    it with no special-casing, and ``--halo`` spotlights the lane that was
    actually on the ball. ``end_s`` and ``duration_s`` are carried too, since a
    touch has a real duration where a detector event is a single instant.

    ``track_teams`` maps track id to kit colour (the ``teams`` block of
    ``tracks.json``); when ``team`` is given, spans belonging to another team are
    dropped. Spans whose lane has no team stay ``None`` and are dropped by an
    explicit ``team`` filter rather than guessed at.
    """
    teams = {int(k): v for k, v in (track_teams or {}).items()}
    events = []
    for s in spans:
        colour = teams.get(s.track_id)
        if team is not None and (colour or "").lower() != team.lower():
            continue
        events.append({
            "label": ON_BALL_LABEL,
            "frame": s.start_frame,
            "timestamp_s": s.start_s,
            "end_s": s.end_s,
            "duration_s": round(s.end_s - s.start_s, 2),
            "track_id": s.track_id,
            # Every lane on the ball in this span, so the halo covers the whole
            # clip even when the player changes lane partway through.
            "track_ids": list(s.track_ids or (s.track_id,)),
            "team": colour,
            "n_samples": s.n_samples,
            "min_dist_px": s.min_dist_px,
        })
    return events
