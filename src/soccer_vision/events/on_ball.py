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
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OnBallSpan:
    track_id: int          # the target lane that was on the ball in this span
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    n_samples: int         # on-ball samples supporting the span
    min_dist_px: float     # closest the player got to the ball (px)


def _foot_point(bbox) -> tuple[float, float]:
    """Bottom-centre of a bbox — where the player meets the ground."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def select_on_ball_spans(
    ball_track: dict,
    tracks: dict,
    target_ids: set[int],
    *,
    max_ball_dist_px: float = 200.0,
    max_gap_s: float = 0.8,
    min_span_s: float = 0.4,
) -> list[OnBallSpan]:
    """Frames where a ``target_ids`` lane is the ball's nearest player.

    ``ball_track`` / ``tracks`` are the parsed ``ball_track.json`` /
    ``tracks.json`` from a run. For each frame the ball is visible, the nearest
    player (by foot point) is found; if it belongs to ``target_ids`` and is
    within ``max_ball_dist_px``, that frame is an on-ball sample. Consecutive
    samples (bridging gaps up to ``max_gap_s``) merge into spans; spans shorter
    than ``min_span_s`` are dropped as incidental.
    """
    fps = ball_track.get("fps") or tracks.get("fps") or 30.0

    # index every track's bbox by frame, once
    boxes_by_frame: dict[int, list[tuple[int, list]]] = {}
    for tid_s, samples in tracks.get("tracks", {}).items():
        tid = int(tid_s)
        for s in samples:
            boxes_by_frame.setdefault(int(s["frame"]), []).append((tid, s["bbox"]))

    # per-frame: is a target lane the nearest player, and how close?
    hits: list[tuple[int, int, float]] = []  # (frame, target_tid, dist)
    for bs in ball_track.get("samples", []):
        if not bs.get("visible"):
            continue
        bx, by = bs["pixel_x"], bs["pixel_y"]
        players = boxes_by_frame.get(int(bs["frame"]))
        if not players:
            continue
        nearest_tid, nearest_d = None, float("inf")
        for tid, bbox in players:
            fx, fy = _foot_point(bbox)
            d = ((fx - bx) ** 2 + (fy - by) ** 2) ** 0.5
            if d < nearest_d:
                nearest_tid, nearest_d = tid, d
        if nearest_tid in target_ids and nearest_d <= max_ball_dist_px:
            hits.append((int(bs["frame"]), nearest_tid, nearest_d))

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
            cur["min_d"] = min(cur["min_d"], dist)
            # keep the span tagged with the lane that got closest to the ball
            if dist <= cur["min_d"]:
                cur["tid"] = tid
        else:
            if cur:
                spans.append(cur)
            cur = {"tid": tid, "first": frame, "last": frame, "n": 1, "min_d": dist}
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
        ))
    return out
