"""Composable event filtering for clip selection.

Shared by the ``extract`` and ``reel`` CLI commands so a user can cut clips for a
whole team, a single player (track), a single event label, or any combination —
independent of which detector produced the events.
"""

from __future__ import annotations


def filter_events(
    events: list[dict],
    *,
    label: str | None = None,
    team: str | None = None,
    track_id: int | None = None,
    track_ids: set[int] | None = None,
) -> list[dict]:
    """Return events matching all provided filters (AND). ``None`` = no filter.

    ``track_id`` selects a single raw lane; ``track_ids`` selects a *set* of
    lanes (used by the individual-player pathway, where one jersey number maps to
    many fragmented ByteTrack lanes). Both may be given — an event matches if its
    lane is in either.
    """
    out = events
    if label is not None:
        out = [e for e in out if e.get("label") == label]
    if team is not None:
        out = [e for e in out if (e.get("team") or "").lower() == team.lower()]
    if track_id is not None or track_ids is not None:
        allowed = set(track_ids or ())
        if track_id is not None:
            allowed.add(track_id)
        out = [e for e in out if e.get("track_id") in allowed]
    return out
