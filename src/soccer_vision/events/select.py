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
) -> list[dict]:
    """Return events matching all provided filters (AND). ``None`` = no filter."""
    out = events
    if label is not None:
        out = [e for e in out if e.get("label") == label]
    if team is not None:
        out = [e for e in out if (e.get("team") or "").lower() == team.lower()]
    if track_id is not None:
        out = [e for e in out if e.get("track_id") == track_id]
    return out
