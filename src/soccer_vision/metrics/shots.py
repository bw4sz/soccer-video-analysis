"""Shot metrics.

This module used to also hold ``is_shot_toward_goal``, which inferred a shot from
ball speed and heading toward a goal mouth in field metres. It was deleted along
with field registration (see :mod:`soccer_vision.pitch`): nothing produces metres
any more, so it was unreachable, and a metric shot test is the wrong shape for
this footage regardless. Shots should come from an action-detection engine.
"""

from __future__ import annotations


def detect_shots_from_events(events: list[dict]) -> list[dict]:
    """Filter spotting model 'shot' events."""
    return [e for e in events if e.get("label") == "shot"]
