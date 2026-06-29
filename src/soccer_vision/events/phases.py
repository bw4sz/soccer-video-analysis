"""In-play vs dead-ball phase detection (rule-based)."""

from __future__ import annotations


def classify_phase(events: list[dict]) -> list[dict]:
    """Annotate events with phase: 'dead_ball' for set pieces, 'in_play' otherwise.

    Returns the same events list with a 'phase' key added.
    """
    dead_ball_labels = {
        "goal_kick", "corner_kick", "throw_in", "free_kick",
        "kickoff", "penalty", "substitution", "halftime",
    }
    for event in events:
        event["phase"] = "dead_ball" if event["label"] in dead_ball_labels else "in_play"
    return events
