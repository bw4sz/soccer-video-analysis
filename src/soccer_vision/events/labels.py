"""Canonical soccer-vision event taxonomy.

Single source of truth for the event labels used across the pipeline. The labels
themselves are produced today by the set-piece heuristics
(:mod:`soccer_vision.events.set_piece`), the phase classifier
(:mod:`soccer_vision.events.phases`), and the SoccerNet mapping in
:mod:`soccer_vision.events.spotting`; this module consolidates them so the
Label Studio config (:mod:`soccer_vision.annotate.label_studio`) and the
SoccerChat verifier (:mod:`soccer_vision.verify.soccerchat`) stay in sync with
one list instead of hard-coding strings in three places.
"""

from __future__ import annotations

# Ordered so the Label Studio choice list groups sensibly for an annotator.
EVENT_LABELS: list[str] = [
    # Set pieces / restarts
    "goal_kick",
    "corner_kick",
    "throw_in",
    "free_kick",
    "kickoff",
    "penalty",
    # Open-play events
    "goal",
    "shot",
    "foul",
    "tackle",
    "clearance",
    "ball_out",
    "offside",
    # Match administration
    "substitution",
    "yellow_card",
    "red_card",
    "halftime",
]

# Human-readable one-liners — used as Label Studio choice hints and in prompts.
LABEL_DESCRIPTIONS: dict[str, str] = {
    "goal_kick": "Restart from the 6-yard box after the ball goes out off the attacking team.",
    "corner_kick": "Restart from the corner arc after the ball goes out off the defending team.",
    "throw_in": "Two-handed throw restart after the ball crosses a touchline.",
    "free_kick": "Direct or indirect free kick after a foul or infringement.",
    "kickoff": "Restart from the centre circle to start a half or after a goal.",
    "penalty": "Penalty kick from the spot.",
    "goal": "The ball fully crosses the goal line between the posts.",
    "shot": "An attempt on goal, on or off target.",
    "foul": "An infringement called by the referee.",
    "tackle": "A player dispossesses an opponent.",
    "clearance": "A defensive clearance out of a dangerous area.",
    "ball_out": "The ball leaves the field of play (generic dead-ball).",
    "offside": "An offside offence.",
    "substitution": "A player substitution.",
    "yellow_card": "A caution (yellow card).",
    "red_card": "A dismissal (red card).",
    "halftime": "Break in play / empty field / players on the sideline (usually excluded).",
}

# Labels the pipeline treats as dead-ball set pieces (mirrors phases.py).
SET_PIECE_LABELS: frozenset[str] = frozenset(
    {"goal_kick", "corner_kick", "throw_in", "free_kick", "kickoff", "penalty"}
)

_VALID = frozenset(EVENT_LABELS)


def is_valid_label(label: str) -> bool:
    """Whether ``label`` is part of the canonical taxonomy."""
    return label in _VALID
