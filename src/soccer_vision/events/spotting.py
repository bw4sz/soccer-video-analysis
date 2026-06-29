"""Action spotting adapter stub for opensportslib / sn-teamspotting."""

from __future__ import annotations

# SoccerNet → soccer-vision label mapping
SOCCERNET_LABEL_MAP = {
    "Goal": "goal",
    "Penalty": "penalty",
    "Corner": "corner_kick",
    "Free Kick": "free_kick",
    "Goal Kick": "goal_kick",
    "Throw-In": "throw_in",
    "Yellow Card": "yellow_card",
    "Red Card": "red_card",
    "Substitution": "substitution",
    "Kick-Off": "kickoff",
    "Foul": "foul",
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Clearance": "clearance",
    "Ball out of play": "ball_out",
    "Indirect free-kick": "free_kick",
    "Direct free-kick": "free_kick",
}


def map_soccernet_label(sn_label: str) -> str:
    return SOCCERNET_LABEL_MAP.get(sn_label, sn_label.lower().replace(" ", "_"))


def is_available() -> bool:
    """Check if opensportslib is installed."""
    try:
        import opensportslib  # noqa: F401
        return True
    except ImportError:
        return False

# Stub — Phase 5: integrate opensportslib LocalizationModel + sn-teamspotting
