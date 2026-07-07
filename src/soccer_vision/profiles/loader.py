"""YAML project profile loader (roster, IDP paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_profile(path: Path | str) -> dict[str, Any]:
    """Load a YAML project profile and return as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_roster(profile: dict) -> list[dict]:
    return profile.get("roster", [])


def get_player(profile: dict, jersey: int) -> dict | None:
    for p in get_roster(profile):
        if p.get("jersey") == jersey:
            return p
    return None


def get_jersey_by_name(profile: dict, name: str) -> int | None:
    """Return the roster jersey number for ``name`` (case-insensitive)."""
    key = name.strip().lower()
    for p in get_roster(profile):
        if (p.get("name") or "").strip().lower() == key:
            j = p.get("jersey")
            return int(j) if j is not None else None
    return None
