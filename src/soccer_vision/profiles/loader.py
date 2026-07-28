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


def get_kits(profile: dict) -> list[str]:
    """Return the declared kit-colour names (e.g. ``["black", "white"]``).

    Accepts either bare strings or ``{name: ...}`` mappings under ``kits``.
    Empty list when none are declared.
    """
    kits: list[str] = []
    for k in profile.get("kits", []) or []:
        name = k.get("name") if isinstance(k, dict) else k
        if name:
            kits.append(str(name).strip().lower())
    return kits


def get_reid(profile: dict) -> dict:
    """Return the ``reid:`` block — gallery path and matching thresholds.

    A long-running team keeps its appearance gallery in the profile alongside the
    roster, so `identify` needs no flags after the first enrolment::

        reid:
          gallery: galleries/saints-u11.npz
          min_similarity: 0.5
          min_margin: 0.05
    """
    return profile.get("reid", {}) or {}


def get_player(profile: dict, jersey: int) -> dict | None:
    for p in get_roster(profile):
        if p.get("jersey") == jersey:
            return p
    return None


def get_jersey_by_name(profile: dict, name: str) -> int | None:
    """Return the roster jersey number for ``name`` (case-insensitive).

    Rosters carry full names ("Simon Weinstein") but people ask for players by
    first name, so an exact match is tried first and then a first-name match.
    A first name shared by two players on the roster is ambiguous and resolves
    to nothing rather than to whichever happens to be listed first — the caller
    reports no match and the user can disambiguate with ``--number``.

    A ``nickname`` counts as an exact match, so ``--player Mo`` reaches Morrighan
    — that is what the squad calls her, and it's the label she was annotated
    under.
    """
    key = name.strip().lower()

    def jersey_of(p: dict) -> int | None:
        j = p.get("jersey")
        return int(j) if j is not None else None

    roster = get_roster(profile)
    for p in roster:
        if (p.get("name") or "").strip().lower() == key:
            return jersey_of(p)
        if (p.get("nickname") or "").strip().lower() == key:
            return jersey_of(p)

    first = [p for p in roster if (p.get("name") or "").strip().lower().split(" ")[0] == key]
    return jersey_of(first[0]) if len(first) == 1 else None
