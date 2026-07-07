"""Resolve a name/number query to the set of track lanes carrying that jersey.

ByteTrack fragments one player into many lanes over a match, and voting assigns
a jersey to each lane independently (:mod:`.vote`). So "number 6" or "Simon" maps
to a *set* of track ids — every lane whose vote is 6 — which is precisely why
jersey identity beats selecting a single raw ``--track`` id.
"""

from __future__ import annotations

from soccer_vision.profiles.loader import get_jersey_by_name


def tracks_for(
    jerseys_doc: dict,
    *,
    number: int | None = None,
    name: str | None = None,
    profile: dict | None = None,
) -> set[int]:
    """Track ids whose voted jersey matches ``number`` (or ``name`` via roster).

    ``jerseys_doc`` is a parsed ``jerseys.json``. Exactly one of ``number`` /
    ``name`` should be given. A ``name`` is resolved to a jersey number through
    the profile roster when supplied; failing that (or with no profile) it falls
    back to matching the ``name`` stored on each track by ``identify``. Returns
    an empty set when nothing matches.
    """
    tracks = jerseys_doc.get("tracks", {})

    if number is None and name is not None:
        if profile is not None:
            number = get_jersey_by_name(profile, name)
        if number is None:
            # No roster mapping — match the name identify stored per track.
            key = name.strip().lower()
            return {
                int(tid)
                for tid, info in tracks.items()
                if (info.get("name") or "").strip().lower() == key
            }

    if number is None:
        return set()

    return {
        int(tid)
        for tid, info in tracks.items()
        if info.get("jersey") == number
    }
