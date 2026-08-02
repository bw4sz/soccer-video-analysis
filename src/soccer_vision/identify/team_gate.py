"""Refuse to put one of our players' names on a lane wearing someone else's kit.

A gallery holds one squad, and `match_track` scores a crop against *only* those
players — so it has no way to answer "none of the above". Handed a referee, an
opponent or a spectator, it returns whichever of our eleven is nearest and the
margin test sees a perfectly ordinary win. Measured on ``runs/saints-u14g-full``
(gallery = 620 black-kit crops of the Saints U14G squad), **878 of 1595 named
lanes were on the white kit** against 256 on our own: the opposing squad, the
yellow-shirted officials, and players on the neighbouring pitch, all confidently
named after somebody's daughter.

The kit colour `process` already stamps into ``tracks.json`` settles it for free.
A lane classified as a kit that is not ours cannot be one of our players,
whatever the embedding thinks, so it is never named at all — and the re-id
forward passes over those lanes are skipped too.

**Absence of a kit is not evidence of the wrong kit.** About a third of lanes get
no colour (too short, or never seen against grass), and rejecting those would
discard real players to no purpose, so by default they stay eligible — the same
abstention logic the OCR veto uses in :mod:`soccer_vision.identify.crosscheck`.
``strict`` flips that for a run where precision matters more than reach.

Pure (no video, no model), so the decision is unit-testable on a dict of kits.
"""

from __future__ import annotations

from collections.abc import Iterable

# why a lane was held back, recorded in jerseys.json so a gap is explainable
EXCLUDED_KIT = "kit"


def gate_by_kit(
    track_ids: Iterable[int],
    track_kits: dict[int, str | None],
    kit: str | None,
    *,
    strict: bool = False,
) -> tuple[set[int], dict[str, int]]:
    """Split lanes into those eligible to be named and those the kit rules out.

    ``kit`` is our squad's colour for this match ("black"); ``None`` disables the
    gate entirely and everything stays eligible. ``track_kits`` maps a lane to the
    colour `process` assigned it, with a missing or ``None`` value meaning no
    colour was assigned.

    Returns ``(eligible_ids, counts)``, where ``counts`` carries ``ours``,
    ``other`` and ``unassigned`` for reporting.
    """
    ids = list(track_ids)
    counts = {"ours": 0, "other": 0, "unassigned": 0}
    if kit is None:
        return set(ids), counts

    want = kit.strip().lower()
    eligible: set[int] = set()
    for tid in ids:
        lane_kit = (track_kits.get(tid) or "").strip().lower() or None
        if lane_kit == want:
            counts["ours"] += 1
            eligible.add(tid)
        elif lane_kit is None:
            counts["unassigned"] += 1
            if not strict:
                eligible.add(tid)
        else:
            counts["other"] += 1
    return eligible, counts
