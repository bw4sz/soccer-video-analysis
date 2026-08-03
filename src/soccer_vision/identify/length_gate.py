"""Refuse to name a lane too short to be worth a name.

A ByteTrack lane is an ephemeral fragment, and at 30 fps most of them are
*very* short: on ``runs/saints-u14g-full-30fps`` the median lane is 0.47 s and
57% of lanes last under a second, together holding **1.7% of all tracked
time**. Re-id happily names them anyway — ``embed_tracks`` samples whatever
crops a lane has, so a one-frame lane is one crop, one pose, one instant of
motion blur, and :func:`~soccer_vision.identify.reid.match_track` still returns
whichever of our eleven is nearest with a perfectly ordinary margin.

Measured on that run, **81% of re-id's naming decisions (1,172 of 1,450) were
made on lanes shorter than one second**, and 968 on lanes under half a second —
mostly a single detection frame. Their mean winning similarity is 0.654 against
0.732 for lanes over 5 s, so they sit comfortably above ``min_similarity``:
**a similarity threshold cannot separate them and lane length can.**

The damage is not confined to the fragment. ``link.propagate_names`` seeds a
whole chain from its highest-similarity member, and 52% of propagated names on
that run trace back to a seed lane under a second — so one bad crop names
everything the linker joined to it. Gating the seeds at 1 s cut the frames where
one player's name is alive on two lanes at once from 20.3% to 12.2%, while
keeping 79% of the named on-ball seconds (352 s → 279 s) and all but 1 s of
Morgan's 22 s.

Two things this gate is *not*:

- **Not a minimum action length.** How long a touch must last to be worth a clip
  is ``--on-ball-min-span`` (0.4 s), a separate question asked of the span, not
  of the lane. A player can be on the ball for three seconds while the tracker
  splits her across six fragments.
- **Not a fix for the name-collision bug** (issue #28). On the 5 fps run, where
  the median lane is already 3.6 s, this gate changes the collision rate by 0.1
  points — those collisions are long lanes sharing a name, which only mutual
  exclusion fixes. This removes the cheapest and most obviously worthless half of
  the input to that bug.

**Run the linker first if you can.** ``link-tracks --in-place`` before
``identify`` turns fragments into chains, so the gate then asks how long the
*player* was tracked rather than how long an id survived, and costs almost
nothing.

Pure (no video, no model), so the decision is unit-testable on a dict of frames.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# why a lane was held back, recorded in jerseys.json so a gap is explainable
EXCLUDED_SHORT = "short"


def lane_span_s(frames: Iterable[int], fps: float) -> float:
    """Seconds from a lane's first detection to its last (0.0 for a single frame)."""
    seq = list(frames)
    if not seq or not fps:
        return 0.0
    return (max(seq) - min(seq)) / fps


def gate_by_length(
    track_frames: Mapping[int, Iterable[int]],
    fps: float,
    min_seconds: float,
) -> tuple[set[int], dict[str, int]]:
    """Split lanes into those long enough to name and those that are not.

    ``track_frames`` maps a lane to the frame numbers it was detected on;
    ``min_seconds`` of 0 (or a missing ``fps``) disables the gate entirely and
    everything stays eligible.

    Returns ``(eligible_ids, counts)`` with ``long_enough`` and ``too_short`` for
    reporting.
    """
    ids = list(track_frames)
    counts = {"long_enough": 0, "too_short": 0}
    if not min_seconds or not fps:
        return set(ids), counts

    eligible: set[int] = set()
    for tid in ids:
        if lane_span_s(track_frames[tid], fps) >= min_seconds:
            counts["long_enough"] += 1
            eligible.add(tid)
        else:
            counts["too_short"] += 1
    return eligible, counts
