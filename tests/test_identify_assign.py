"""Tests for naming as an assignment: one player holds one lane at a time.

Per-lane naming has no way to know there is one Morgan on the pitch, so on 18.4%
of the frames she was named at, two to four lanes carried her name at once. These
pin the two things the constraint is for: a name can never land on overlapping
lanes, and the margin is measured against the identities still *available* rather
than against whoever happens to sit second in the gallery.
"""

from __future__ import annotations

import numpy as np

from soccer_vision.identify.assign import (
    Lane,
    assign_identities,
    concurrency,
    explain,
)
from soccer_vision.identify.gallery import NEGATIVE_LABEL

NAMES = ["Morgan", "Quinn", "Leire", NEGATIVE_LABEL]


def test_one_name_cannot_hold_two_overlapping_lanes():
    """The whole point: the runner-up gets the weaker of two concurrent claims."""
    lanes = [Lane("a", 0.0, 10.0), Lane("b", 5.0, 15.0)]
    #            Morgan Quinn Leire  neg
    scores = [[0.90, 0.60, 0.55, 0.10],   # a: clearly Morgan
              [0.80, 0.50, 0.52, 0.10]]   # b: also looks like Morgan, less so

    out = assign_identities(lanes, np.array(scores), NAMES)

    assert out["a"].name == "Morgan"
    assert out["b"].name != "Morgan"
    assert max(concurrency(out, lanes).values()) == 1


def test_the_same_name_twice_is_fine_when_the_lanes_do_not_overlap():
    """A player is one person at an instant, not across a match."""
    lanes = [Lane("a", 0.0, 10.0), Lane("b", 20.0, 30.0)]
    scores = [[0.90, 0.50, 0.50, 0.10],
              [0.85, 0.50, 0.50, 0.10]]

    out = assign_identities(lanes, np.array(scores), NAMES)

    assert out["a"].name == "Morgan"
    assert out["b"].name == "Morgan"


def test_margin_is_measured_against_what_is_still_available():
    """A rival committed to an overlapping lane is not an alternative.

    Lane `b` is torn between Quinn and Leire (0.02 apart), which the per-lane
    margin would abstain on. But Leire is already held by a lane `b` overlaps, so
    the real question is Quinn against the next *available* identity — a lead of
    0.20, and a confident name.
    """
    lanes = [Lane("a", 0.0, 10.0), Lane("b", 5.0, 15.0)]
    scores = [[0.30, 0.30, 0.95, 0.10],   # a: certainly Leire
              [0.50, 0.72, 0.70, 0.10]]   # b: Quinn 0.72 vs Leire 0.70

    out = assign_identities(lanes, np.array(scores), NAMES, min_margin=0.05)

    assert out["a"].name == "Leire"
    assert out["b"].name == "Quinn"
    assert out["b"].open_margin < 0.05        # per-lane would have abstained
    assert out["b"].margin >= 0.05            # against available rivals, it's clear
    assert out["b"].blocked_by == [("Leire", "a")]


def test_a_lane_still_abstains_when_no_rival_is_blocked():
    """The constraint only helps where there is a conflict — it invents nothing."""
    lanes = [Lane("a", 0.0, 10.0)]
    scores = [[0.72, 0.70, 0.30, 0.10]]

    out = assign_identities(lanes, np.array(scores), NAMES, min_margin=0.05)

    assert out["a"].name is None
    assert out["a"].margin == out["a"].open_margin


def test_the_negative_class_is_never_exclusive():
    """A crowd is full of people who are all simultaneously not ours."""
    lanes = [Lane("a", 0.0, 10.0), Lane("b", 2.0, 12.0)]
    scores = [[0.30, 0.30, 0.30, 0.90],
              [0.30, 0.30, 0.30, 0.88]]

    out = assign_identities(lanes, np.array(scores), NAMES)

    assert out["a"].rejected and out["b"].rejected
    assert out["a"].name is None and out["b"].name is None


def test_nothing_below_the_similarity_floor_is_named():
    lanes = [Lane("a", 0.0, 10.0)]
    scores = [[0.40, 0.10, 0.10, 0.05]]

    out = assign_identities(lanes, np.array(scores), NAMES, min_similarity=0.5)

    assert out["a"].name is None


def test_a_blocked_lane_is_reconsidered_once_its_rival_is_committed():
    """Losing the top identity resolves an ambiguity rather than ending the lane.

    Lane `b` ties Morgan and Quinn, so on its own it abstains. Lane `a` then takes
    Morgan outright; with Morgan gone, `b`'s choice is no longer a tie and it is
    named Quinn.
    """
    lanes = [Lane("a", 0.0, 10.0), Lane("b", 5.0, 15.0)]
    scores = [[0.95, 0.20, 0.20, 0.10],
              [0.71, 0.70, 0.30, 0.10]]

    out = assign_identities(lanes, np.array(scores), NAMES, min_margin=0.05)

    assert out["a"].name == "Morgan"
    assert out["b"].name == "Quinn"


def test_overlap_tolerance_forgives_a_handoff_frame():
    """A frame of overhang at a tracker handoff is not two players on the pitch."""
    lanes = [Lane("a", 0.0, 10.05), Lane("b", 10.0, 20.0)]
    scores = [[0.90, 0.30, 0.30, 0.10],
              [0.85, 0.30, 0.30, 0.10]]

    strict = assign_identities(lanes, np.array(scores), NAMES)
    loose = assign_identities(lanes, np.array(scores), NAMES, overlap_tolerance_s=0.2)

    assert strict["b"].name != "Morgan"
    assert loose["b"].name == "Morgan"


def test_containment_is_a_conflict():
    """A duplicate box living entirely inside a long lane competes with it.

    `merge_duplicate_lanes` only ever pairs a lane's death with another's birth,
    so this case reaches naming intact — it is how a 3-frame ghost took a name
    from the 30 s lane on the same player.
    """
    lanes = [Lane("long", 0.0, 30.0), Lane("ghost", 12.0, 12.1)]
    scores = [[0.80, 0.30, 0.30, 0.10],
              [0.90, 0.30, 0.30, 0.10]]

    out = assign_identities(lanes, np.array(scores), NAMES)

    assert sum(a.name == "Morgan" for a in out.values()) == 1
    assert max(concurrency(out, lanes).values()) == 1


def test_explain_names_the_lane_that_blocked_it():
    lanes = [Lane("a", 0.0, 10.0), Lane("b", 5.0, 15.0)]
    scores = [[0.30, 0.30, 0.95, 0.10],
              [0.50, 0.72, 0.70, 0.10]]

    out = assign_identities(lanes, np.array(scores), NAMES)

    assert "Leire" in explain(out, "b") and "lane a" in explain(out, "b")
    assert "not scored" in explain(out, "nope")
