"""Tests for `identify --team`: a gallery of our squad must not name their squad.

`match_track` scores a crop against our players only, so a referee or an opponent
comes back as whichever of ours is nearest, with an ordinary-looking margin. The
kit colour is the one signal that can say "none of the above", and these tests
pin how it is applied — including that a *missing* colour is not treated as a
wrong one.
"""

from __future__ import annotations

from soccer_vision.identify.team_gate import gate_by_kit


def test_other_kit_is_excluded():
    kits = {1: "black", 2: "white", 3: "black"}

    eligible, counts = gate_by_kit(kits, kits, "black")

    assert eligible == {1, 3}
    assert counts == {"ours": 2, "other": 1, "unassigned": 0}


def test_unassigned_lanes_stay_eligible_by_default():
    """No kit assigned is an absence of evidence, not evidence of the wrong kit."""
    kits = {1: "black", 2: None, 3: "white"}

    eligible, counts = gate_by_kit([1, 2, 3, 4], kits, "black")

    assert eligible == {1, 2, 4}          # 4 is missing from the map entirely
    assert counts == {"ours": 1, "other": 1, "unassigned": 2}


def test_strict_holds_back_unassigned_lanes():
    kits = {1: "black", 2: None, 3: "white"}

    eligible, counts = gate_by_kit(kits, kits, "black", strict=True)

    assert eligible == {1}
    assert counts == {"ours": 1, "other": 1, "unassigned": 1}


def test_no_kit_named_disables_the_gate():
    """`--team` unset must not quietly change who gets named."""
    kits = {1: "black", 2: "white"}

    eligible, counts = gate_by_kit(kits, kits, None)

    assert eligible == {1, 2}
    assert counts == {"ours": 0, "other": 0, "unassigned": 0}


def test_kit_comparison_ignores_case_and_padding():
    kits = {1: "Black", 2: " black ", 3: "WHITE"}

    eligible, _ = gate_by_kit(kits, kits, "BLACK")

    assert eligible == {1, 2}


def test_empty_string_kit_counts_as_unassigned():
    """The classifier writing "" must not be read as a kit named "" that differs."""
    kits = {1: "", 2: "white"}

    eligible, counts = gate_by_kit(kits, kits, "black")

    assert eligible == {1}
    assert counts["unassigned"] == 1
