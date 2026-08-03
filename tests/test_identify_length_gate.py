"""Tests for `identify --min-lane-seconds`: a fragment is not evidence.

At 30 fps most ByteTrack lanes are shorter than a second and 81% of re-id's
naming decisions were made on them — on one crop, at a similarity indistinguishable
from a good match. These pin that lane length, not similarity, is what holds them
back, and that the gate is off by construction when it can't measure length.
"""

from __future__ import annotations

from soccer_vision.identify.length_gate import gate_by_length, lane_span_s


def test_short_lanes_are_excluded():
    # 30 fps: lane 1 spans 2s, lane 2 a single frame, lane 3 exactly 1s
    frames = {1: list(range(0, 61)), 2: [100], 3: list(range(200, 231))}

    eligible, counts = gate_by_length(frames, 30.0, 1.0)

    assert eligible == {1, 3}
    assert counts == {"long_enough": 2, "too_short": 1}


def test_the_threshold_is_inclusive():
    """A lane exactly at the bar is long enough — no off-by-one at the boundary."""
    frames = {1: [0, 30]}

    eligible, _ = gate_by_length(frames, 30.0, 1.0)

    assert eligible == {1}


def test_span_is_measured_not_counted():
    """A sparse lane spanning 4s is eligible; detection count is a different question."""
    frames = {1: [0, 120]}

    eligible, _ = gate_by_length(frames, 30.0, 1.0)

    assert eligible == {1}
    assert lane_span_s(frames[1], 30.0) == 4.0


def test_sampled_runs_use_their_own_fps():
    """`sample_interval` shows up in the frame numbers, so seconds stay seconds."""
    frames = {1: [0, 6, 12, 18, 24, 30], 2: [0, 6]}   # 5 fps sampling of 30 fps video

    eligible, counts = gate_by_length(frames, 30.0, 1.0)

    assert eligible == {1}
    assert counts == {"long_enough": 1, "too_short": 1}


def test_zero_disables_the_gate():
    """Opting out must not quietly change who gets named."""
    frames = {1: [0], 2: [5, 500]}

    eligible, counts = gate_by_length(frames, 30.0, 0.0)

    assert eligible == {1, 2}
    assert counts == {"long_enough": 0, "too_short": 0}


def test_missing_fps_disables_the_gate():
    """Without fps a lane's length is unknown, which is not the same as short."""
    frames = {1: [0], 2: [5, 500]}

    eligible, _ = gate_by_length(frames, 0.0, 1.0)

    assert eligible == {1, 2}


def test_single_frame_lane_spans_nothing():
    assert lane_span_s([42], 30.0) == 0.0
    assert lane_span_s([], 30.0) == 0.0
