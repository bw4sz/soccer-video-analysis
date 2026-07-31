"""Tests for the on-field cut.

The cut is asymmetric on purpose: our cameras stand at the touchline, so the
bottom and the sides of the frame are our own pitch and only the top holds other
people's matches. The centred rectangle this replaced discarded 36% of detected
people (job 38180242) and clipped every track in `runs/saints-u14g-full` into
x in [288, 1632], y <= 918 — so a player in the near corner never got a track id
and never reached a labelling clip.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import supervision as sv

from soccer_vision.cli.main import field_filter_kwargs
from soccer_vision.detection.field_filter import filter_spectators

FRAME = (1080, 1920, 3)


def _dets(*feet):
    """Detections whose foot points (bottom-centre) are the given (x, y)."""
    xyxy = np.array([[x - 10, y - 40, x + 10, y] for x, y in feet], dtype=float)
    return sv.Detections(xyxy=xyxy)


def test_bottom_corner_player_is_kept():
    """The regression this exists for: a player in the near corner survives.

    Foot point (83, 963) is the player the old rectangle dropped on both axes —
    inside its left margin of 288 px and below its bottom limit of 918.
    """
    kept = filter_spectators(_dets((83, 963)), FRAME)
    assert len(kept) == 1


def test_sides_and_bottom_are_open_by_default():
    feet = [(5, 540), (1915, 540), (960, 1079), (960, 1079)]
    assert len(filter_spectators(_dets(*feet), FRAME)) == len(feet)


def test_top_is_cut_by_default():
    """Above 15% of frame height is sky, trees and rooftops."""
    kept = filter_spectators(_dets((960, 100), (960, 200)), FRAME)
    assert len(kept) == 1
    assert kept.xyxy[0][3] == 200


def test_foot_point_decides_not_the_whole_box():
    """A tall player at the top of the band is kept even though her head is above it.

    Box spans y 122..200; only the feet are tested, so she survives a 15% cut.
    """
    assert len(filter_spectators(_dets((960, 200)), FRAME)) == 1


def test_side_and_bottom_cuts_still_available():
    """A venue where the pitch really does end inside the frame can opt in."""
    dets = _dets((5, 540), (960, 1070), (960, 540))
    kept = filter_spectators(dets, FRAME, side_frac=0.15, bottom_frac=0.15)
    assert len(kept) == 1
    assert tuple(kept.xyxy[0][[2, 3]]) == (970.0, 540.0)


def test_zero_top_frac_is_honoured_not_replaced_by_the_default():
    """`--field-top 0` means cut nothing — a falsy value that must survive."""
    kw = field_filter_kwargs(Namespace(field_top=0.0, field_sides=0.0, field_bottom=0.0))
    assert kw["top_frac"] == 0.0
    assert len(filter_spectators(_dets((960, 5)), FRAME, **kw)) == 1


def test_missing_args_fall_back_to_shipped_defaults():
    kw = field_filter_kwargs(Namespace())
    assert kw == {"top_frac": 0.15, "side_frac": 0.0, "bottom_frac": 0.0}


def test_empty_detections_pass_through():
    empty = sv.Detections(xyxy=np.empty((0, 4)))
    assert len(filter_spectators(empty, FRAME)) == 0
