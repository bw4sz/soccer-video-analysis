"""Tests for `enroll --team`: only dump the squad you're enrolling.

The classification pass itself needs a video, but everything that decides *which*
lanes get dumped is pure — where team labels come from, and the turf rejection
that makes colour sampling usable on a run with no segmentation masks.
"""

from __future__ import annotations

import json
from argparse import Namespace

import numpy as np

from soccer_vision.cli.enroll import _not_turf_mask, _resolve_track_teams


def _args(**kw):
    base = dict(profile=None, team_frames=300, team=None)
    base.update(kw)
    return Namespace(**base)


def test_not_turf_mask_rejects_pitch_keeps_kit():
    """Green pitch pixels are excluded; black and white kits survive."""
    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    frame[0, 0] = (60, 160, 60)     # turf green (BGR)
    frame[0, 1] = (20, 20, 20)      # black kit
    frame[0, 2] = (240, 240, 240)   # white kit

    mask = _not_turf_mask(frame)

    assert not mask[0, 0]
    assert mask[0, 1]
    assert mask[0, 2]


def test_teams_block_is_preferred(tmp_path):
    """A processed run stamps kit colours; that costs nothing, so use it."""
    tracks = tmp_path / "tracks.json"
    tracks.write_text(json.dumps({"teams": {"1": "black", "2": "white"}, "tracks": {}}))

    teams = _resolve_track_teams(tmp_path, tracks, [], reader=None, args=_args())

    assert teams == {1: "black", 2: "white"}


def test_falls_back_to_cache_when_run_predates_kit_stamping(tmp_path):
    """Older runs carry no `teams` block, so a previous pass's cache is reused."""
    tracks = tmp_path / "tracks.json"
    tracks.write_text(json.dumps({"tracks": {}}))
    (tmp_path / "track_teams.json").write_text(
        json.dumps({"teams": {"7": "black", "9": "white"}})
    )

    teams = _resolve_track_teams(tmp_path, tracks, [], reader=None, args=_args())

    assert teams == {7: "black", 9: "white"}
