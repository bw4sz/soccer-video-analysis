"""Tracklet labelling: window selection, the slot indirection, and harvesting.

The video rendering needs a real file and is not covered here; everything that
decides *what* gets labelled and *whose* crops come back is pure.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_vision.annotate.tracklets import (
    NOT_OURS,
    UNSURE,
    boxes_from_tracklet_export,
    build_tasks,
    choose_windows,
    labeling_config,
    slot_label,
)


def lane(start, n, step=6, x=100.0):
    """A lane sampled every ``step`` frames, as tracks.json stores them."""
    return [(start + i * step, np.asarray([x, 50.0, x + 30, 130.0], dtype=np.float32))
            for i in range(n)]


@pytest.fixture
def tracks():
    return {
        1: lane(0, 40),          # long, spans the first half of the clip
        2: lane(0, 40, x=200),
        3: lane(300, 40, x=300),  # long, later
        4: lane(0, 2),            # a fragment — too short to label
    }


def test_short_lanes_are_never_offered(tracks):
    windows = choose_windows(tracks, fps=30, window_s=10, n_windows=1,
                             min_track_frames=5, max_lanes=12)
    assert 4 not in [x["track_id"] for w in windows for x in w["lanes"]]


def test_windows_spread_across_the_match(tracks):
    windows = choose_windows(tracks, fps=30, window_s=5, n_windows=3,
                             min_track_frames=5, max_lanes=12)
    starts = [w["start_frame"] for w in windows]
    assert starts == sorted(starts)
    assert starts[0] < starts[-1], "windows must not all sit on the same moment"


def test_lanes_are_ranked_longest_first_and_capped(tracks):
    windows = choose_windows(tracks, fps=30, window_s=60, n_windows=1,
                             min_track_frames=5, max_lanes=2)
    lanes = windows[0]["lanes"]
    assert [x["slot"] for x in lanes] == [1, 2]
    assert lanes[0]["n_frames"] >= lanes[1]["n_frames"]


def test_team_filter_keeps_only_our_squad(tracks):
    windows = choose_windows(tracks, fps=30, window_s=60, n_windows=1,
                             min_track_frames=5, max_lanes=12,
                             teams={1: "black", 2: "white", 3: "white"}, team="black")
    assert [x["track_id"] for x in windows[0]["lanes"]] == [1]


def test_slot_chips_match_the_form_labels():
    """A clip showing "J" against a form listing "Player 1" costs a translation."""
    assert [slot_label(i) for i in (1, 2, 12)] == ["1", "2", "12"]


def test_config_declares_one_dropdown_per_slot():
    xml = labeling_config(["Mo", "Evie"], max_lanes=3)
    for slot in (1, 2, 3):
        assert f'name="p{slot}"' in xml
    assert 'name="p4"' not in xml
    assert '<Choice value="Mo"/>' in xml
    assert '<Header value="Player 1' in xml

    # Abstaining has to be as easy as naming, or people guess to clear the form.
    assert f'<Choice value="{NOT_OURS}"/>' in xml
    assert f'<Choice value="{UNSURE}"/>' in xml


def _export(window, slot_choices: dict):
    return [{
        "data": {"window": window},
        "annotations": [{"result": [
            {"from_name": f"p{slot}", "to_name": "video", "type": "choices",
             "value": {"choices": [name]}}
            for slot, name in slot_choices.items()
        ]}],
    }]


def test_named_lane_yields_every_crop_in_it(tracks):
    windows = choose_windows(tracks, fps=30, window_s=60, n_windows=1,
                             min_track_frames=5, max_lanes=12)
    manifest = {"windows": windows}
    slot = windows[0]["lanes"][0]["slot"]
    track_id = windows[0]["lanes"][0]["track_id"]

    boxes, summary = boxes_from_tracklet_export(
        _export(1, {slot: "Mo"}), manifest, tracks, max_samples_per_lane=100)

    assert summary["lanes_named"] == 1
    assert {n for _, _, n in boxes} == {"Mo"}
    # One decision, every crop in the lane — the whole point of the workflow.
    assert len(boxes) == len(tracks[track_id])


def test_not_ours_and_unsure_enrol_nothing(tracks):
    windows = choose_windows(tracks, fps=30, window_s=60, n_windows=1,
                             min_track_frames=5, max_lanes=12)
    slots = [x["slot"] for x in windows[0]["lanes"]][:2]

    boxes, summary = boxes_from_tracklet_export(
        _export(1, {slots[0]: NOT_OURS, slots[1]: UNSURE}),
        {"windows": windows}, tracks)

    assert boxes == []
    assert summary["lanes_skipped"] == 2
    assert summary["lanes_named"] == 0


def test_a_lane_is_subsampled_not_dumped_whole(tracks):
    """129 near-duplicate crops from one lane would swamp a dozen real views."""
    windows = choose_windows(tracks, fps=30, window_s=60, n_windows=1,
                             min_track_frames=5, max_lanes=12)
    slot = windows[0]["lanes"][0]["slot"]

    boxes, _ = boxes_from_tracklet_export(
        _export(1, {slot: "Mo"}), {"windows": windows}, tracks,
        max_samples_per_lane=5)

    assert len(boxes) == 5


def test_slot_numbers_mean_different_tracks_in_different_windows(tracks):
    """The indirection the config forces: slot 1 is not one player across tasks."""
    windows = choose_windows(tracks, fps=30, window_s=5, n_windows=3,
                             min_track_frames=5, max_lanes=1)
    slot1 = {w["window"]: w["lanes"][0]["track_id"] for w in windows}
    assert len(set(slot1.values())) > 1, "test fixture should exercise the remap"

    # Naming slot 1 in the last window must not enrol the first window's player.
    last = windows[-1]
    boxes, _ = boxes_from_tracklet_export(
        _export(last["window"], {1: "Mo"}), {"windows": windows}, tracks,
        max_samples_per_lane=100)
    expected = last["lanes"][0]
    assert all(expected["first_frame"] <= f <= expected["last_frame"]
               for f, _, _ in boxes)


def test_tasks_carry_the_window_and_clip(tracks):
    windows = choose_windows(tracks, fps=30, window_s=5, n_windows=2,
                             min_track_frames=5, max_lanes=12)
    urls = {w["window"]: f"/data/local-files/?d=w{w['window']}.mp4" for w in windows}

    tasks = build_tasks(windows, urls, fps=30)

    assert len(tasks) == len(windows)
    assert tasks[0]["data"]["video"] == urls[windows[0]["window"]]
    assert tasks[0]["data"]["n_lanes"] == len(windows[0]["lanes"])
    assert "timestamp" in tasks[0]["data"]


def test_slots_are_numbered_in_order_of_first_appearance():
    """Numbering by lane length made the sequence look random when scrubbing.

    The longest lanes are still the ones chosen — they carry the most crops — but
    numbering follows the clip, so playing it once walks the form top to bottom.
    """
    tracks = {
        1: lane(300, 30),           # longest, but enters late
        2: lane(0, 20, x=200),      # shorter, enters first
        3: lane(120, 25, x=300),    # middle on both counts
    }
    windows = choose_windows(tracks, fps=30, window_s=60, n_windows=1,
                             min_track_frames=5, max_lanes=3)
    lanes = windows[0]["lanes"]

    assert [x["track_id"] for x in lanes] == [2, 3, 1]
    assert [x["slot"] for x in lanes] == [1, 2, 3]
    assert [x["enters_s"] for x in lanes] == sorted(x["enters_s"] for x in lanes)


def test_onscreen_guide_names_the_slots_a_clip_does_not_use():
    """Ten dropdowns over a six-player clip otherwise reads as broken."""
    from soccer_vision.annotate.tracklets import onscreen_guide

    window = {"lanes": [{"slot": 1, "enters_s": 0.0, "leaves_s": 9.0},
                        {"slot": 2, "enters_s": 3.0, "leaves_s": 18.0}]}

    guide = onscreen_guide(window, max_lanes=4)

    assert "Player 1: 0-9s" in guide
    assert "Player 2: 3-18s" in guide
    assert "Players 3, 4 are not in this clip" in guide
