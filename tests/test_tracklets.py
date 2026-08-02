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


# --- contiguous, complete labelling (for measuring track linking) -------------
#
# Building a gallery wants a *sample* of the match: spread windows, longest lanes
# only. Measuring whether the linker rejoined a player wants the opposite — one
# contiguous stretch with nothing left out, because the lanes it drops are the
# short ones linking exists to join.


@pytest.fixture
def crowded():
    """Ten lanes alive at once, more than one form can carry."""
    return {i: lane(0, 30, x=50.0 * i) for i in range(1, 11)}


def test_at_pins_the_window_instead_of_spreading(tracks):
    windows = choose_windows(tracks, fps=30, window_s=5, n_windows=1,
                             min_track_frames=3, max_lanes=12, start_s=10.0)
    assert [w["start_frame"] for w in windows] == [300]
    assert windows[0]["lanes"], "lane 3 starts at frame 300 and should be ringed"


def test_pinned_windows_run_back_to_back_not_spread(tracks):
    windows = choose_windows(tracks, fps=30, window_s=2, n_windows=3,
                             min_track_frames=3, max_lanes=12, start_s=0.0)
    # 2s at 30fps = 60 frames, so consecutive starts are 0, 60, 120 — contiguous
    # coverage, unlike the spread-across-the-match default.
    assert [w["start_frame"] for w in windows] == [0, 60, 120]


def test_all_lanes_pages_rather_than_truncating(crowded):
    capped = choose_windows(crowded, fps=30, window_s=60, n_windows=1,
                            min_track_frames=3, max_lanes=4)
    paged = choose_windows(crowded, fps=30, window_s=60, n_windows=1,
                           min_track_frames=3, max_lanes=4, all_lanes=True)

    assert sum(len(w["lanes"]) for w in capped) == 4, "default still drops the rest"
    assert sum(len(w["lanes"]) for w in paged) == 10, "paging must lose nobody"
    assert [w["n_pages"] for w in paged] == [3, 3, 3]
    assert [w["page"] for w in paged] == [1, 2, 3]
    # Same footage every page — only the rings differ.
    assert len({w["start_frame"] for w in paged}) == 1


def test_every_lane_appears_exactly_once_across_the_pages(crowded):
    paged = choose_windows(crowded, fps=30, window_s=60, n_windows=1,
                           min_track_frames=3, max_lanes=4, all_lanes=True)
    seen = [l["track_id"] for w in paged for l in w["lanes"]]
    assert sorted(seen) == sorted(crowded), "a lane in two pages is labelled twice"
    # Slots restart per page, so the manifest is what disambiguates them.
    assert all(l["slot"] <= 4 for w in paged for l in w["lanes"])


def test_pages_are_longest_first_so_stopping_early_is_measurable():
    tracks = {1: lane(0, 30), 2: lane(0, 20, x=200), 3: lane(0, 10, x=300),
              4: lane(0, 5, x=400)}
    paged = choose_windows(tracks, fps=30, window_s=60, n_windows=1,
                           min_track_frames=3, max_lanes=2, all_lanes=True)
    assert [l["track_id"] for l in paged[0]["lanes"]] == [1, 2]
    assert [l["track_id"] for l in paged[1]["lanes"]] == [3, 4]


def test_repeated_footage_is_announced_so_page_two_is_not_skipped(crowded):
    from soccer_vision.annotate.tracklets import onscreen_guide

    paged = choose_windows(crowded, fps=30, window_s=60, n_windows=1,
                           min_track_frames=3, max_lanes=4, all_lanes=True)
    guide = onscreen_guide(paged[1], 4)
    assert "PASS 2 of 3" in guide
    # And a single-page window says nothing about passes.
    single = choose_windows(crowded, fps=30, window_s=60, n_windows=1,
                            min_track_frames=3, max_lanes=20, all_lanes=True)
    assert "PASS" not in onscreen_guide(single[0], 20)


def test_tasks_carry_the_page_so_an_export_can_be_reconciled(crowded):
    paged = choose_windows(crowded, fps=30, window_s=60, n_windows=1,
                           min_track_frames=3, max_lanes=4, all_lanes=True)
    tasks = build_tasks(paged, {w["window"]: f"u{w['window']}" for w in paged},
                        fps=30, max_lanes=4)
    assert [t["data"]["page"] for t in tasks] == [1, 2, 3]
    assert {t["data"]["n_pages"] for t in tasks} == {3}
    # Window numbers stay unique across pages — they key the clip files.
    assert len({t["data"]["window"] for t in tasks}) == 3
