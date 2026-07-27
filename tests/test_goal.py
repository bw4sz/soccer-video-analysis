"""Goal detection: mouth consolidation and the ball-dwell heuristic.

Both halves are pure functions over JSON, so none of this needs SAM3 or a GPU —
the model only ever produces the boxes that :func:`consolidate` folds down.
"""

import pytest

from soccer_vision.detection.goal import assign_side, consolidate
from soccer_vision.events.goal import (
    GOAL_LABEL,
    GoalRegion,
    detect_goals,
    load_goal_regions,
)

FPS = 10.0
SAMPLE_DT = 0.1

# A goal mouth at the left end of a 1920x1080 frame.
LEFT_MOUTH = (100.0, 400.0, 300.0, 560.0)


def _region(bbox=LEFT_MOUTH, side="left", **kw):
    return GoalRegion(side=side, bbox=bbox, score=kw.pop("score", 0.8), **kw)


def _track(points, fps=FPS):
    """Ball track from a list of ``(x, y)`` or ``None`` (offscreen), 0.1s apart."""
    return {
        "fps": fps,
        "samples": [
            {"frame": int(i * fps * SAMPLE_DT), "timestamp_s": round(i * SAMPLE_DT, 2),
             "visible": p is not None,
             "pixel_x": p[0] if p else None, "pixel_y": p[1] if p else None}
            for i, p in enumerate(points)
        ],
    }


def _approach(n=5):
    """Ball on the pitch, right of the left-hand goal, approaching it."""
    return [(600.0 - i * 40, 480.0) for i in range(n)]


# --- the ball dwelling in the net --------------------------------------------


def test_ball_resting_in_the_net_is_a_goal():
    # 1.0s inside the mouth, comfortably past the 0.6s dwell threshold.
    track = _track(_approach() + [(200.0, 480.0)] * 11)
    events = detect_goals(track, [_region()])

    assert len(events) == 1
    e = events[0]
    assert e["label"] == GOAL_LABEL
    assert e["goal_zone"] == "left"
    assert e["dwell_s"] == pytest.approx(1.0, abs=0.05)
    assert e["drift_px"] == 0.0
    assert e["confidence"] > 0.8  # long dwell + still ball


def test_ball_passing_in_front_of_the_net_is_not_a_goal():
    """The whole point of the dwell: a 0.2s transit through the box is a fly-by."""
    track = _track([
        (600.0, 480.0), (400.0, 480.0),
        (250.0, 480.0), (200.0, 480.0),   # 0.2s inside
        (60.0, 480.0), (-20.0, 480.0),    # out the far side
    ])
    assert detect_goals(track, [_region()]) == []


def test_dwell_threshold_is_the_only_thing_separating_them():
    """Same fly-by, lower threshold — fires. Confirms the knob does the work."""
    track = _track([
        (600.0, 480.0), (400.0, 480.0),
        (250.0, 480.0), (200.0, 480.0),
        (60.0, 480.0),
    ])
    assert detect_goals(track, [_region()], min_dwell_s=0.1)


def test_ball_lost_in_the_netting_keeps_the_dwell_open():
    """Losing the ball inside the mouth is evidence *for* a goal, not against."""
    track = _track(
        _approach()
        + [(200.0, 480.0), (205.0, 485.0)]   # 0.1s visible inside
        + [None] * 4                          # detector loses it in the net
        + [(210.0, 480.0)] * 6                # re-acquired, still there
    )
    events = detect_goals(track, [_region()])

    assert len(events) == 1
    assert events[0]["dwell_s"] > 0.6
    assert events[0]["ball_lost_frac"] > 0


def test_long_offscreen_gap_ends_the_dwell():
    """Beyond --max-gap the ball is gone, not in the net; don't stitch across it."""
    track = _track(
        _approach()
        + [(200.0, 480.0), (205.0, 485.0)]
        + [None] * 20                         # 2.0s > max_gap_s
        + [(210.0, 480.0)] * 3
    )
    assert detect_goals(track, [_region()]) == []


def test_visible_ball_outside_the_mouth_ends_the_dwell():
    track = _track(
        _approach()
        + [(200.0, 480.0)] * 4        # 0.3s inside — short of the threshold
        + [(800.0, 480.0)]            # cleared away
        + [(200.0, 480.0)] * 4        # back in, still short
    )
    assert detect_goals(track, [_region()]) == []


# --- guards -------------------------------------------------------------------


def test_ball_parked_in_the_region_is_not_a_goal():
    """Seen on real data: a 16s dwell is dead time, not a score."""
    track = _track(_approach() + [(200.0, 480.0)] * 200)   # 20s parked
    assert detect_goals(track, [_region()]) == []
    assert detect_goals(track, [_region()], max_dwell_s=30.0)
    assert detect_goals(track, [_region()], max_dwell_s=0)  # 0 = no upper bound


def test_ball_arriving_from_behind_the_goal_is_rejected():
    """A ball retrieved from out of play crosses the mouth the wrong way."""
    track = _track([(20.0, 480.0), (60.0, 480.0)] + [(200.0, 480.0)] * 11)
    assert detect_goals(track, [_region()]) == []
    # ...unless the caller says not to care about entry direction.
    assert detect_goals(track, [_region()], require_entry=False)


def test_entry_check_is_mirrored_for_a_right_hand_goal():
    right = _region(bbox=(1600.0, 400.0, 1800.0, 560.0), side="right")
    from_pitch = _track([(1000.0, 480.0), (1400.0, 480.0)] + [(1700.0, 480.0)] * 11)
    from_behind = _track([(1900.0, 480.0), (1850.0, 480.0)] + [(1700.0, 480.0)] * 11)

    assert len(detect_goals(from_pitch, [right])) == 1
    assert detect_goals(from_behind, [right]) == []


def test_shot_level_with_the_post_is_inset_away():
    """A ball on the rim of the detected box isn't inside the goal."""
    on_the_post = 100.0 + 0.12 * 200.0 - 5  # just outside the 12% inset
    track = _track(_approach() + [(on_the_post, 480.0)] * 11)

    assert detect_goals(track, [_region()]) == []
    assert detect_goals(track, [_region()], inset_frac=0.0)


def test_rattling_in_the_net_collapses_to_one_goal():
    """Ball in, briefly out of the inset box, back in — one goal, not three."""
    track = _track(
        _approach()
        + [(200.0, 480.0)] * 8
        + [(400.0, 480.0)]
        + [(200.0, 480.0)] * 8
        + [(400.0, 480.0)]
        + [(200.0, 480.0)] * 8
    )
    events = detect_goals(track, [_region()], require_entry=False)
    assert len(events) == 1


def test_both_ends_are_detected_independently():
    left = _region()
    right = _region(bbox=(1600.0, 400.0, 1800.0, 560.0), side="right")
    track = _track(
        _approach() + [(200.0, 480.0)] * 11
        + [(900.0, 480.0)] * 200                       # 20s of midfield play
        + [(1400.0, 480.0)] + [(1700.0, 480.0)] * 11
    )
    events = detect_goals(track, [left, right])
    assert [e["goal_zone"] for e in events] == ["left", "right"]


def test_no_regions_means_no_events():
    assert detect_goals(_track([(200.0, 480.0)] * 20), []) == []


def test_empty_track_is_handled():
    assert detect_goals({"fps": FPS, "samples": []}, [_region()]) == []


# --- per-frame mouths (panning camera) ---------------------------------------


def test_nearest_sampled_mouth_tracks_a_moving_camera():
    """With per-frame observations the mouth follows the pan, not the median."""
    region = GoalRegion(
        side="left", bbox=(0.0, 400.0, 200.0, 560.0), score=0.8,
        samples=[{"frame": 0, "bbox": [0.0, 400.0, 200.0, 560.0]},
                 {"frame": 100, "bbox": [400.0, 400.0, 600.0, 560.0]}],
    )
    assert region.bbox_at(0)[0] == 0.0
    assert region.bbox_at(100)[0] == 400.0
    # Nothing sampled nearby: fall back to the consolidated box.
    assert region.bbox_at(100_000, max_frame_delta=50)[0] == 0.0


# --- mouth consolidation ------------------------------------------------------


def _obs(frame, bbox, side="left", score=0.8):
    return {"frame": frame, "bbox": list(bbox), "side": side, "score": score}


def test_consolidate_takes_the_median_box():
    obs = [_obs(i, (100 + i, 400, 300 + i, 560)) for i in range(5)]
    (goal,) = consolidate(obs)

    assert goal["side"] == "left"
    assert goal["bbox"] == [102.0, 400.0, 302.0, 560.0]
    assert goal["n_obs"] == 5
    assert len(goal["samples"]) == 5


def test_consolidate_ignores_a_single_wild_box():
    """One frame latching onto a banner shouldn't move the mouth."""
    obs = [_obs(i, (100, 400, 300, 560)) for i in range(8)]
    obs.append(_obs(9, (120, 300, 320, 460)))
    (goal,) = consolidate(obs)
    assert goal["bbox"] == [100.0, 400.0, 300.0, 560.0]


def test_consolidate_drops_a_side_with_too_few_observations():
    assert consolidate([_obs(0, (100, 400, 300, 560))], min_obs=3) == []


def test_consolidate_drops_a_side_whose_boxes_disagree():
    """Boxes scattered across the frame aren't one goal; a median of them is a lie."""
    obs = [_obs(i, (100 + i * 200, 400, 300 + i * 200, 560)) for i in range(6)]
    assert consolidate(obs) == []


def test_consolidate_separates_the_two_ends():
    obs = ([_obs(i, (100, 400, 300, 560), side="left") for i in range(4)]
           + [_obs(i, (1600, 400, 1800, 560), side="right") for i in range(4)])
    goals = consolidate(obs)
    assert sorted(g["side"] for g in goals) == ["left", "right"]


def test_assign_side_splits_on_the_frame_midpoint():
    assert assign_side((100, 400, 300, 560), 1920) == "left"
    assert assign_side((1600, 400, 1800, 560), 1920) == "right"


def test_load_goal_regions_round_trips_the_detector_payload():
    obs = [_obs(i, (100, 400, 300, 560)) for i in range(4)]
    payload = {"goals": consolidate(obs)}
    (region,) = load_goal_regions(payload)

    assert region.side == "left"
    assert region.bbox == (100.0, 400.0, 300.0, 560.0)
    assert region.n_obs == 4
