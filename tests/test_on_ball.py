"""On-ball span selection and the ``--player`` fallback that cuts them."""

import json
from types import SimpleNamespace

from soccer_vision.cli.extract import _on_ball_fallback, _reel_window
from soccer_vision.clips.extract import halo_samples_for
from soccer_vision.events.on_ball import (
    ON_BALL_LABEL,
    OnBallSpan,
    select_on_ball_spans,
    spans_to_events,
)

FPS = 10.0


def _ball(frames_xy):
    """Ball track from ``{frame: (x, y) or None}``; None = offscreen."""
    return {
        "fps": FPS,
        "samples": [
            {"frame": f, "visible": xy is not None,
             "pixel_x": xy[0] if xy else None, "pixel_y": xy[1] if xy else None}
            for f, xy in sorted(frames_xy.items())
        ],
    }


def _tracks(by_tid, teams=None):
    """Tracks doc from ``{tid: {frame: (foot_x, foot_y)}}`` (bbox is 20px wide)."""
    return {
        "fps": FPS,
        "teams": teams or {},
        "tracks": {
            str(tid): [
                {"frame": f, "bbox": [x - 10, y - 40, x + 10, y]}
                for f, (x, y) in sorted(frames.items())
            ]
            for tid, frames in by_tid.items()
        },
    }


# --- span selection -------------------------------------------------------

def test_nearest_player_within_range_makes_a_span():
    # Track 3 shadows the ball for 6 frames; track 9 is far away throughout.
    ball = _ball({f: (100.0 + f, 200.0) for f in range(6)})
    tracks = _tracks({
        3: {f: (105.0 + f, 210.0) for f in range(6)},
        9: {f: (900.0, 800.0) for f in range(6)},
    })
    spans = select_on_ball_spans(ball, tracks, {3})
    assert len(spans) == 1
    assert spans[0].track_id == 3
    assert spans[0].start_frame == 0 and spans[0].end_frame == 5
    assert spans[0].n_samples == 6


def test_contested_ball_counts_for_both_players():
    """Being near the ball is enough — the target needn't be the *nearest* player.

    Requiring nearest-player dropped every contested moment (a tackle, a
    challenge, pressing an opponent) whenever the opponent was fractionally
    closer, which is exactly the footage worth cutting. Both lanes here are
    inside the radius, so both are on the ball.
    """
    ball = _ball({f: (100.0, 200.0) for f in range(6)})
    tracks = _tracks({
        3: {f: (110.0, 205.0) for f in range(6)},   # nearest
        9: {f: (160.0, 205.0) for f in range(6)},   # contesting, 60px away
    })
    assert len(select_on_ball_spans(ball, tracks, {9})) == 1
    assert len(select_on_ball_spans(ball, tracks, {3})) == 1


def test_radius_still_gates_a_player_merely_in_frame():
    """The relaxed rule is paid for by a tight radius: a fly-by doesn't count."""
    ball = _ball({f: (100.0, 200.0) for f in range(6)})
    tracks = _tracks({
        3: {f: (110.0, 205.0) for f in range(6)},   # on the ball
        9: {f: (260.0, 205.0) for f in range(6)},   # 160px — in play, not in it
    })
    assert select_on_ball_spans(ball, tracks, {9}) == []
    # The old 200px default would have let that fly-by through.
    assert len(select_on_ball_spans(ball, tracks, {9}, max_ball_dist_px=200)) == 1


def test_distance_gate_excludes_a_lone_distant_player():
    ball = _ball({f: (100.0, 200.0) for f in range(6)})
    tracks = _tracks({3: {f: (900.0, 800.0) for f in range(6)}})
    assert select_on_ball_spans(ball, tracks, {3}) == []
    assert len(select_on_ball_spans(ball, tracks, {3}, max_ball_dist_px=2000)) == 1


def test_offscreen_ball_frames_are_skipped():
    ball = _ball({0: (100.0, 200.0), 1: None, 2: None, 3: (100.0, 200.0)})
    tracks = _tracks({3: {f: (105.0, 205.0) for f in range(4)}})
    spans = select_on_ball_spans(ball, tracks, {3})
    # A 0.2s ball gap is under max_gap_s, so this stays one span of 2 samples.
    assert len(spans) == 1 and spans[0].n_samples == 2


def test_long_gap_splits_into_two_spans():
    frames = list(range(4)) + list(range(40, 44))  # 3.6s apart at 10fps
    ball = _ball({f: (100.0, 200.0) for f in frames})
    tracks = _tracks({3: {f: (105.0, 205.0) for f in frames}})
    assert len(select_on_ball_spans(ball, tracks, {3})) == 2


def test_lane_handoff_stays_one_span_carrying_both_lanes():
    """One continuous touch across a lane handoff must not split into two clips."""
    ball = _ball({f: (100.0, 200.0) for f in range(8)})
    tracks = _tracks({
        3: {f: (105.0, 205.0) for f in range(4)},
        11: {f: (105.0, 205.0) for f in range(4, 8)},
    })
    (span,) = select_on_ball_spans(ball, tracks, {3, 11})
    assert span.start_frame == 0 and span.end_frame == 7
    assert span.track_ids == (3, 11)


def test_span_is_tagged_with_the_closest_lane():
    ball = _ball({f: (100.0, 200.0) for f in range(8)})
    tracks = _tracks({
        3: {f: (160.0, 200.0) for f in range(4)},    # 60px away
        11: {f: (110.0, 200.0) for f in range(4, 8)},  # 10px away — closer
    })
    (span,) = select_on_ball_spans(ball, tracks, {3, 11})
    assert span.track_id == 11
    assert span.min_dist_px == 10.0


def test_incidental_single_sample_span_is_dropped():
    ball = _ball({0: (100.0, 200.0), 60: (100.0, 200.0)})
    tracks = _tracks({3: {0: (105.0, 205.0), 60: (105.0, 205.0)}})
    assert select_on_ball_spans(ball, tracks, {3}) == []


def test_a_multi_sample_span_still_has_to_clear_min_span_s():
    """Several samples are not a touch if they cover no time.

    At a dense detection rate three consecutive frames are 0.07 s, and a
    duplicate detection box lives exactly that long. The old test exempted any
    span of two or more samples from ``min_span_s``, which put 0.07 s "touches"
    into player reels once the run moved from 5 fps to 30.
    """
    ball = _ball({f: (100.0, 200.0) for f in range(3)})
    tracks = _tracks({3: {f: (105.0, 205.0) for f in range(3)}})
    assert select_on_ball_spans(ball, tracks, {3}, min_span_s=0.4) == []
    # The same three samples do make a span when the bar is set below them.
    assert len(select_on_ball_spans(ball, tracks, {3}, min_span_s=0.1)) == 1


# --- spans -> events ------------------------------------------------------

def _span(tid=3, start=10, end=25):
    return OnBallSpan(track_id=tid, start_frame=start, end_frame=end,
                      start_s=start / FPS, end_s=end / FPS,
                      n_samples=end - start + 1, min_dist_px=12.0)


def test_event_shape_matches_the_clip_pipeline():
    (event,) = spans_to_events([_span()])
    assert event["label"] == ON_BALL_LABEL
    assert event["track_id"] == 3
    assert event["timestamp_s"] == 1.0
    assert event["duration_s"] == 1.5
    assert event["team"] is None


def test_team_is_stamped_from_the_tracks_teams_block():
    (event,) = spans_to_events([_span()], track_teams={"3": "black"})
    assert event["team"] == "black"


def test_team_filter_drops_other_teams_and_unknown_lanes():
    spans = [_span(tid=3), _span(tid=9), _span(tid=4)]
    teams = {"3": "black", "9": "white"}  # lane 4 has no team
    events = spans_to_events(spans, track_teams=teams, team="black")
    assert [e["track_id"] for e in events] == [3]
    # Case-insensitive, matching filter_events.
    assert len(spans_to_events(spans, track_teams=teams, team="BLACK")) == 1


# --- halo across a lane handoff -------------------------------------------

def _box(x, y):
    """A 20x40 bbox with its foot point at ``(x, y)``."""
    return [x - 10.0, y - 40.0, x + 10.0, y]


def test_halo_follows_every_lane_in_a_span():
    """The spotlight must not drop out when the player changes lane mid-clip."""
    halo_tracks = {3: [(0, _box(100, 200)), (1, _box(105, 200))],
                   11: [(4, _box(120, 200)), (5, _box(125, 200))]}
    event = {"track_id": 11, "track_ids": [3, 11]}
    samples = halo_samples_for(event, halo_tracks)
    assert [f for f, _ in samples] == [0, 1, 4, 5]


def test_two_anchor_lanes_alive_at_once_do_not_both_halo():
    """Anchors are vetted like any other candidate — one player, one halo.

    Two lanes of one name alive in the same frames means at least one is another
    child (issue #28). Accepting both let the per-frame dedupe pick between them
    arbitrarily, which is the strobing halo. The longer lane seeds; the
    overlapping one is dropped rather than blended in.
    """
    halo_tracks = {3: [(f, _box(100 + f, 200)) for f in range(10)],
                   11: [(4, _box(900, 700)), (5, _box(905, 700))]}
    samples = halo_samples_for({"track_id": 11, "track_ids": [3, 11]}, halo_tracks)
    assert [f for f, _ in samples] == list(range(10))


def test_a_short_anchor_does_not_outrank_a_longer_one():
    """The longest anchor seeds the halo, not whichever id came first."""
    halo_tracks = {3: [(4, _box(900, 700))],
                   11: [(f, _box(100 + f, 200)) for f in range(10)]}
    samples = halo_samples_for({"track_id": 3, "track_ids": [3, 11]}, halo_tracks)
    assert [f for f, _ in samples] == list(range(10))


def test_halo_falls_back_to_the_single_track_id():
    halo_tracks = {3: [(0, "a")], 11: [(4, "c")]}
    assert halo_samples_for({"track_id": 3}, halo_tracks) == [(0, "a")]


def test_halo_is_none_when_no_lane_has_boxes():
    assert halo_samples_for({"track_id": 99, "track_ids": [99]}, {3: [(0, "a")]}) is None
    assert halo_samples_for({"track_id": 3}, None) is None


# --- reel windows ---------------------------------------------------------

def test_detector_event_gets_the_fixed_window():
    start, duration = _reel_window({"timestamp_s": 100.0})
    assert (start, duration) == (95.0, 20.0)


def test_on_ball_span_window_tracks_its_duration():
    """A short touch and a long dribble must not both become 20s of footage.

    The padding is passed explicitly rather than left to the defaults: those are
    a *policy* and get retuned (c1a0e9e raised post_s from 2.5 to 4.0, and this
    test failed for weeks pinning the old constant). What must not change is the
    rule — a span's window is its own duration plus the lead-in and the trail.
    """
    pre, post = 5.0, 4.0
    _, short = _reel_window({"timestamp_s": 100.0, "duration_s": 1.0},
                            pre_s=pre, post_s=post)
    _, long_ = _reel_window({"timestamp_s": 100.0, "duration_s": 30.0},
                            pre_s=pre, post_s=post)
    assert short < long_
    assert short == pre + 1.0 + post
    assert long_ == pre + 30.0 + post


def test_window_start_is_clamped_at_zero():
    start, _ = _reel_window({"timestamp_s": 1.0, "duration_s": 2.0})
    assert start == 0.0


# --- fallback gating ------------------------------------------------------

def _run(tmp_path, *, ball=True, tracks=True):
    if ball:
        (tmp_path / "ball_track.json").write_text(
            json.dumps(_ball({f: (100.0, 200.0) for f in range(6)})))
    if tracks:
        (tmp_path / "tracks.json").write_text(json.dumps(
            _tracks({3: {f: (105.0, 205.0) for f in range(6)}}, teams={"3": "black"})))
    return tmp_path


def _args(**kw):
    base = dict(on_ball=True, on_ball_force=False, track=None, team=None,
                on_ball_dist=200.0, on_ball_min_span=0.4)
    base.update(kw)
    return SimpleNamespace(**base)


def test_fallback_fires_for_a_player_with_no_detector_events(tmp_path):
    events = _on_ball_fallback(_args(), _run(tmp_path), {3}, None)
    assert len(events) == 1 and events[0]["label"] == ON_BALL_LABEL


def test_fallback_is_suppressed_by_an_explicit_event_label(tmp_path):
    """--events pass must not silently return touches instead."""
    assert _on_ball_fallback(_args(), _run(tmp_path), {3}, ["pass"]) == []


def test_on_ball_force_overrides_the_label_suppression(tmp_path):
    events = _on_ball_fallback(_args(on_ball_force=True), _run(tmp_path), {3}, ["pass"])
    assert len(events) == 1


def test_no_on_ball_disables_the_fallback(tmp_path):
    assert _on_ball_fallback(_args(on_ball=False), _run(tmp_path), {3}, None) == []


def test_a_bare_team_query_anchors_on_every_lane_of_that_kit(tmp_path):
    """"When was one of ours on the ball" needs no identity, so it is answerable.

    This used to be refused as "not a player query". But identity is where the
    pipeline loses most of the football — 1,831 s of Saints touches against 365 s
    over named lanes on the 30 fps U14G run — so a team reel is the widest true
    view of a match we can cut, and it is the control the player reels are read
    against.
    """
    assert len(_on_ball_fallback(_args(team="black"), _run(tmp_path), None, None)) == 1


def test_a_team_with_no_lanes_in_that_kit_falls_back_to_nothing(tmp_path):
    assert _on_ball_fallback(_args(team="green"), _run(tmp_path), None, None) == []


def test_no_selection_at_all_still_means_no_fallback(tmp_path):
    """No player, no track, no kit — there is nothing to anchor spans on."""
    assert _on_ball_fallback(_args(), _run(tmp_path), None, None) == []


def test_raw_track_id_anchors_the_fallback(tmp_path):
    events = _on_ball_fallback(_args(track=3), _run(tmp_path), None, None)
    assert len(events) == 1


def test_team_filter_applies_to_fallback_spans(tmp_path):
    assert _on_ball_fallback(_args(team="white"), _run(tmp_path), {3}, None) == []
    assert len(_on_ball_fallback(_args(team="black"), _run(tmp_path), {3}, None)) == 1


def test_missing_ball_track_returns_nothing(tmp_path, capsys):
    """Runs predating ball_track.json must say so, not read as 'player did nothing'."""
    assert _on_ball_fallback(_args(), _run(tmp_path, ball=False), {3}, None) == []
    assert "ball_track.json" in capsys.readouterr().out
