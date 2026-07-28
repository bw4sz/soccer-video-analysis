"""Tests for the hand-drawn 'our pitch' region."""

from __future__ import annotations

import json

import numpy as np
import pytest

from soccer_vision.detection.pitch_region import (
    PitchKeyframe,
    PitchRegion,
    PitchRegionError,
    PitchRegionTracker,
    apply_affine,
    compose_affine,
    estimate_similarity,
    parse_polygon_spec,
    points_in_polygon,
    polygon_below_line,
)

SQUARE = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])


# -- spec parsing -------------------------------------------------------------


def test_parse_normalised_spec():
    poly = parse_polygon_spec("0.05,0.45 0.95,0.42 0.98,0.99 0.02,0.99")
    assert poly.shape == (4, 2)
    assert poly[0].tolist() == [0.05, 0.45]


def test_parse_pixel_spec_normalises():
    poly = parse_polygon_spec("96,486 1824,453 1920,1069", width=1920, height=1080)
    assert poly[0] == pytest.approx([0.05, 0.45], abs=1e-3)


def test_pixel_spec_without_frame_size_is_an_error():
    with pytest.raises(PitchRegionError, match="frame size"):
        parse_polygon_spec("96,486 1824,453 1920,1069")


def test_too_few_corners_rejected():
    with pytest.raises(PitchRegionError, match="at least 3"):
        parse_polygon_spec("0.1,0.1 0.9,0.9")


def test_semicolons_and_newlines_separate_points():
    assert parse_polygon_spec("0.1,0.1; 0.9,0.1\n0.9,0.9").shape == (3, 2)


def test_explicit_normalized_mode_allows_off_frame_coordinates():
    # -1 and 2 are deliberate (a region wider than the frame), not pixels.
    poly = parse_polygon_spec("-1.0,0.4 2.0,0.36 2.0,2.0", mode="normalized")
    assert poly[1].tolist() == [2.0, 0.36]


def test_explicit_pixel_mode_normalises_small_values():
    poly = parse_polygon_spec("0,0 0.5,0.5 960,540", width=1920, height=1080,
                              mode="pixels")
    assert poly[2] == pytest.approx([0.5, 0.5])


# -- below-a-line shorthand ---------------------------------------------------


def test_polygon_below_line_extends_past_the_frame():
    poly = polygon_below_line((0.0, 0.40), (1.0, 0.36))
    assert poly[:, 0].min() == -1.0
    assert poly[:, 0].max() == 2.0
    assert poly[:, 1].max() == 2.0


def test_polygon_below_line_extrapolates_the_slope():
    poly = polygon_below_line((0.0, 0.40), (1.0, 0.30))  # -0.1 per unit x
    assert poly[0][1] == pytest.approx(0.50)  # at x = -1
    assert poly[1][1] == pytest.approx(0.20)  # at x = 2


def test_below_line_keeps_the_near_side_only():
    poly = polygon_below_line((0.0, 0.40), (1.0, 0.40))
    pts = np.array([[0.5, 0.6], [0.5, 0.2], [1.4, 0.6]])
    assert points_in_polygon(pts, poly).tolist() == [True, False, True]


def test_vertical_boundary_line_rejected():
    with pytest.raises(PitchRegionError, match="differ in x"):
        polygon_below_line((0.5, 0.1), (0.5, 0.9))


# -- containment --------------------------------------------------------------


def test_points_in_polygon_square():
    poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    pts = np.array([[5.0, 5.0], [-1.0, 5.0], [11.0, 5.0], [5.0, 20.0]])
    assert points_in_polygon(pts, poly).tolist() == [True, False, False, False]


def test_points_in_polygon_handles_concave_shapes():
    # An L: the notch at (8, 8) is outside even though it is inside the bbox.
    poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [5.0, 5.0],
                     [5.0, 10.0], [0.0, 10.0]])
    inside = points_in_polygon(np.array([[2.0, 2.0], [8.0, 8.0], [2.0, 8.0]]), poly)
    assert inside.tolist() == [True, False, True]


def test_empty_point_set():
    assert points_in_polygon(np.zeros((0, 2)), SQUARE).shape == (0,)


# -- keyframes ----------------------------------------------------------------


def test_single_keyframe_holds_everywhere():
    region = PitchRegion.from_points(SQUARE, frame=0)
    assert np.allclose(region.polygon_at(0), SQUARE)
    assert np.allclose(region.polygon_at(99999), SQUARE)


def test_keyframes_interpolate_linearly():
    later = SQUARE + np.array([0.1, 0.0])
    region = PitchRegion(keyframes=[PitchKeyframe(0, SQUARE), PitchKeyframe(100, later)])
    mid = region.polygon_at(50)
    assert np.allclose(mid, SQUARE + np.array([0.05, 0.0]))


def test_keyframes_hold_not_extrapolate_outside_their_range():
    later = SQUARE + np.array([0.1, 0.0])
    region = PitchRegion(keyframes=[PitchKeyframe(0, SQUARE), PitchKeyframe(100, later)])
    assert np.allclose(region.polygon_at(-10), SQUARE)
    assert np.allclose(region.polygon_at(1000), later)


def test_keyframes_are_sorted_by_frame():
    later = SQUARE + 0.1
    region = PitchRegion(keyframes=[PitchKeyframe(100, later), PitchKeyframe(0, SQUARE)])
    assert [k.frame for k in region.keyframes] == [0, 100]


def test_mismatched_corner_counts_rejected():
    with pytest.raises(PitchRegionError, match="same corners"):
        PitchRegion(keyframes=[PitchKeyframe(0, SQUARE), PitchKeyframe(10, SQUARE[:3])])


def test_with_keyframe_replaces_at_same_frame():
    region = PitchRegion.from_points(SQUARE, frame=0).with_keyframe(SQUARE + 0.05, 0)
    assert len(region.keyframes) == 1
    assert np.allclose(region.keyframes[0].polygon, SQUARE + 0.05)


# -- pixels, margin, round-trip ----------------------------------------------


def test_pixel_polygon_scales_to_frame_shape():
    region = PitchRegion.from_points(SQUARE)
    px = region.pixel_polygon((1080, 1920))
    assert px[0] == pytest.approx([0.2 * 1920, 0.2 * 1080])


def test_pixel_polygon_is_resolution_independent():
    region = PitchRegion.from_points(SQUARE)
    hd = region.pixel_polygon((1080, 1920)) / np.array([1920, 1080])
    sd = region.pixel_polygon((360, 640)) / np.array([640, 360])
    assert np.allclose(hd, sd)


def test_margin_grows_the_polygon_about_its_centre():
    region = PitchRegion.from_points(SQUARE, margin=0.1)
    px = region.pixel_polygon((100, 100))
    centre = px.mean(axis=0)
    assert centre == pytest.approx([50.0, 50.0])
    assert px[:, 0].min() < 20.0  # grown outward past the un-margined 0.2


def test_json_round_trip(tmp_path):
    region = PitchRegion(
        keyframes=[PitchKeyframe(0, SQUARE), PitchKeyframe(900, SQUARE + 0.05)],
        video="match.mp4", width=1920, height=1080, track_pan=False, margin=0.02,
        notes="north pitch",
    )
    path = region.save(tmp_path / "pitch.json")
    back = PitchRegion.load(path)
    assert back.track_pan is False
    assert back.margin == 0.02
    assert back.video == "match.mp4"
    assert back.notes == "north pitch"
    assert [k.frame for k in back.keyframes] == [0, 900]
    assert np.allclose(back.keyframes[0].polygon, SQUARE)


def test_load_rejects_a_file_with_no_keyframes(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"video": "m.mp4"}))
    with pytest.raises(PitchRegionError, match="keyframes"):
        PitchRegion.load(path)


# -- camera motion ------------------------------------------------------------


def _textured_frame(w=640, h=360, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    return np.repeat(img[:, :, None], 3, axis=2)


def test_estimate_similarity_recovers_a_known_shift():
    cv2 = pytest.importorskip("cv2")
    ref = _textured_frame()
    M_true = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -12.0]])
    cur = cv2.warpAffine(ref, M_true, (ref.shape[1], ref.shape[0]))
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    M = estimate_similarity(gray, cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), work_width=640)
    assert M is not None
    assert M[0, 2] == pytest.approx(25.0, abs=3.0)
    assert M[1, 2] == pytest.approx(-12.0, abs=3.0)


def test_estimate_similarity_rejects_an_implausible_jump():
    cv2 = pytest.importorskip("cv2")
    ref = _textured_frame()
    # Shift by more than a quarter frame: not something these cameras do, so the
    # estimate is refused rather than moving the pitch across the screen.
    M_true = np.array([[1.0, 0.0, 300.0], [0.0, 1.0, 0.0]])
    cur = cv2.warpAffine(ref, M_true, (ref.shape[1], ref.shape[0]))
    M = estimate_similarity(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY))
    assert M is None


def test_apply_affine_translates_the_polygon():
    poly = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0]])
    M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    assert np.allclose(apply_affine(poly, M), poly + np.array([5.0, -3.0]))


def test_compose_affine_matches_applying_both_in_order():
    a = np.array([[1.1, -0.2, 5.0], [0.2, 1.1, -3.0]])
    b = np.array([[0.9, 0.1, -7.0], [-0.1, 0.9, 2.0]])
    pts = np.array([[10.0, 10.0], [40.0, 25.0], [3.0, -8.0]])
    assert np.allclose(apply_affine(apply_affine(pts, a), b),
                       apply_affine(pts, compose_affine(a, b)))


# -- tracker ------------------------------------------------------------------


def test_tracker_without_frames_returns_the_static_polygon():
    tracker = PitchRegionTracker(PitchRegion.from_points(SQUARE), track_pan=True)
    poly = tracker.polygon_for(1234, frame_shape=(1080, 1920))
    assert poly[0] == pytest.approx([0.2 * 1920, 0.2 * 1080])


def test_tracker_follows_a_panning_camera():
    cv2 = pytest.importorskip("cv2")
    ref = _textured_frame()
    tracker = PitchRegionTracker(PitchRegion.from_points(SQUARE), track_pan=True)
    tracker._refs[0] = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)  # stand in for the video read

    shifted = cv2.warpAffine(ref, np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 0.0]]),
                             (ref.shape[1], ref.shape[0]))
    poly = tracker.polygon_for(300, frame=shifted)
    static = PitchRegion.from_points(SQUARE).pixel_polygon(shifted.shape)
    assert np.allclose(poly[:, 0] - static[:, 0], 30.0, atol=3.0)
    assert tracker.n_tracked == 1


def test_tracker_holds_the_last_transform_when_matching_fails():
    cv2 = pytest.importorskip("cv2")
    ref = _textured_frame()
    tracker = PitchRegionTracker(PitchRegion.from_points(SQUARE), track_pan=True)
    tracker._refs[0] = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    shifted = cv2.warpAffine(ref, np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 0.0]]),
                             (ref.shape[1], ref.shape[0]))
    good = tracker.polygon_for(300, frame=shifted)

    blank = np.zeros_like(ref)  # featureless: no estimate possible
    held = tracker.polygon_for(600, frame=blank)
    assert np.allclose(held, good)
    assert tracker.n_failed >= 1


def test_tracker_chains_between_reanchors():
    """The keyframe reference goes stale within seconds on real footage, so the
    polygon has to be carried frame to frame and composed."""
    cv2 = pytest.importorskip("cv2")
    ref = _textured_frame()
    tracker = PitchRegionTracker(PitchRegion.from_points(SQUARE), track_pan=True,
                                 reanchor_every=1000)
    tracker._refs[0] = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    poly = None
    for i, dx in enumerate([10.0, 20.0, 30.0]):
        cur = cv2.warpAffine(ref, np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]]),
                             (ref.shape[1], ref.shape[0]))
        poly = tracker.polygon_for(i * 100, frame=cur)

    assert tracker.n_reanchored == 1  # only the first call
    assert tracker.n_chained == 2
    static = PitchRegion.from_points(SQUARE).pixel_polygon(ref.shape)
    assert np.allclose(poly[:, 0] - static[:, 0], 30.0, atol=4.0)


def test_tracker_can_turn_pan_tracking_off():
    cv2 = pytest.importorskip("cv2")
    ref = _textured_frame()
    tracker = PitchRegionTracker(PitchRegion.from_points(SQUARE), track_pan=False)
    shifted = cv2.warpAffine(ref, np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 0.0]]),
                             (ref.shape[1], ref.shape[0]))
    assert np.allclose(tracker.polygon_for(300, frame=shifted),
                       PitchRegion.from_points(SQUARE).pixel_polygon(shifted.shape))


# -- detection filtering ------------------------------------------------------


def test_filter_spectators_uses_the_region_when_given_one():
    sv = pytest.importorskip("supervision")
    from soccer_vision.detection.field_filter import filter_spectators

    # Feet at (425, 500) — inside the 0.2-0.8 box of a 1000x1000 frame — and at
    # (425, 950), below it on the touchline.
    dets = sv.Detections(
        xyxy=np.array([[400.0, 400.0, 450.0, 500.0], [400.0, 900.0, 450.0, 950.0]]),
        class_id=np.array([0, 0]),
        confidence=np.array([0.9, 0.9]),
    )
    tracker = PitchRegionTracker(PitchRegion.from_points(SQUARE), track_pan=False)
    kept = filter_spectators(dets, (1000, 1000), region_tracker=tracker)
    assert len(kept) == 1
    assert kept.xyxy[0][3] == 500.0


def test_filter_spectators_falls_back_to_the_hull_without_a_region():
    sv = pytest.importorskip("supervision")
    from soccer_vision.detection.field_filter import filter_spectators

    dets = sv.Detections(
        xyxy=np.array([[490.0, 400.0, 510.0, 500.0]]),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
    )
    assert len(filter_spectators(dets, (1000, 1000))) == 1
