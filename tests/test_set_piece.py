"""Set-piece detection tests with synthetic ball positions."""

from soccer_vision.events.set_piece import (
    detect_all_set_pieces,
    detect_corner_kicks,
    detect_goal_kicks,
    detect_throw_ins,
    in_corner_zone,
    in_goal_zone,
    is_stationary,
    near_touchline,
)


def test_in_goal_zone_left():
    assert in_goal_zone(2.0, 18.0) == "left"


def test_in_goal_zone_right():
    assert in_goal_zone(52.0, 18.0) == "right"


def test_in_goal_zone_midfield():
    assert in_goal_zone(27.5, 18.0) is None


def test_in_corner_zone():
    assert in_corner_zone(0.5, 0.5) is True
    assert in_corner_zone(27.5, 18.0) is False


def test_near_touchline():
    assert near_touchline(10.0, 1.0) is True  # near top touchline
    assert near_touchline(10.0, 35.5) is True  # near bottom touchline
    assert near_touchline(10.0, 18.0) is False  # midfield


def test_is_stationary():
    assert is_stationary([(100, 200), (102, 201), (101, 200)], thresh_px=40) is True
    assert is_stationary([(100, 200), (200, 300)], thresh_px=40) is False
    assert is_stationary([(100, 200)], thresh_px=40) is False


def test_detect_goal_kicks():
    positions = []
    for i in range(10):
        positions.append({
            "frame": i * 10,
            "timestamp_s": i * 0.33,
            "pixel_x": 100.0 + i * 0.5,
            "pixel_y": 200.0,
            "field_x": 3.0,  # inside goal zone
            "field_y": 18.0,
            "confidence": 0.8,
        })

    candidates = detect_goal_kicks(positions, stationary_frames=3, stationary_px=40)
    assert len(candidates) >= 1
    assert candidates[0]["label"] == "goal_kick"
    assert candidates[0]["goal_zone"] == "left"


def test_detect_corner_kicks():
    positions = []
    for i in range(5):
        positions.append({
            "frame": i * 10,
            "timestamp_s": i * 0.33,
            "pixel_x": 50.0,
            "pixel_y": 50.0,
            "field_x": 0.5,  # corner zone
            "field_y": 0.5,
            "confidence": 0.7,
        })

    candidates = detect_corner_kicks(positions, stationary_frames=3, stationary_px=40)
    assert len(candidates) >= 1
    assert candidates[0]["label"] == "corner_kick"


def test_detect_throw_ins():
    positions = []
    for i in range(5):
        positions.append({
            "frame": i * 10,
            "timestamp_s": i * 0.33,
            "pixel_x": 300.0,
            "pixel_y": 50.0,
            "field_x": 25.0,  # mid-touchline
            "field_y": 0.5,   # near top touchline
            "confidence": 0.6,
        })

    candidates = detect_throw_ins(positions, stationary_frames=3, stationary_px=40)
    assert len(candidates) >= 1
    assert candidates[0]["label"] == "throw_in"


def test_detect_all_set_pieces_sorted():
    positions = []
    # Goal kick at t=1
    for i in range(4):
        positions.append({
            "frame": (10 + i) * 10,
            "timestamp_s": 1.0 + i * 0.1,
            "pixel_x": 100.0,
            "pixel_y": 200.0,
            "field_x": 3.0,
            "field_y": 18.0,
            "confidence": 0.8,
        })
    # Corner at t=20
    for i in range(4):
        positions.append({
            "frame": (200 + i) * 10,
            "timestamp_s": 20.0 + i * 0.1,
            "pixel_x": 50.0,
            "pixel_y": 50.0,
            "field_x": 0.5,
            "field_y": 0.5,
            "confidence": 0.7,
        })

    events = detect_all_set_pieces(positions, stationary_frames=3, stationary_px=40)
    assert len(events) >= 2
    assert events[0]["timestamp_s"] < events[1]["timestamp_s"]
