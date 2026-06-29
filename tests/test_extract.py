"""Clip timecode math tests."""

from soccer_vision.metrics.distance import detect_sprints, distance_covered


def test_distance_covered_straight_line():
    positions = [(0.0, 0.0), (3.0, 4.0)]
    assert abs(distance_covered(positions) - 5.0) < 0.001


def test_distance_covered_zero():
    positions = [(5.0, 5.0), (5.0, 5.0)]
    assert distance_covered(positions) == 0.0


def test_distance_covered_multiple_segments():
    positions = [(0, 0), (3, 0), (3, 4)]
    assert abs(distance_covered(positions) - 7.0) < 0.001


def test_detect_sprints():
    # 10 frames at 10 fps, moving 6 m/frame = 60 m/s >> 5 m/s threshold
    positions = [(i * 6.0, 0) for i in range(10)]
    sprints = detect_sprints(positions, fps=10.0, speed_threshold_ms=5.0, min_duration_s=0.5)
    assert len(sprints) >= 1


def test_detect_sprints_none():
    # Stationary
    positions = [(0.0, 0.0)] * 10
    sprints = detect_sprints(positions, fps=10.0, speed_threshold_ms=5.0, min_duration_s=0.5)
    assert len(sprints) == 0
