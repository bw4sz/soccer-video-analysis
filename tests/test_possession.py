"""Possession computation tests with known positions."""

from soccer_vision.metrics.possession import assign_possession_frame, compute_possession


def test_assign_nearest_player():
    ball = (10.0, 10.0)
    players = {1: (11.0, 10.0), 2: (40.0, 20.0)}
    teams = {1: "home", 2: "away"}
    assert assign_possession_frame(ball, players, teams) == "home"


def test_assign_no_player_nearby():
    ball = (10.0, 10.0)
    players = {1: (40.0, 40.0)}
    teams = {1: "home"}
    assert assign_possession_frame(ball, players, teams, max_distance_m=3.0) is None


def test_compute_possession_even():
    frames = [
        {"ball": (10.0, 10.0), "players": {1: (11.0, 10.0)}, "teams": {1: "home"}},
        {"ball": (40.0, 20.0), "players": {2: (40.5, 20.0)}, "teams": {2: "away"}},
    ]
    result = compute_possession(frames)
    assert result["home"] == 50.0
    assert result["away"] == 50.0


def test_compute_possession_dominant():
    frames = [
        {"ball": (10.0, 10.0), "players": {1: (11.0, 10.0)}, "teams": {1: "home"}},
        {"ball": (12.0, 10.0), "players": {1: (12.5, 10.0)}, "teams": {1: "home"}},
        {"ball": (14.0, 10.0), "players": {1: (14.5, 10.0)}, "teams": {1: "home"}},
        {"ball": (40.0, 20.0), "players": {2: (40.5, 20.0)}, "teams": {2: "away"}},
    ]
    result = compute_possession(frames)
    assert result["home"] == 75.0
    assert result["away"] == 25.0


def test_compute_possession_no_ball():
    frames = [{"ball": None, "players": {}, "teams": {}}]
    result = compute_possession(frames)
    assert result == {}
