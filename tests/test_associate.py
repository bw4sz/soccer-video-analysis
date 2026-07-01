"""Event→player/team association and clip-selection filter tests."""

from soccer_vision.events.associate import associate_events
from soccer_vision.events.select import filter_events


class _StubTeams:
    """Minimal TeamClassifier stand-in mapping track ids to colours."""

    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, track_id):
        return self.mapping.get(track_id)


def test_associate_tags_nearest_track_and_team():
    events = [{"label": "tackle", "frame": 10, "field_x": 20.0, "field_y": 15.0}]
    frame_players = {
        10: {
            7: {"field_x": 20.5, "field_y": 15.0, "pixel_x": 0, "pixel_y": 0},
            9: {"field_x": 40.0, "field_y": 30.0, "pixel_x": 0, "pixel_y": 0},
        }
    }
    teams = _StubTeams({7: "blue", 9: "white"})

    associate_events(events, frame_players, teams)
    assert events[0]["track_id"] == 7
    assert events[0]["team"] == "blue"


def test_associate_uses_nearby_frame_window():
    events = [{"label": "corner_kick", "frame": 10, "field_x": 5.0, "field_y": 5.0}]
    frame_players = {12: {3: {"field_x": 5.2, "field_y": 5.0}}}  # 2 frames later
    associate_events(events, frame_players, None, search_frames=3)
    assert events[0]["track_id"] == 3


def test_associate_none_when_no_player_near():
    events = [{"label": "goal_kick", "frame": 1, "field_x": 0.0, "field_y": 0.0}]
    frame_players = {1: {5: {"field_x": 50.0, "field_y": 30.0}}}
    associate_events(events, frame_players, None, max_distance_field_m=5.0)
    assert events[0]["track_id"] is None
    assert events[0]["team"] is None


def test_associate_falls_back_to_pixel_space():
    events = [{"label": "tackle", "frame": 2, "pixel_x": 300.0, "pixel_y": 200.0}]
    frame_players = {2: {4: {"pixel_x": 305.0, "pixel_y": 205.0, "field_x": None,
                             "field_y": None}}}
    associate_events(events, frame_players, None)
    assert events[0]["track_id"] == 4


def test_filter_events_composable():
    events = [
        {"label": "tackle", "team": "blue", "track_id": 7},
        {"label": "tackle", "team": "white", "track_id": 9},
        {"label": "goal_kick", "team": "blue", "track_id": 7},
    ]
    assert len(filter_events(events, team="blue")) == 2
    assert len(filter_events(events, label="tackle")) == 2
    assert len(filter_events(events, label="tackle", team="blue")) == 1
    assert len(filter_events(events, track_id=9)) == 1
    # team match is case-insensitive
    assert len(filter_events(events, team="BLUE")) == 2
