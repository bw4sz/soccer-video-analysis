"""Name/number → track-lane resolution and jersey event filtering."""

from soccer_vision.events.select import filter_events
from soccer_vision.identify.resolve import tracks_for
from soccer_vision.profiles.loader import get_jersey_by_name

PROFILE = {"roster": [{"name": "Noah", "jersey": 7}, {"name": "Bryce", "jersey": 9}]}

# Two ByteTrack lanes (3 and 11) both resolved to jersey 7 — the fragmentation
# case that jersey identity fixes.
JERSEYS = {
    "tracks": {
        "3": {"jersey": 7, "confidence": 0.8, "name": "Noah"},
        "11": {"jersey": 7, "confidence": 0.7, "name": "Noah"},
        "5": {"jersey": 9, "confidence": 0.9, "name": "Bryce"},
        "8": {"jersey": None, "confidence": 0.0, "name": None},
    }
}


def test_number_unions_fragmented_lanes():
    assert tracks_for(JERSEYS, number=7) == {3, 11}


def test_name_via_roster():
    assert tracks_for(JERSEYS, name="Noah", profile=PROFILE) == {3, 11}
    assert tracks_for(JERSEYS, name="bryce", profile=PROFILE) == {5}


def test_name_fallback_to_stored_name_without_profile():
    assert tracks_for(JERSEYS, name="Noah") == {3, 11}


def test_unknown_name_and_number():
    assert tracks_for(JERSEYS, name="Ghost", profile=PROFILE) == set()
    assert tracks_for(JERSEYS, number=99) == set()


def test_get_jersey_by_name_case_insensitive():
    assert get_jersey_by_name(PROFILE, "noah") == 7
    assert get_jersey_by_name(PROFILE, "Missing") is None


def test_filter_events_by_track_id_set():
    events = [
        {"label": "pass", "track_id": 3},
        {"label": "pass", "track_id": 11},
        {"label": "pass", "track_id": 5},
        {"label": "pass", "track_id": None},
    ]
    got = filter_events(events, track_ids={3, 11})
    assert [e["track_id"] for e in got] == [3, 11]
