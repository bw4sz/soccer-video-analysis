"""Team-colour assignment tests with synthetic jersey patches."""

import numpy as np

from soccer_vision.tracking.teams import (
    TeamClassifier,
    name_bgr_colour,
    sample_jersey_bgr,
)


def _player_frame(bgr, size=100):
    """A frame with a single solid-colour player box centred in view."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_name_primary_colours():
    assert name_bgr_colour(np.array([255, 0, 0])) == "blue"   # BGR blue
    assert name_bgr_colour(np.array([0, 0, 255])) == "red"    # BGR red
    assert name_bgr_colour(np.array([255, 255, 255])) == "white"
    assert name_bgr_colour(np.array([10, 10, 10])) == "black"


def test_sample_jersey_reads_torso():
    frame = _player_frame((200, 50, 50))  # bluish
    colour = sample_jersey_bgr(frame, (10, 10, 90, 90))
    assert colour is not None
    assert name_bgr_colour(colour) == "blue"


def test_sample_jersey_rejects_tiny_box():
    frame = _player_frame((200, 50, 50))
    assert sample_jersey_bgr(frame, (10, 10, 12, 14)) is None


def test_classifier_splits_two_teams_and_names_blue():
    clf = TeamClassifier(min_samples=2)
    blue = _player_frame((220, 40, 40))
    white = _player_frame((240, 240, 240))
    box = (10, 10, 90, 90)

    # tracks 1,2 are blue; tracks 3,4 are white
    for _ in range(3):
        clf.add_sample(1, blue, box)
        clf.add_sample(2, blue, box)
        clf.add_sample(3, white, box)
        clf.add_sample(4, white, box)
    clf.fit()

    names = set(clf.team_names().values())
    assert names == {"blue", "white"}
    assert clf.predict(1) == clf.predict(2)
    assert clf.predict(3) == clf.predict(4)
    assert clf.predict(1) != clf.predict(3)
    assert clf.predict(1) == "blue"


def test_predict_before_fit_raises():
    clf = TeamClassifier()
    try:
        clf.predict(1)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_track_below_min_samples_is_unknown():
    clf = TeamClassifier(min_samples=5)
    blue = _player_frame((220, 40, 40))
    clf.add_sample(1, blue, (10, 10, 90, 90))
    clf.fit()
    assert clf.predict(1) is None
