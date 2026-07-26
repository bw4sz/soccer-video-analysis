"""Team-colour assignment tests with synthetic jersey patches."""

import numpy as np

from soccer_vision.tracking.teams import (
    TeamClassifier,
    assign_kits_to_clusters,
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


def test_kits_name_navy_black_kit_as_black():
    # The regression from issue #11: a black kit a camera renders as mid-value
    # navy (blue channel dominant) is named "blue" by the HSV heuristic, but the
    # profile-kit path should name it "black".
    navy_black = np.array([152, 111, 87], float)  # HSV ~ (109,109,152)
    white = np.array([200, 205, 205], float)
    assert name_bgr_colour(navy_black) == "blue"  # the bug the heuristic still has
    mapping = assign_kits_to_clusters([navy_black, white], ["black", "white"])
    assert mapping == {0: "black", 1: "white"}
    # one-to-one and order-robust: the two clusters never collapse to one name
    swapped = assign_kits_to_clusters([white, navy_black], ["black", "white"])
    assert swapped == {0: "white", 1: "black"}
    # no declared kits -> empty, so the caller falls back to the heuristic
    assert assign_kits_to_clusters([navy_black, white], []) == {}


def test_classifier_fit_with_kits_overrides_heuristic():
    clf = TeamClassifier(min_samples=2)
    navy_black = _player_frame((87, 111, 152))  # BGR of the navy-black kit
    white = _player_frame((205, 205, 200))
    box = (10, 10, 90, 90)
    for _ in range(3):
        clf.add_sample(1, navy_black, box)
        clf.add_sample(2, navy_black, box)
        clf.add_sample(3, white, box)
        clf.add_sample(4, white, box)
    clf.fit(kits=["black", "white"])
    assert set(clf.team_names().values()) == {"black", "white"}
    assert clf.predict(1) == "black"
    assert clf.predict(3) == "white"


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
