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


def test_navy_black_kit_named_black_with_profile_kits():
    """Regression for #11: a black kit photographing as dark navy (t54:
    BGR ~(130,106,94)) must be named 'black' when the profile declares
    kits [black, white] — the HSV heuristic alone calls it 'blue'."""
    from soccer_vision.tracking.teams import assign_kit_names

    navy_black = np.array([130.0, 106.0, 94.0])  # sampled torso, black kit
    white = np.array([137.0, 158.0, 139.0])  # sampled torso, white kit

    # The heuristic is exactly what goes wrong…
    assert name_bgr_colour(navy_black) == "blue"
    # …and the declared kits fix it.
    names = assign_kit_names(np.stack([navy_black, white]), ["black", "white"])
    assert names == ["black", "white"]


def test_assign_kit_names_never_duplicates():
    from soccer_vision.tracking.teams import assign_kit_names

    close_a = np.array([30.0, 30.0, 30.0])
    close_b = np.array([50.0, 50.0, 50.0])  # both nearer black than white
    names = assign_kit_names(np.stack([close_a, close_b]), ["black", "white"])
    assert sorted(names) == ["black", "white"]


def test_assign_kit_names_rejects_unknown_kit():
    from soccer_vision.tracking.teams import assign_kit_names

    assert assign_kit_names(np.array([[0.0, 0.0, 0.0]]), ["chartreuse"]) is None


def test_classifier_uses_profile_kits_end_to_end():
    clf = TeamClassifier(min_samples=2, kits=["black", "white"])
    navy = _player_frame((130, 106, 94))  # black kit as the camera sees it
    white = _player_frame((225, 228, 224))
    for _ in range(3):
        clf.add_sample(1, navy, (10, 10, 90, 90))
        clf.add_sample(2, white, (10, 10, 90, 90))
    clf.fit()
    assert sorted(clf.team_names().values()) == ["black", "white"]
    assert clf.predict(1) == "black"
    assert clf.predict(2) == "white"


def test_classifier_without_kits_keeps_heuristic_naming():
    clf = TeamClassifier(min_samples=2)
    blue = _player_frame((220, 40, 40))
    white = _player_frame((225, 228, 224))
    for _ in range(3):
        clf.add_sample(1, blue, (10, 10, 90, 90))
        clf.add_sample(2, white, (10, 10, 90, 90))
    clf.fit()
    assert sorted(clf.team_names().values()) == ["blue", "white"]


def test_render_preview_montage():
    clf = TeamClassifier(min_samples=2, kits=["black", "white"])
    navy = _player_frame((130, 106, 94))
    white = _player_frame((225, 228, 224))
    for _ in range(3):
        clf.add_sample(1, navy, (10, 10, 90, 90))
        clf.add_sample(2, white, (10, 10, 90, 90))
    clf.fit()
    preview = clf.render_preview()
    assert preview is not None
    assert preview.ndim == 3 and preview.shape[2] == 3
    assert preview.shape[0] > 0 and preview.shape[1] > 0
    # Both team rows rendered.
    assert preview.shape[0] == 2 * (64 + 12)


def test_get_kits_from_profile():
    from soccer_vision.profiles.loader import get_kits

    assert get_kits({"kits": ["Black", " white "]}) == ["black", "white"]
    assert get_kits({}) == []
