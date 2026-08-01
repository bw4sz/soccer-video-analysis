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


# --- illumination handling -------------------------------------------------
#
# Regression tests for the shadow failure on low-sun overhead footage: a white
# kit photographed in shade is darker than a black kit in sun, so absolute torso
# lightness put 174 of 188 U14G tracks in one cluster (job 38180242). Grass beside
# a player shares that player's lighting, so the *sign* of torso-minus-turf
# lightness identifies the kit where absolute lightness cannot.

def _pitch_frame(turf_bgr, player_bgr, box=(40, 30, 60, 70), size=100):
    """Turf everywhere, with one solid player rectangle standing on it."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = turf_bgr
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = player_bgr
    return frame


SUN_TURF = (60, 170, 70)     # brightly lit grass
SHADE_TURF = (30, 80, 35)    # the same grass in shadow


def test_turf_pixels_finds_grass_not_kit():
    from soccer_vision.tracking.teams import turf_pixels

    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    patch[:, :5] = SUN_TURF
    patch[:, 5:] = (240, 240, 240)  # white shirt
    mask = turf_pixels(patch)
    assert mask[:, :5].all()
    assert not mask[:, 5:].any()


def test_local_illuminant_none_without_turf():
    from soccer_vision.tracking.teams import estimate_local_illuminant

    frame = _player_frame((200, 50, 50))
    assert estimate_local_illuminant(frame, (10, 10, 90, 90)) is None


def test_local_illuminant_tracks_lighting():
    from soccer_vision.tracking.teams import estimate_local_illuminant

    box = (40, 30, 60, 70)
    sun = estimate_local_illuminant(_pitch_frame(SUN_TURF, (240, 240, 240)), box)
    shade = estimate_local_illuminant(_pitch_frame(SHADE_TURF, (240, 240, 240)), box)
    assert sun is not None and shade is not None
    # It reports the grass it saw, so the two differ by the lighting.
    assert sun[1] > shade[1]


def test_lightness_split_only_when_kits_straddle_turf():
    from soccer_vision.tracking.teams import lightness_split_kits

    assert lightness_split_kits(["black", "white"]) == ("black", "white")
    assert lightness_split_kits(["white", "black"]) == ("black", "white")
    assert lightness_split_kits(["blue", "white"]) == ("blue", "white")
    # both darker than grass -> hue is the separator, so no lightness split
    assert lightness_split_kits(["red", "blue"]) is None
    assert lightness_split_kits(["black"]) is None
    assert lightness_split_kits(None) is None


def test_white_kit_in_shadow_is_not_called_black():
    """The exact failure this machinery exists for.

    A white kit in shade (BGR ~110) is *darker* than a black kit in sun (~120),
    so any absolute-lightness split misassigns it. Judged against the grass each
    player stands on, the white kit is still the lighter of the two.
    """
    clf = TeamClassifier(min_samples=2)
    box = (40, 30, 60, 70)
    white_in_shade = _pitch_frame(SHADE_TURF, (110, 110, 110))
    black_in_sun = _pitch_frame(SUN_TURF, (120, 120, 120))

    for _ in range(3):
        clf.add_sample(1, white_in_shade, box)
        clf.add_sample(2, white_in_shade, box)
        clf.add_sample(3, black_in_sun, box)
        clf.add_sample(4, black_in_sun, box)
    clf.fit(kits=["black", "white"])

    assert clf.predict(1) == clf.predict(2) == "white"
    assert clf.predict(3) == clf.predict(4) == "black"


def test_falls_back_to_clustering_without_turf():
    """No grass in view -> no illuminant -> the original clustering path runs."""
    clf = TeamClassifier(min_samples=2)
    box = (10, 10, 90, 90)
    for _ in range(3):
        clf.add_sample(1, _player_frame((20, 20, 20)), box)
        clf.add_sample(2, _player_frame((20, 20, 20)), box)
        clf.add_sample(3, _player_frame((240, 240, 240)), box)
        clf.add_sample(4, _player_frame((240, 240, 240)), box)
    clf.fit(kits=["black", "white"])

    assert set(clf.team_names().values()) == {"black", "white"}
    assert clf.predict(1) == "black"
    assert clf.predict(3) == "white"


def test_retained_crops_are_globally_capped():
    """Crops feed a 16-tile preview; keeping 4 per track OOM-killed a full match.

    The cost has to be independent of video length — RF-DETR fragments this
    footage into hundreds of lanes a minute, so a per-track budget grows without
    limit (job 38313387, killed at 64 GB 18% into a 60-minute match).
    """
    import numpy as np

    from soccer_vision.tracking.teams import TeamClassifier

    clf = TeamClassifier(keep_crops=4, max_crops=10)
    frame = np.full((200, 200, 3), 120, dtype=np.uint8)
    for tid in range(50):
        clf.add_sample(tid, frame, (10, 10, 40, 80))

    retained = sum(len(c) for c in clf._crops.values())
    assert retained <= 10
    # Colour samples are cheap and still recorded for every track — capping the
    # preview must not cost team assignment any data.
    assert len(clf._samples) == 50
