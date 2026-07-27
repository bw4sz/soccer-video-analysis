"""Gallery matching maths, on synthetic embeddings (no model, no video).

Mirrors ``test_jersey_vote.py``: the interesting behaviour is when the gallery
*abstains*, so most of these assert that an unenrolled or ambiguous track comes
back ``None`` and falls through to OCR rather than getting a confident wrong name.
"""

import numpy as np
import pytest

from soccer_vision.identify.enroll import (
    boxes_from_label_studio,
    crops_from_directory,
    names_from_jerseys,
)
from soccer_vision.identify.gallery import (
    build_gallery,
    load_gallery,
    match_track,
    merge_galleries,
    save_gallery,
)


def player_embeddings(axis: int, n: int = 8, jitter: float = 0.05, dim: int = 16, seed: int = 0):
    """``n`` noisy embeddings clustered on one axis — a stand-in for one player."""
    rng = np.random.default_rng(seed)
    base = np.zeros(dim, dtype=np.float32)
    base[axis] = 1.0
    return base + jitter * rng.standard_normal((n, dim)).astype(np.float32)


@pytest.fixture
def gallery():
    emb = np.concatenate([player_embeddings(0, seed=1), player_embeddings(1, seed=2)])
    names = ["Simon"] * 8 + ["Ada"] * 8
    return build_gallery(emb, names)


def test_matches_the_enrolled_player(gallery):
    m = match_track(player_embeddings(0, n=5, seed=9), gallery)
    assert m.name == "Simon"
    assert m.similarity > 0.9
    assert m.n_crops == 5


def test_abstains_on_a_player_who_was_never_enrolled(gallery):
    # An opponent or referee: far from every enrolled cluster, so no name.
    assert match_track(player_embeddings(7, n=5, seed=3), gallery).name is None


def test_abstains_when_two_players_score_alike(gallery):
    # Halfway between both clusters — similar to each, decisive for neither.
    between = np.zeros((4, 16), dtype=np.float32)
    between[:, 0] = between[:, 1] = 0.7
    m = match_track(between, gallery)
    assert m.name is None
    assert m.margin < 0.05


def test_empty_track_and_empty_gallery_abstain(gallery):
    assert match_track(np.zeros((0, 16)), gallery).name is None
    empty = build_gallery(np.zeros((0, 16)), [])
    assert match_track(player_embeddings(0, n=3), empty).name is None


def test_exemplars_are_capped_per_player():
    g = build_gallery(player_embeddings(0, n=50), ["Simon"] * 50, max_per_player=10)
    assert len(g["emb"]) == 10


def test_save_load_roundtrip_and_merge_across_matches(gallery, tmp_path):
    path = tmp_path / "gallery.npz"
    save_gallery(gallery, path)
    assert match_track(player_embeddings(0, n=3, seed=4), load_gallery(path)).name == "Simon"

    # Next match adds a third player without re-embedding the first two.
    second = build_gallery(player_embeddings(2, n=8, seed=5), ["Kit"] * 8)
    merged = merge_galleries(gallery, second)
    assert merged["names"] == ["Ada", "Kit", "Simon"]
    assert match_track(player_embeddings(2, n=3, seed=6), merged).name == "Kit"


def test_enrols_only_confident_ocr_tracks_and_honours_exclusions():
    doc = {"tracks": {
        "1": {"jersey": 6, "confidence": 0.95, "n_obs": 12},
        "2": {"jersey": 9, "confidence": 0.55, "n_obs": 12},   # unsure vote
        "3": {"jersey": 9, "confidence": 0.99, "n_obs": 2},    # too few reads
        "4": {"jersey": 1, "confidence": 0.99, "n_obs": 20},   # hallucination class
        "5": {"jersey": None, "confidence": 0.0, "n_obs": 0},
    }}
    profile = {"roster": [{"name": "Simon Weinstein", "jersey": 6}]}
    got = names_from_jerseys(doc, profile, exclude={1})
    assert got == {1: "Simon Weinstein"}

    # No roster entry still enrols, under the number.
    assert names_from_jerseys(doc, None, exclude={1}) == {1: "#6"}


def test_crop_folders_enrol_only_once_renamed(tmp_path):
    for folder, files in [
        ("Simon Weinstein", ["000420.jpg", "000900.png"]),
        ("track_0034__ocr20", ["000100.jpg"]),  # dumped but not yet labelled
    ]:
        (tmp_path / folder).mkdir()
        for f in files:
            (tmp_path / folder / f).touch()
    (tmp_path / "Simon Weinstein" / "notes.txt").touch()

    got = crops_from_directory(tmp_path)
    assert [(p.name, n) for p, n in got] == [
        ("000420.jpg", "Simon Weinstein"),
        ("000900.png", "Simon Weinstein"),
    ]


def test_parses_label_studio_boxes_to_pixels():
    export = [{
        "data": {"frame": 1200},
        "annotations": [{"result": [{
            "original_width": 1000, "original_height": 500,
            "value": {"x": 10.0, "y": 20.0, "width": 5.0, "height": 30.0,
                      "rectanglelabels": ["Simon Weinstein"]},
        }]}],
    }]
    (frame, bbox, name), = boxes_from_label_studio(export)
    assert frame == 1200
    assert name == "Simon Weinstein"
    np.testing.assert_allclose(bbox, [100.0, 100.0, 150.0, 250.0])
