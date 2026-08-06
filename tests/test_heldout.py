"""Tests for the held-out registry: the spans nothing may learn from.

The failure this guards against is silent by nature — a gallery banking a crop
from the footage we later score on doesn't crash, it just returns a flattering
number. So these pin the behaviours that make the gate *noisy* instead: an
unregistered run reports itself as unprotected rather than clean, `only` mode is
the exact complement of `exclude`, and a gallery merged with an unstamped one
loses its provenance rather than keeping a half-truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_vision import heldout
from soccer_vision.identify.gallery import (
    build_gallery,
    load_gallery,
    merge_galleries,
    parse_source,
    save_gallery,
    source_key,
)

REGISTRY = """
version: 1
blocks:
  - id: block-a
    flavour: within-video
    video: match.mp4
    runs: [run-a, run-a-30fps]
    fps: 30.0
    start_s: 100
    end_s: 200
    gold:
      - id: gold-a
        start_s: 120
        end_s: 180
        project: runs/run-a/heldout_gold
"""


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "heldout.yaml"
    path.write_text(REGISTRY)
    return heldout.load_registry(path)


def test_a_block_matches_its_video_and_every_run_derived_from_it(registry):
    """Runs rename the footage to broadcast_proxy.mp4, so the run name is the key."""
    span = registry.spans_for("run-a-30fps")
    assert len(span) == 1
    assert registry.spans_for("data/match.mp4")[0].block_id == "block-a"
    assert registry.spans_for("runs/run-a")[0].block_id == "block-a"


def test_seconds_become_frames_at_the_blocks_declared_rate(registry):
    span = registry.spans_for("run-a")[0]

    assert (span.first_frame, span.last_frame) == (3000, 6000)
    assert span.contains(3000) and span.contains(6000)
    assert not span.contains(2999)


def test_an_unregistered_run_is_reported_as_unprotected_not_as_safe(registry):
    """The dangerous direction. Silence here is how footage leaks in unnoticed."""
    assert registry.spans_for("some-other-run") == []

    message = registry.describe_coverage("some-other-run")
    assert "nothing declared" in message
    assert "unprotected, not safe" in message


def test_only_mode_is_the_exact_complement_of_exclude(registry):
    """The gold set is built from what enrolment refuses — they must not overlap."""
    frames = [0, 2999, 3000, 4500, 6000, 6001, 9999]
    items = [(f, "box") for f in frames]

    kept_out, _ = heldout.filter_frames(items, heldout.EXCLUDE, key="run-a",
                                        registry=registry)
    kept_in, _ = heldout.filter_frames(items, heldout.ONLY, key="run-a",
                                       registry=registry)

    assert [f for f, _ in kept_in] == [3000, 4500, 6000]
    assert sorted(f for f, _ in kept_out + kept_in) == frames
    assert not set(kept_out) & set(kept_in)


def test_off_mode_filters_nothing(registry):
    items = [(4500, "box")]

    kept, dropped = heldout.filter_frames(items, heldout.OFF, key="run-a",
                                          registry=registry)

    assert kept == items and dropped == []


def test_no_registry_at_all_filters_nothing(tmp_path):
    empty = heldout.load_registry(tmp_path / "does-not-exist.yaml")
    items = [(4500, "box")]

    kept, dropped = heldout.filter_frames(items, heldout.EXCLUDE, key="run-a",
                                          registry=empty)

    assert kept == items and dropped == []


def test_frame_of_lets_the_gate_read_any_shaped_item(registry):
    """Track samples, named boxes and whole windows all carry a frame differently."""
    windows = [{"mid": 4500}, {"mid": 10}]

    kept, dropped = heldout.filter_frames(windows, heldout.EXCLUDE, key="run-a",
                                          registry=registry,
                                          frame_of=lambda w: w["mid"])

    assert kept == [{"mid": 10}] and dropped == [{"mid": 4500}]


def test_gold_sets_are_findable_by_id(registry):
    block, gold = registry.gold("gold-a")

    assert block.id == "block-a"
    assert (gold.start_s, gold.end_s) == (120, 180)


# -- provenance ----------------------------------------------------------


def test_provenance_survives_the_per_player_cap(tmp_path):
    """The cap subsamples rows; the stamps must follow the rows they belong to."""
    # One distinguishable row per exemplar, so each survivor can be traced back.
    emb = np.eye(10, dtype=np.float32)
    names = ["Ana"] * 10
    source = [source_key("run-a", 100 + i) for i in range(10)]

    gallery = build_gallery(emb, names, max_per_player=3, source=source)

    assert len(gallery["source"]) == len(gallery["emb"]) == 3
    for row, stamp in zip(gallery["emb"], gallery["source"]):
        original = int(np.argmax(np.abs(emb @ row)))
        assert stamp == source[original]


def test_provenance_round_trips_through_the_npz(tmp_path):
    emb = np.eye(3, 4, dtype=np.float32)
    gallery = build_gallery(emb, ["Ana", "Bo", "Ana"],
                            source=[source_key("run-a", f) for f in (1, 2, 3)])
    path = tmp_path / "g.npz"

    save_gallery(gallery, path)
    loaded = load_gallery(path)

    assert loaded["source"] == gallery["source"]
    assert parse_source(loaded["source"][0])[0] == "run-a"


def test_merging_with_an_unstamped_gallery_drops_provenance_entirely(tmp_path):
    """Half-stamped would audit clean on the half it recorded and lie about the rest."""
    emb = np.eye(2, 4, dtype=np.float32)
    stamped = build_gallery(emb, ["Ana", "Bo"],
                            source=[source_key("run-a", 1), source_key("run-a", 2)])
    bare = build_gallery(emb, ["Ana", "Bo"])

    merged = merge_galleries(stamped, bare)

    assert not merged.get("source")


def test_a_run_level_stamp_parses_but_names_no_frame():
    """The OCR bootstrap route knows the footage, never which frames survived."""
    assert parse_source(source_key("run-a", -1)) == ("run-a", -1)
    assert parse_source("no-at-sign-here") is None
