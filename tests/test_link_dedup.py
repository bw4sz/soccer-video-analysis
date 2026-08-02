"""Collapsing lanes that are one player under two track ids.

The detector sometimes mints a second box on a player it is already tracking.
ByteTrack gives it a fresh id and the original lane dies a frame or two later,
so the two lanes *overlap in time* — which is exactly the case ``link_tracks``
refuses, since it only ever joins a lane's end to a later lane's start. On the
30 fps U14G run that is how 9.9% of lanes lasting over 10 s end.
"""

from soccer_vision.tracking.link import merge_duplicate_lanes


def _lane(start, n, x, y=500.0, w=30.0, h=60.0, step=1):
    return [{"frame": start + i * step, "bbox": [x, y, x + w, y + h]}
            for i in range(n)]


def _doc(tracks, teams, fps=30.0):
    return {"fps": fps, "tracks": tracks, "teams": teams}


def test_merges_a_duplicate_id_born_on_top_of_a_dying_lane():
    doc, stats = merge_duplicate_lanes(_doc(
        {"a": _lane(0, 30, 100.0), "b": _lane(28, 30, 101.0)},
        {"a": "black", "b": "black"},
    ))
    assert stats["merges"] == 1
    assert len(doc["tracks"]) == 1
    (samples,) = doc["tracks"].values()
    assert [s["frame"] for s in samples] == list(range(58))


def test_a_merged_lane_can_still_be_found_by_its_old_id():
    """Callers hold lane ids from before the merge — a seed, a jerseys.json key."""
    doc, stats = merge_duplicate_lanes(_doc(
        {"a": _lane(0, 30, 100.0), "b": _lane(28, 30, 101.0)},
        {"a": "black", "b": "black"},
    ))
    assert stats["alias"]["b"] in doc["tracks"]
    assert stats["alias"]["a"] == stats["alias"]["b"]


def test_never_merges_across_kits():
    """Geometry cannot tell one player under two ids from two players in contact.

    Without this gate 16.3% of these merges joined lanes the team classifier had
    placed in different kits, one of them stitching a 62 s white-kit lane onto a
    hand-verified black one.
    """
    doc, stats = merge_duplicate_lanes(_doc(
        {"a": _lane(0, 30, 100.0), "b": _lane(28, 30, 101.0)},
        {"a": "black", "b": "white"},
    ))
    assert stats["merges"] == 0
    assert len(doc["tracks"]) == 2


def test_leaves_lanes_with_no_kit_alone():
    doc, stats = merge_duplicate_lanes(_doc(
        {"a": _lane(0, 30, 100.0), "b": _lane(28, 30, 101.0)}, {},
    ))
    assert stats["merges"] == 0


def test_ignores_a_neighbour_standing_a_box_away():
    """Two players side by side are not one player, however well timed."""
    doc, stats = merge_duplicate_lanes(_doc(
        {"a": _lane(0, 30, 100.0), "b": _lane(28, 30, 200.0)},
        {"a": "black", "b": "black"},
    ))
    assert stats["merges"] == 0


def test_matches_the_same_player_through_a_box_scale_change():
    """IoU is the wrong test: the real handoff pairs a 30x59 box with a 51x83 one.

    Both boxes sit on one player; they disagree about how much of her the
    detector found. An IoU gate loose enough to accept that is loose enough to
    accept a neighbour.
    """
    doc, stats = merge_duplicate_lanes(_doc(
        {"a": _lane(0, 30, 224.0, y=501.0, w=30.0, h=59.0),
         "b": _lane(29, 30, 212.0, y=502.0, w=51.0, h=83.0)},
        {"a": "black", "b": "black"},
    ))
    assert stats["merges"] == 1


def test_does_not_merge_lanes_that_merely_run_alongside_each_other():
    """A merge is a *handoff*: the predecessor must be ending, not mid-lane."""
    doc, stats = merge_duplicate_lanes(_doc(
        {"a": _lane(0, 200, 100.0), "b": _lane(5, 200, 101.0)},
        {"a": "black", "b": "black"},
    ))
    assert stats["merges"] == 0


def test_reports_lane_counts():
    _, stats = merge_duplicate_lanes(_doc(
        {"a": _lane(0, 30, 100.0), "b": _lane(28, 30, 101.0),
         "c": _lane(0, 30, 900.0)},
        {"a": "black", "b": "black", "c": "white"},
    ))
    assert (stats["lanes_before"], stats["lanes_after"]) == (3, 2)
