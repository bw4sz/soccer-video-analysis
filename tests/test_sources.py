"""Event-source abstraction tests."""

from soccer_vision.events.set_piece import detect_all_set_pieces
from soccer_vision.events.sources import (
    DetectionContext,
    SetPieceSource,
    TackleSource,
    active_sources,
    run_sources,
)


def _ball_track():
    """A stationary ball in the left goal box → a goal kick."""
    return [
        {"frame": i, "timestamp_s": i * 0.2, "pixel_x": 100.0, "pixel_y": 100.0,
         "field_x": 2.0, "field_y": 18.0, "confidence": 0.9}
        for i in range(6)
    ]


def test_setpiece_source_matches_heuristic():
    ball = _ball_track()
    ctx = DetectionContext(fps=5.0, ball_positions=ball)
    src = SetPieceSource()
    assert src.is_available()

    from_source = src.detect(ctx)
    direct = detect_all_set_pieces(ball)
    assert [e["label"] for e in from_source] == [e["label"] for e in direct]
    assert from_source and from_source[0]["source"] == "set_piece"


def test_tackle_source_unavailable_and_empty():
    src = TackleSource()
    assert src.is_available() is False
    assert src.detect(DetectionContext(fps=5.0)) == []


def test_active_sources_drops_unavailable():
    sources = active_sources()
    names = {s.name for s in sources}
    assert "set_piece" in names
    assert "tackle" not in names  # no checkpoint yet


def test_run_sources_merges_and_sorts():
    ctx = DetectionContext(fps=5.0, ball_positions=_ball_track())
    events = run_sources(active_sources(), ctx)
    times = [e.get("timestamp_s", 0) for e in events]
    assert times == sorted(times)
