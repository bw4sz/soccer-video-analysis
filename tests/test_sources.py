"""Action-detector abstraction tests."""

from soccer_vision.events.set_piece import detect_all_set_pieces
from soccer_vision.events.sources import (
    ActionContext,
    LearnedActionDetector,
    RulesActionDetector,
    active_detectors,
    run_detectors,
)


def _ball_track():
    """A stationary ball in the left goal box → a goal kick."""
    return [
        {"frame": i, "timestamp_s": i * 0.2, "pixel_x": 100.0, "pixel_y": 100.0,
         "field_x": 2.0, "field_y": 18.0, "confidence": 0.9}
        for i in range(6)
    ]


def test_rules_engine_matches_heuristic():
    ball = _ball_track()
    ctx = ActionContext(fps=5.0, ball_positions=ball)
    det = RulesActionDetector()
    assert det.is_available()

    from_engine = det.detect(ctx)
    direct = detect_all_set_pieces(ball)
    assert [e["label"] for e in from_engine] == [e["label"] for e in direct]
    assert from_engine and from_engine[0]["source"] == "rules"


def test_learned_engine_unavailable_and_empty():
    det = LearnedActionDetector()
    assert det.is_available() is False
    assert det.detect(ActionContext(fps=5.0)) == []


def test_active_detectors_drops_unavailable():
    detectors = active_detectors()
    names = {d.name for d in detectors}
    assert "rules" in names
    assert "learned" not in names  # no checkpoint yet


def test_run_detectors_merges_and_sorts():
    ctx = ActionContext(fps=5.0, ball_positions=_ball_track())
    events = run_detectors(active_detectors(), ctx)
    times = [e.get("timestamp_s", 0) for e in events]
    assert times == sorted(times)


def test_legacy_source_names_still_import():
    """Back-compat aliases keep the pre-rename import paths working."""
    from soccer_vision.events.sources import (
        DetectionContext,
        SetPieceSource,
        active_sources,
    )

    assert SetPieceSource is RulesActionDetector
    assert DetectionContext is ActionContext
    # legacy config key 'sources' still selects engines
    names = {d.name for d in active_sources({"sources": ["set_piece"]})}
    assert names == {"rules"}
