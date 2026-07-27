"""Action-detector abstraction tests."""

import pytest

from soccer_vision.events.sources import (
    ActionContext,
    LearnedActionDetector,
    active_detectors,
    run_detectors,
)


def test_learned_engine_unavailable_and_empty():
    det = LearnedActionDetector()
    assert det.is_available() is False
    assert det.detect(ActionContext(fps=5.0)) == []


def test_active_detectors_drops_unavailable():
    names = {d.name for d in active_detectors()}
    assert "learned" not in names  # no checkpoint yet
    assert "vlm" not in names  # opt-in only


def test_run_detectors_merges_and_sorts():
    ctx = ActionContext(fps=5.0)
    events = run_detectors(active_detectors(), ctx)
    times = [e.get("timestamp_s", 0) for e in events]
    assert times == sorted(times)


@pytest.mark.parametrize("name", ["rules", "set_piece"])
def test_retired_rules_engine_raises_rather_than_silently_detecting_nothing(name):
    """Asking for the retired set-piece engine must be loud.

    Returning an empty list is exactly how that engine hid its own failure, so a
    stale config gets a ValueError explaining where set-piece detection went.
    """
    with pytest.raises(ValueError, match="no longer available"):
        active_detectors({"action_engines": [name]})


def test_set_piece_module_is_gone():
    """The metric set-piece heuristics were deleted, not merely unwired."""
    with pytest.raises(ImportError):
        import soccer_vision.events.set_piece  # noqa: F401


def test_legacy_source_names_still_import():
    """Back-compat aliases keep the pre-rename import paths working."""
    from soccer_vision.events.sources import DetectionContext, active_sources

    assert DetectionContext is ActionContext
    # legacy config key 'sources' still selects engines
    assert active_sources({"sources": ["learned"]}) == []
