"""Offline tests for the SoccerChat verifier (no model / GPU required)."""

from __future__ import annotations

from soccer_vision.events.labels import EVENT_LABELS
from soccer_vision.verify import soccerchat as sc


def test_map_covers_all_classes_and_targets_canonical_labels():
    # Every class SoccerChat can emit maps to a canonical soccer-vision label.
    for cls in sc.SOCCERCHAT_CLASSES:
        assert cls in sc.SOCCERCHAT_TO_LABEL, f"unmapped SoccerChat class: {cls}"
    for cls, label in sc.SOCCERCHAT_TO_LABEL.items():
        assert label in EVENT_LABELS, f"{cls} -> non-canonical {label}"


def test_normalise_class_exact_substring_and_miss():
    assert sc._normalise_class("Corner") == "Corner"
    assert sc._normalise_class("throw-in") == "Throw-in"  # case-insensitive
    assert sc._normalise_class("I think this is a Foul by the defender") == "Foul"
    assert sc._normalise_class("no idea") is None


class FakeModel:
    """Stand-in for SoccerChatModel: returns canned classes per clip path."""

    def __init__(self, by_clip: dict[str, str | None]):
        self.by_clip = by_clip

    def classify(self, clip_path):
        cls = self.by_clip.get(str(clip_path))
        if cls is None:
            return {"sc_class": None, "label": None, "confidence": 0.0, "raw": "???"}
        return {
            "sc_class": cls,
            "label": sc.SOCCERCHAT_TO_LABEL.get(cls),
            "confidence": 0.9,
            "raw": cls,
        }

    def describe(self, clip_path):
        return "a short caption"


def test_verify_clip_confirmed_when_labels_agree():
    model = FakeModel({"a.mp4": "Corner"})
    r = sc.verify_clip(model, "a.mp4", "corner_kick")
    assert r["verdict"] == "CONFIRMED"
    assert r["sc_label"] == "corner_kick"
    assert r["caption"] == "a short caption"


def test_verify_clip_shots_map_to_shot():
    model = FakeModel({"a.mp4": "Shots on target"})
    assert sc.verify_clip(model, "a.mp4", "shot")["verdict"] == "CONFIRMED"


def test_verify_clip_goal_kick_is_plausible_via_proxy_class():
    # SoccerChat has no goal-kick class; a Kick-off/Ball-out reading is PLAUSIBLE.
    for proxy in ("Kick-off", "Ball out of play"):
        r = sc.verify_clip(FakeModel({"a.mp4": proxy}), "a.mp4", "goal_kick")
        assert r["verdict"] == "PLAUSIBLE", proxy


def test_verify_clip_rejected_on_disagreement():
    r = sc.verify_clip(FakeModel({"a.mp4": "Throw-in"}), "a.mp4", "goal_kick")
    assert r["verdict"] == "REJECTED"
    assert r["sc_label"] == "throw_in"


def test_verify_clip_unknown_when_unmappable():
    r = sc.verify_clip(FakeModel({"a.mp4": None}), "a.mp4", "shot")
    assert r["verdict"] == "UNKNOWN"


def test_verify_events_batches_and_skips_missing_clips():
    events = [
        {"label": "corner_kick", "frame": 100, "timestamp_s": 10.0},
        {"label": "goal_kick", "frame": 200, "timestamp_s": 20.0},
        {"label": "shot", "frame": 300, "timestamp_s": 30.0},  # no clip → skipped
    ]
    pairs = [
        (events[0], "a.mp4"),
        (events[1], "b.mp4"),
        (events[2], None),
    ]
    model = FakeModel({"a.mp4": "Corner", "b.mp4": "Foul"})
    out = sc.verify_events(pairs, model=model)
    assert len(out["results"]) == 2  # None clip skipped
    assert [v["frame"] for v in out["verified"]] == [100]
    assert [r["frame"] for r in out["rejected"]] == [200]
    assert out["results"][0]["clip"] == "a.mp4"


def test_is_available_returns_bool():
    assert isinstance(sc.is_available(), bool)
