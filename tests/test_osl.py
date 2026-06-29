"""OSL JSON round-trip read/write tests."""

import tempfile
from pathlib import Path

from soccer_vision.io.osl import add_event, new_osl_document, read_osl, write_osl


def test_new_document_structure():
    doc = new_osl_document("match_001", fps=30.0)
    assert doc["format"] == "osl-json"
    assert doc["version"] == "2.0"
    assert doc["match_id"] == "match_001"
    assert doc["events"] == []
    assert doc["metadata"]["fps"] == 30.0


def test_add_event():
    doc = new_osl_document("match_001")
    event = add_event(doc, label="goal_kick", position_ms=45000, frame=1350, confidence=0.85)
    assert len(doc["events"]) == 1
    assert event["label"] == "goal_kick"
    assert event["position_ms"] == 45000
    assert event["confidence"] == 0.85


def test_round_trip():
    doc = new_osl_document("match_002", video_path="test.mp4", fps=25.0,
                           field_dimensions={"width": 55.0, "height": 36.0})
    add_event(doc, label="corner_kick", position_ms=120000)
    add_event(doc, label="throw_in", position_ms=180000, confidence=0.7)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        write_osl(doc, path)
        loaded = read_osl(path)

    assert loaded["match_id"] == "match_002"
    assert len(loaded["events"]) == 2
    assert loaded["events"][0]["label"] == "corner_kick"
    assert loaded["events"][1]["confidence"] == 0.7
    assert loaded["metadata"]["field_dimensions"]["width"] == 55.0
