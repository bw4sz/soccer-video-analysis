"""Offline tests for Label Studio task building + fine-tune export."""

from __future__ import annotations

import json
from xml.etree import ElementTree

from soccer_vision.annotate import label_studio as ls
from soccer_vision.clips.extract import pair_events_with_clips, parse_clip_name
from soccer_vision.events.labels import EVENT_LABELS


def _make_run(tmp_path):
    """A minimal processed run: OSL events + matching clip files."""
    run = tmp_path / "runs" / "match01"
    (run / "clips").mkdir(parents=True)
    events = [
        {"label": "goal_kick", "position_ms": 10000, "frame": 300, "confidence": 0.6},
        {"label": "corner_kick", "position_ms": 25000, "frame": 750, "confidence": 0.5},
    ]
    (run / "annotations.json").write_text(
        json.dumps({"format": "osl-json", "version": "2.0", "events": events})
    )
    # Names follow extract_event_clips: clip_{i:03d}_{label}_{ts}s.mp4
    (run / "clips" / "clip_001_goal_kick_10s.mp4").write_bytes(b"x")
    (run / "clips" / "clip_002_corner_kick_25s.mp4").write_bytes(b"x")
    return run, events


def test_parse_clip_name_roundtrip():
    p = parse_clip_name("clip_001_goal_kick_10s.mp4")
    assert p["index"] == 1 and p["label"] == "goal_kick" and p["timestamp_s"] == 10.0
    assert parse_clip_name("not-a-clip.mp4") is None


def test_pair_events_with_clips_aligns_by_index_and_ts(tmp_path):
    run, events = _make_run(tmp_path)
    pairs = pair_events_with_clips(events, run / "clips")
    assert pairs[0][1].name == "clip_001_goal_kick_10s.mp4"
    assert pairs[1][1].name == "clip_002_corner_kick_25s.mp4"


def test_pair_events_with_clips_none_when_missing(tmp_path):
    run, events = _make_run(tmp_path)
    events.append({"label": "shot", "position_ms": 90000, "frame": 2700})
    pairs = pair_events_with_clips(events, run / "clips")
    assert pairs[2][1] is None  # no clip extracted for the shot


def test_labeling_config_lists_every_label():
    xml = ls.labeling_config_xml()
    root = ElementTree.fromstring(xml)  # must be well-formed
    choices = {c.get("value") for c in root.iter("Choice")}
    assert choices == set(EVENT_LABELS)


def test_build_tasks_prefills_pipeline_prediction(tmp_path):
    run, _ = _make_run(tmp_path)
    tasks = ls.build_tasks(run, serve_root=run.parent)
    assert len(tasks) == 2
    t0 = tasks[0]
    expected_url = ls.LOCAL_FILES_PREFIX + "match01/clips/clip_001_goal_kick_10s.mp4"
    assert t0["data"]["video"] == expected_url
    assert t0["data"]["pipeline_label"] == "goal_kick"
    # Prediction pre-fills the annotator's choice with the pipeline label.
    preds = t0["predictions"]
    assert preds[0]["model_version"] == ls.PIPELINE_MODEL_VERSION
    assert preds[0]["result"][0]["value"]["choices"] == ["goal_kick"]


def test_build_tasks_includes_soccerchat_when_present(tmp_path):
    run, _ = _make_run(tmp_path)
    (run / "soccerchat.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "frame": 300,
                        "sc_label": "kickoff",
                        "sc_class": "Kick-off",
                        "caption": "A goalkeeper restarts play.",
                        "verdict": "PLAUSIBLE",
                        "confidence": 0.9,
                    }
                ]
            }
        )
    )
    tasks = ls.build_tasks(run, serve_root=run.parent)
    t0 = next(t for t in tasks if t["data"]["frame"] == 300)
    assert t0["data"]["sc_caption"] == "A goalkeeper restarts play."
    assert t0["data"]["verdict"] == "PLAUSIBLE"
    # SoccerChat contributes a second prediction.
    versions = {p["model_version"] for p in t0["predictions"]}
    assert ls.SOCCERCHAT_MODEL_VERSION in versions


def test_write_project_files(tmp_path):
    run, _ = _make_run(tmp_path)
    paths = ls.write_project_files(run, serve_root=run.parent)
    assert paths["config"].exists() and paths["tasks"].exists()
    assert paths["_n_tasks"] == 2


def test_export_finetune_uses_corrected_label(tmp_path):
    export = [
        {
            "data": {"clip": "clip_001_goal_kick_10s.mp4", "video": "..."},
            "annotations": [
                {
                    "result": [
                        {
                            "from_name": "label",
                            "to_name": "video",
                            "type": "choices",
                            "value": {"choices": ["kickoff"]},  # annotator corrected it
                        }
                    ]
                }
            ],
        },
        {"data": {"clip": "clip_002_corner_kick_25s.mp4"}, "annotations": []},  # skipped
    ]
    records = ls.export_finetune(export, clips_root="/data/clips")
    assert len(records) == 1
    rec = records[0]
    assert rec["response"] == "kickoff"
    assert rec["videos"] == ["/data/clips/clip_001_goal_kick_10s.mp4"]
    assert "<video>" in rec["query"]
