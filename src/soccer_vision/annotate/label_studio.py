"""Label Studio integration for reviewing pipeline events on youth clips.

Turns a processed run directory into a Label Studio *video classification*
project where each task is one extracted event clip, **pre-filled** with:

* the pipeline's detected label (as the Label Studio *prediction* the annotator
  confirms or overrides), and
* SoccerChat's caption + predicted class, when ``soccer-vision describe`` has
  been run (shown as read-only context, plus a second prediction).

The annotator's corrected choices export straight back to
:func:`export_finetune` as SoccerChat/ms-swift training records — closing the
loop from "pro-trained model guesses" to "youth ground truth".

This module is pure JSON/string manipulation (no cv2/torch), so it is fully
covered by offline tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from soccer_vision.clips.extract import pair_events_with_clips
from soccer_vision.events.labels import EVENT_LABELS, LABEL_DESCRIPTIONS
from soccer_vision.io.osl import read_osl

# Label Studio's built-in local file server exposes files under this route,
# resolved relative to LOCAL_FILES_DOCUMENT_ROOT. See label_studio/README.md.
LOCAL_FILES_PREFIX = "/data/local-files/?d="

PIPELINE_MODEL_VERSION = "soccer-vision-pipeline"
SOCCERCHAT_MODEL_VERSION = "soccerchat-qwen2vl"


def labeling_config_xml(labels: list[str] | None = None) -> str:
    """Return the Label Studio labeling-config XML for clip classification.

    Generated from :data:`EVENT_LABELS` so the annotation UI never drifts from
    the taxonomy the pipeline emits.
    """
    labels = labels or EVENT_LABELS
    choices = "\n".join(
        f'    <Choice value="{escape(lbl)}" '
        f'hint="{escape(LABEL_DESCRIPTIONS.get(lbl, ""))}"/>'
        for lbl in labels
    )
    return (
        '<View>\n'
        '  <Style>.sv-meta { font-size: 14px; color: #444; }</Style>\n'
        '  <Header value="Confirm or correct the event label for this clip"/>\n'
        '  <Video name="video" value="$video"/>\n'
        '  <Text name="pipeline" value="Pipeline: $pipeline_label   |   '
        'SoccerChat: $sc_label ($verdict)" className="sv-meta"/>\n'
        '  <Text name="caption" value="SoccerChat: $sc_caption" className="sv-meta"/>\n'
        '  <Text name="ts" value="t=$timestamp_s s   frame=$frame" className="sv-meta"/>\n'
        '  <Choices name="label" toName="video" choice="single" showInLine="false">\n'
        f'{choices}\n'
        '  </Choices>\n'
        '  <TextArea name="notes" toName="video" editable="true" perRegion="false"\n'
        '            placeholder="Optional notes / why you changed it" rows="2"/>\n'
        '</View>\n'
    )


def local_files_url(clip_path: str | Path, serve_root: str | Path) -> str:
    """Build the ``/data/local-files/?d=...`` URL for a clip.

    ``serve_root`` must equal Label Studio's ``LOCAL_FILES_DOCUMENT_ROOT``; the
    ``d=`` value is the clip path relative to it.
    """
    rel = Path(clip_path).resolve().relative_to(Path(serve_root).resolve())
    return f"{LOCAL_FILES_PREFIX}{rel.as_posix()}"


def _prediction(model_version: str, label: str | None, score: float | None) -> dict | None:
    if not label:
        return None
    pred: dict[str, Any] = {
        "model_version": model_version,
        "result": [
            {
                "from_name": "label",
                "to_name": "video",
                "type": "choices",
                "value": {"choices": [label]},
            }
        ],
    }
    if score is not None:
        pred["score"] = score
    return pred


def _load_soccerchat_by_frame(run_dir: Path) -> dict[Any, dict]:
    """Index ``soccerchat.json`` results by frame (if present)."""
    sc_path = run_dir / "soccerchat.json"
    if not sc_path.exists():
        return {}
    data = json.loads(sc_path.read_text())
    return {r.get("frame"): r for r in data.get("results", [])}


def build_tasks(
    run_dir: str | Path,
    *,
    serve_root: str | Path | None = None,
) -> list[dict]:
    """Build Label Studio tasks (with predictions) from a processed run.

    ``serve_root`` defaults to the run's parent (the ``runs/`` base), which is
    what you point ``LOCAL_FILES_DOCUMENT_ROOT`` at to serve every match at once.
    """
    run_dir = Path(run_dir)
    serve_root = Path(serve_root) if serve_root else run_dir.parent

    osl = read_osl(run_dir / "annotations.json")
    events = osl.get("events", [])
    for e in events:
        e.setdefault("timestamp_s", round(e.get("position_ms", 0) / 1000, 2))

    pairs = pair_events_with_clips(events, run_dir / "clips")
    sc_by_frame = _load_soccerchat_by_frame(run_dir)

    tasks: list[dict] = []
    for event, clip_path in pairs:
        if clip_path is None:
            continue  # no clip extracted for this event → nothing to show
        sc = sc_by_frame.get(event.get("frame"), {})
        data = {
            "video": local_files_url(clip_path, serve_root),
            "clip": Path(clip_path).name,
            "pipeline_label": event["label"],
            "timestamp_s": event.get("timestamp_s"),
            "frame": event.get("frame"),
            "sc_label": sc.get("sc_label") or "—",
            "sc_caption": sc.get("caption") or "(run `soccer-vision describe` for a caption)",
            "verdict": sc.get("verdict") or "—",
        }
        predictions = [
            p
            for p in (
                _prediction(PIPELINE_MODEL_VERSION, event["label"], event.get("confidence")),
                _prediction(SOCCERCHAT_MODEL_VERSION, sc.get("sc_label"), sc.get("confidence")),
            )
            if p is not None
        ]
        tasks.append({"data": data, "predictions": predictions})

    return tasks


# --- Fine-tune export ------------------------------------------------------

# The instruction the exported records pair with each clip. Kept close to the
# SoccerChat classification prompt so the fine-tune data matches inference use.
FINETUNE_QUERY = (
    "<video>Classify the main event in this youth soccer clip into one of: "
    + ", ".join(EVENT_LABELS)
    + "."
)


def _annotation_label(task: dict) -> str | None:
    """Extract the human-chosen label from an exported Label Studio task."""
    for ann in task.get("annotations", []):
        if ann.get("was_cancelled"):
            continue
        for res in ann.get("result", []):
            if res.get("from_name") == "label":
                choices = res.get("value", {}).get("choices", [])
                if choices:
                    return choices[0]
    return None


def export_finetune(
    ls_export: list[dict] | str | Path,
    *,
    clips_root: str | Path | None = None,
) -> list[dict]:
    """Convert a Label Studio JSON export into ms-swift fine-tune records.

    Each returned record is ``{"query", "response", "videos": [clip_path]}`` —
    the schema SoccerChat's training notebooks consume — using the annotator's
    corrected label as the target ``response``. ``clips_root`` resolves the
    stored clip filename to an absolute path for training; when omitted the
    task's ``video`` URL/filename is used as-is.
    """
    if isinstance(ls_export, (str, Path)):
        ls_export = json.loads(Path(ls_export).read_text())

    records: list[dict] = []
    for task in ls_export:
        label = _annotation_label(task)
        if not label:
            continue  # unannotated / skipped task
        data = task.get("data", {})
        clip_name = data.get("clip") or data.get("video", "")
        clip = str(Path(clips_root) / clip_name) if clips_root else clip_name
        records.append(
            {"query": FINETUNE_QUERY, "response": label, "videos": [clip]}
        )
    return records


def write_project_files(
    run_dir: str | Path,
    *,
    serve_root: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write ``labeling_config.xml`` + ``label_studio_tasks.json`` for a run.

    Returns the two paths. This is what the ``annotate`` CLI calls for the
    offline (no running server) path.
    """
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir
    out.mkdir(parents=True, exist_ok=True)

    config_path = out / "labeling_config.xml"
    tasks_path = out / "label_studio_tasks.json"

    config_path.write_text(labeling_config_xml())
    tasks = build_tasks(run_dir, serve_root=serve_root)
    tasks_path.write_text(json.dumps(tasks, indent=2))
    return {"config": config_path, "tasks": tasks_path, "_n_tasks": len(tasks)}
