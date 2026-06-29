"""OSL JSON 2.0 read/write for soccer-vision pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def new_osl_document(
    match_id: str,
    *,
    video_path: str | None = None,
    fps: float | None = None,
    field_dimensions: dict[str, float] | None = None,
) -> dict[str, Any]:
    doc = {
        "format": "osl-json",
        "version": "2.0",
        "match_id": match_id,
        "metadata": {},
        "events": [],
        "tracking": [],
    }
    if video_path:
        doc["metadata"]["video_path"] = video_path
    if fps:
        doc["metadata"]["fps"] = fps
    if field_dimensions:
        doc["metadata"]["field_dimensions"] = field_dimensions
    return doc


def add_event(
    doc: dict[str, Any],
    *,
    label: str,
    position_ms: int,
    frame: int | None = None,
    confidence: float | None = None,
    source: str = "heuristic",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "label": label,
        "position_ms": position_ms,
        "source": source,
    }
    if frame is not None:
        event["frame"] = frame
    if confidence is not None:
        event["confidence"] = round(confidence, 4)
    if extra:
        event.update(extra)
    doc["events"].append(event)
    return event


def write_osl(doc: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def read_osl(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)
