"""Read a Label Studio **video object tracking** export back into source frames.

The tracklet export answers "who is this ringed lane"; this one answers "where
was this player", with no lane involved. It is the only annotation in the repo
that owes the tracker nothing, which is what lets it measure a *miss* — a player
we never detected has no lane to be named, but she does have a hand-drawn box.

Two properties of the format cost a run each to discover, so they are handled
here rather than at every call site:

- **Frames are 1-indexed** in Label Studio and 0-indexed in our tracks, and the
  clip starts at the window's ``start_frame`` in the source video. Both offsets
  are applied here, so callers get source frame numbers.
- **The sequence holds keyframes, not frames.** The annotator drags the box back
  onto the player every second or so and Label Studio draws the between-frames
  by linear interpolation. A consumer that reads only the keyframes sees a
  player who teleports once a second, so the interpolation is redone here.
  ``enabled: false`` closes a span — the player left the shot — and no box is
  emitted from there until the next keyframe opens one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNSURE = "unsure"


def _sequence_spans(sequence: list[dict]) -> list[list[dict]]:
    """Split a keyframe sequence into runs the region is actually visible for.

    A single region can leave the shot and come back; Label Studio marks the
    departure by ``enabled: false`` on the last keyframe of the run. Interpolating
    across that gap would draw a box gliding through the place the player was not.
    """
    spans: list[list[dict]] = []
    current: list[dict] = []
    for kf in sorted(sequence, key=lambda k: k.get("frame", 0)):
        current.append(kf)
        if not kf.get("enabled", True):
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return [s for s in spans if s]


def _interpolate(span: list[dict], frame_w: float, frame_h: float
                 ) -> list[tuple[int, np.ndarray]]:
    """Every clip frame covered by one visible run, boxes in pixels.

    Percentages are converted here rather than at the end because a box is
    interpolated in the space it was drawn in, and the two only agree while the
    frame size is constant (it is, but the conversion is cheap and the assumption
    is not worth carrying).
    """
    def px(kf):
        return np.array([
            kf["x"] / 100.0 * frame_w,
            kf["y"] / 100.0 * frame_h,
            (kf["x"] + kf["width"]) / 100.0 * frame_w,
            (kf["y"] + kf["height"]) / 100.0 * frame_h,
        ], dtype=float)

    out: list[tuple[int, np.ndarray]] = []
    if len(span) == 1:
        return [(int(span[0]["frame"]), px(span[0]))]
    for a, b in zip(span, span[1:]):
        fa, fb = int(a["frame"]), int(b["frame"])
        box_a, box_b = px(a), px(b)
        if fb <= fa:
            out.append((fa, box_a))
            continue
        for f in range(fa, fb):
            t = (f - fa) / (fb - fa)
            out.append((f, box_a + t * (box_b - box_a)))
    last = span[-1]
    out.append((int(last["frame"]), px(last)))
    return out


def boxes_from_export(
    export: list[dict],
    manifest: dict,
    *,
    frame_w: float,
    frame_h: float,
    keep_unsure: bool = False,
) -> tuple[list[tuple[int, np.ndarray, str]], dict]:
    """``[(source_frame, bbox_px, player_name), ...]`` from a video-tracking export.

    ``manifest`` is the ``hand_tracking.json`` written at staging; it carries the
    ``start_frame`` each clip was cut at. Tasks also carry their own
    ``start_frame``, and that is preferred — an export can outlive a re-render,
    and a box placed against the wrong window is worse than a box dropped.

    ``unsure`` is excluded by default. It marks one of ours the annotator could
    not name, which is a real observation for *detection* recall and a
    contradiction for *identity*, so the caller has to ask for it.
    """
    windows = {w["window"]: w for w in manifest.get("windows", [])}
    out: list[tuple[int, np.ndarray, str]] = []
    summary = {"regions": 0, "unsure_skipped": 0, "unmatched_tasks": 0,
               "tasks": 0, "players": set()}

    for task in export:
        data = task.get("data", {}) or {}
        start_frame = data.get("start_frame")
        if start_frame is None:
            window = windows.get(data.get("window"))
            if window is None:
                summary["unmatched_tasks"] += 1
                continue
            start_frame = window["start_frame"]
        start_frame = int(start_frame)
        summary["tasks"] += 1

        for ann in task.get("annotations", []) or []:
            for res in ann.get("result", []) or []:
                if res.get("type") != "videorectangle":
                    continue
                value = res.get("value", {}) or {}
                labels = value.get("labels") or []
                sequence = value.get("sequence") or []
                if not labels or not sequence:
                    continue
                name = str(labels[0]).strip()
                if name.lower() == UNSURE and not keep_unsure:
                    summary["unsure_skipped"] += 1
                    continue
                summary["regions"] += 1
                summary["players"].add(name)
                for span in _sequence_spans(sequence):
                    for clip_frame, bbox in _interpolate(span, frame_w, frame_h):
                        # Label Studio counts frames from 1; our tracks count
                        # from 0, and clip frame 1 *is* the window's start frame.
                        out.append((start_frame + clip_frame - 1, bbox, name))

    summary["players"] = sorted(summary["players"])
    out.sort(key=lambda r: (r[0], r[2]))
    return out, summary


def load_manifest(project_dir: str | Path) -> dict:
    """Read ``hand_tracking.json`` from a staged project directory."""
    import json

    return json.loads((Path(project_dir) / "hand_tracking.json").read_text())
