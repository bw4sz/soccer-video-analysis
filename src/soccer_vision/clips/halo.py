"""Draw a gentle player halo onto a clip window.

Clips are normally pure ffmpeg cuts (see :mod:`soccer_vision.clips.extract`).
When a clip is about *one* player, ``--halo`` re-renders the window frame by
frame and paints a soft team-coloured spotlight on the target track so the
viewer's eye lands on the right player.

The pipeline produces bounding boxes, not segmentation masks, so
``sv.HaloAnnotator`` (which needs ``detections.mask``) is not usable yet. The
box-only equivalent used by soccer broadcasts is a feet **ellipse**
(``sv.EllipseAnnotator``); a ``circle`` variant is offered too. Once SAM3 masks
land (``soccer_vision.tracking.sam3``) a true mask halo can be added here.

Per-frame track boxes come from ``tracks.json`` written by the process
pipeline. Tracking is sampled (≈5 fps), so boxes are interpolated up to the
clip's native frame rate; gaps longer than ``max_gap_frames`` are left
un-haloed rather than guessed.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

_STYLES = ("ellipse", "circle")


def load_track_boxes(tracks_path: str | Path) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Load ``tracks.json`` into ``{track_id: [(frame, bbox), ...]}`` sorted by frame.

    ``bbox`` is a ``float32`` array ``[x1, y1, x2, y2]`` in proxy pixels.
    """
    doc = json.loads(Path(tracks_path).read_text())
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    for tid, samples in doc.get("tracks", {}).items():
        seq = [
            (int(s["frame"]), np.asarray(s["bbox"], dtype=np.float32))
            for s in samples
            if s.get("bbox") is not None
        ]
        seq.sort(key=lambda fb: fb[0])
        out[int(tid)] = seq
    return out


def interpolate_bbox(
    samples: list[tuple[int, np.ndarray]],
    frame: int,
    max_gap_frames: int,
) -> np.ndarray | None:
    """Bbox for ``frame`` by linear interpolation between the two nearest samples.

    Returns ``None`` when ``frame`` is outside the sampled span or the enclosing
    sample gap exceeds ``max_gap_frames`` (the player was lost there, so drawing
    a halo would be a guess).
    """
    if not samples:
        return None
    frames = [f for f, _ in samples]
    i = bisect_left(frames, frame)

    if i < len(frames) and frames[i] == frame:
        return samples[i][1]
    if i == 0 or i >= len(frames):
        return None  # before first / after last sample — don't extrapolate

    f0, b0 = samples[i - 1]
    f1, b1 = samples[i]
    if f1 - f0 > max_gap_frames:
        return None
    t = (frame - f0) / (f1 - f0)
    return (b0 + t * (b1 - b0)).astype(np.float32)


def render_halo_clip(
    video_path: str | Path,
    out_path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    track_samples: list[tuple[int, np.ndarray]],
    color: tuple[int, int, int] = (56, 130, 246),
    style: str = "ellipse",
    thickness: int = 2,
    max_gap_frames: int = 20,
) -> Path:
    """Re-encode ``[start_s, start_s+duration_s]`` with a halo on the target track.

    ``color`` is BGR (OpenCV order). ``track_samples`` is one track's
    ``[(frame, bbox), ...]`` list from :func:`load_track_boxes`; the frames must
    be in the *source video's* numbering. Frames without a confident box are
    written through unannotated.
    """
    import supervision as sv

    if style not in _STYLES:
        raise ValueError(f"style must be one of {_STYLES}, got {style!r}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    palette = sv.ColorPalette(colors=[sv.Color(*(int(c) for c in reversed(color)))])
    if style == "ellipse":
        annotator = sv.EllipseAnnotator(color=palette, thickness=thickness)
    else:
        annotator = sv.CircleAnnotator(color=palette, thickness=thickness)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, int(round(start_s * fps)))
    n_frames = int(round(duration_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        for offset in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            bbox = interpolate_bbox(track_samples, start_frame + offset, max_gap_frames)
            if bbox is not None:
                det = sv.Detections(
                    xyxy=bbox.reshape(1, 4), class_id=np.zeros(1, dtype=int)
                )
                frame = annotator.annotate(frame, det)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    return out_path
