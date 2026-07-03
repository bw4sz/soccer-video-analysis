"""``trim-empty`` editor: cut dead time out of a match, keep the original.

Workflow::

    ball track (JSON)  ──▶  plan_trim (events.deadball)  ──▶  keep segments
                                                                  │
                                          ffmpeg splice ◀─────────┘
                                                                  │
                                    <name>.trimmed.mp4  +  <name>.trim.json (EDL)

The source video is never modified: a new file is written alongside it. If no
ball track exists yet (the tracker is optional), :func:`build_ball_track` will
generate one from the RF-DETR ball detector when it is installed; otherwise the
caller is told to supply a track and the schema it needs (see
:mod:`soccer_vision.events.deadball`).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from soccer_vision.events.deadball import (
    MIN_DEAD_S,
    PAD_S,
    STATIONARY_PX,
    STATIONARY_WINDOW_S,
    plan_trim,
)
from soccer_vision.io.video import (
    VideoReader,
    ffmpeg_concat,
    ffmpeg_extract_clip,
)


# ---------------------------------------------------------------------------
# Ball-track I/O
# ---------------------------------------------------------------------------

def load_ball_track(path: str | Path) -> dict:
    """Load and lightly validate a ball-track JSON file (schema in deadball)."""
    with open(path) as f:
        track = json.load(f)
    if "samples" not in track:
        raise ValueError(
            f"{path} is not a ball track: missing 'samples'. "
            "See soccer_vision.events.deadball for the expected schema."
        )
    return track


def write_ball_track(track: dict, path: str | Path) -> Path:
    """Persist a ball track to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(track, f, indent=2)
    return path


def track_duration_s(track: dict) -> float:
    """Best available source duration: total_frames/fps, else last sample."""
    total_frames = track.get("total_frames")
    fps = track.get("fps")
    if total_frames and fps:
        return total_frames / fps
    samples = track.get("samples", [])
    return samples[-1]["timestamp_s"] if samples else 0.0


# ---------------------------------------------------------------------------
# Optional ball-track producer (real when the detector is installed)
# ---------------------------------------------------------------------------

def build_ball_track(
    video_path: str | Path,
    *,
    sample_fps: float = 5.0,
    detector=None,
    conf_threshold: float = 0.2,
    device: str | None = None,
) -> dict:
    """Produce a ball track from a video using the RF-DETR ball detector.

    This is the "if we have a ball tracking module" path. The detector is
    imported lazily so the rest of the workflow (and the tests) do not depend
    on the model weights being present. Pass a preloaded ``detector`` to reuse
    one; otherwise it is loaded from pretrained weights on first call.

    Every sampled frame becomes a sample: visible with a position when the ball
    is found, or ``visible: false`` when it is offscreen/undetected.
    """
    from soccer_vision.detection.ball import detect_ball_position

    if detector is None:
        from soccer_vision.detection.rfdetr import RFDETRSoccerDetector
        detector = RFDETRSoccerDetector.from_pretrained(device=device)

    with VideoReader(video_path) as reader:
        native_fps = reader.fps
        interval = max(1, round(native_fps / sample_fps))
        samples: list[dict] = []
        for frame_no, frame in reader.sample_frames(interval):
            ts = frame_no / native_fps
            pos = detect_ball_position(frame, detector, conf_threshold)
            if pos is None:
                samples.append({
                    "frame": frame_no,
                    "timestamp_s": round(ts, 3),
                    "visible": False,
                    "pixel_x": None,
                    "pixel_y": None,
                    "confidence": 0.0,
                })
            else:
                cx, cy, conf = pos
                samples.append({
                    "frame": frame_no,
                    "timestamp_s": round(ts, 3),
                    "visible": True,
                    "pixel_x": round(cx, 1),
                    "pixel_y": round(cy, 1),
                    "confidence": round(conf, 3),
                })
        return {
            "video": str(video_path),
            "fps": native_fps,
            "sample_fps": sample_fps,
            "width": reader.width,
            "height": reader.height,
            "total_frames": reader.total_frames,
            "samples": samples,
        }


# ---------------------------------------------------------------------------
# ffmpeg assembly
# ---------------------------------------------------------------------------

def assemble_segments(
    video_path: str | Path,
    keep_segments: list[dict],
    out_path: str | Path,
    *,
    reencode: bool = True,
) -> Path:
    """Splice ``keep_segments`` of ``video_path`` into a single ``out_path``.

    Each segment is cut to a temp file (re-encoded so the concat demuxer has
    matching streams), then concatenated. The source is read-only.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not keep_segments:
        raise ValueError("Nothing to keep — the whole clip was classified dead.")

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths = []
        for i, seg in enumerate(keep_segments):
            duration = seg["end_s"] - seg["start_s"]
            if duration <= 0:
                continue
            tmp = Path(tmpdir) / f"keep_{i:04d}.mp4"
            ffmpeg_extract_clip(
                video_path, seg["start_s"], duration, tmp, reencode=reencode
            )
            clip_paths.append(tmp)

        if len(clip_paths) == 1:
            import shutil
            shutil.copy2(clip_paths[0], out_path)
        else:
            ffmpeg_concat(clip_paths, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Top-level editor
# ---------------------------------------------------------------------------

def trim_empty(
    video_path: str | Path,
    track: dict,
    *,
    out_path: str | Path | None = None,
    edl_path: str | Path | None = None,
    min_dead_s: float = MIN_DEAD_S,
    stationary_px: float = STATIONARY_PX,
    stationary_window_s: float = STATIONARY_WINDOW_S,
    pad_s: float = PAD_S,
    reencode: bool = True,
    dry_run: bool = False,
) -> dict:
    """Cut dead time out of ``video_path`` guided by ``track``.

    Writes ``<video>.trimmed.mp4`` (or ``out_path``) plus a ``.trim.json``
    edit-decision list, and returns the plan. The original is left untouched.
    ``dry_run`` computes and writes the plan/EDL without invoking ffmpeg.
    """
    video_path = Path(video_path)
    if out_path is None:
        out_path = video_path.with_suffix(".trimmed.mp4")
    out_path = Path(out_path)
    if edl_path is None:
        edl_path = video_path.with_suffix(".trim.json")
    edl_path = Path(edl_path)

    duration = track_duration_s(track)
    plan = plan_trim(
        track["samples"],
        duration,
        min_dead_s=min_dead_s,
        stationary_px=stationary_px,
        stationary_window_s=stationary_window_s,
        pad_s=pad_s,
    )

    edl = {
        "video": str(video_path),
        "output": str(out_path),
        "source": "ball_track",
        **plan,
    }

    # Persist the plan before touching ffmpeg so the edit list survives even if
    # rendering fails (e.g. a degenerate keep list from sparse ball detections).
    edl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(edl_path, "w") as f:
        json.dump(edl, f, indent=2)
    edl["edl_path"] = str(edl_path)

    if not dry_run:
        assemble_segments(
            video_path, plan["keep_segments"], out_path, reencode=reencode
        )

    return edl
