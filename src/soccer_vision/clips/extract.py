"""Clip extraction using ffmpeg."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.io.video import ffmpeg_extract_clip


def extract_event_clips(
    video_path: str | Path,
    events: list[dict],
    out_dir: str | Path,
    *,
    pre_s: float = 5.0,
    post_s: float = 30.0,
    prefix: str = "clip",
    reencode: bool = True,
) -> list[Path]:
    """Extract a clip for each event, return list of output paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []

    for i, event in enumerate(events, 1):
        ts = event.get("timestamp_s", event.get("position_ms", 0) / 1000)
        label = event.get("label", "event")
        start = max(0.0, ts - pre_s)
        duration = pre_s + post_s
        out_path = out_dir / f"{prefix}_{i:03d}_{label}_{ts:.0f}s.mp4"

        print(f"  [{i}/{len(events)}] {label} at {ts:.1f}s → {out_path.name}")
        ffmpeg_extract_clip(video_path, start, duration, out_path, reencode=reencode)
        clip_paths.append(out_path)

    return clip_paths
