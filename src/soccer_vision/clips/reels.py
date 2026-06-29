"""Highlight reel generation: per-player and per-event clip merging."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.io.video import ffmpeg_concat, ffmpeg_extract_clip


def build_reel(
    clip_paths: list[Path],
    out_path: str | Path,
) -> Path:
    """Concatenate clips into a single highlight reel."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(clip_paths) == 1:
        import shutil
        shutil.copy2(clip_paths[0], out_path)
    else:
        ffmpeg_concat(clip_paths, out_path)

    print(f"  Reel saved: {out_path} ({len(clip_paths)} clips)")
    return out_path


def build_event_reel(
    video_path: str | Path,
    events: list[dict],
    out_path: str | Path,
    *,
    event_label: str | None = None,
    pre_s: float = 5.0,
    post_s: float = 15.0,
) -> Path:
    """Build a reel from events, optionally filtering by label."""
    import tempfile

    if event_label:
        events = [e for e in events if e.get("label") == event_label]

    if not events:
        raise ValueError(f"No events found for label '{event_label}'")

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths = []
        for i, event in enumerate(events):
            ts = event.get("timestamp_s", event.get("position_ms", 0) / 1000)
            start = max(0.0, ts - pre_s)
            tmp_path = Path(tmpdir) / f"tmp_{i:03d}.mp4"
            ffmpeg_extract_clip(video_path, start, pre_s + post_s, tmp_path)
            clip_paths.append(tmp_path)
        return build_reel(clip_paths, out_path)
