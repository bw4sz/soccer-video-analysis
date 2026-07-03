"""Clip extraction using ffmpeg."""

from __future__ import annotations

import re
from pathlib import Path

from soccer_vision.io.video import ffmpeg_extract_clip

# Matches the names produced by ``extract_event_clips`` below:
# ``{prefix}_{index:03d}_{label}_{ts}s.mp4`` where label may contain underscores.
_CLIP_NAME_RE = re.compile(r"^(?P<prefix>.+?)_(?P<index>\d+)_(?P<label>.+)_(?P<ts>\d+)s\.mp4$")


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


def parse_clip_name(path: str | Path) -> dict | None:
    """Parse an ``extract_event_clips`` filename back into its parts.

    Returns ``{"index", "label", "timestamp_s", "path"}`` or ``None`` if the name
    does not follow the scheme.
    """
    path = Path(path)
    m = _CLIP_NAME_RE.match(path.name)
    if not m:
        return None
    return {
        "index": int(m["index"]),
        "label": m["label"],
        "timestamp_s": float(m["ts"]),
        "path": path,
    }


def pair_events_with_clips(
    events: list[dict],
    clips_dir: str | Path,
    *,
    ts_tol_s: float = 2.0,
) -> list[tuple[dict, Path | None]]:
    """Pair each event with its extracted clip.

    ``extract_event_clips`` enumerates ``events`` in order and encodes the
    1-based index in the filename, so index alignment is the primary match; the
    timestamp embedded in the name is used to validate it and, if it disagrees,
    to fall back to the nearest unused clip. Events with no clip pair to ``None``.
    """
    parsed = [p for p in (parse_clip_name(c) for c in sorted(Path(clips_dir).glob("*.mp4"))) if p]
    by_index = {p["index"]: p for p in parsed}
    used: set[Path] = set()

    def _event_ts(event: dict) -> float:
        return event.get("timestamp_s", event.get("position_ms", 0) / 1000)

    pairs: list[tuple[dict, Path | None]] = []
    for i, event in enumerate(events, start=1):
        ev_ts = _event_ts(event)
        chosen: Path | None = None

        cand = by_index.get(i)
        if cand and cand["path"] not in used and abs(cand["timestamp_s"] - ev_ts) <= ts_tol_s:
            chosen = cand["path"]

        if chosen is None:  # fall back to nearest unused clip by timestamp
            remaining = [p for p in parsed if p["path"] not in used]
            if remaining:
                best = min(remaining, key=lambda p: abs(p["timestamp_s"] - ev_ts))
                if abs(best["timestamp_s"] - ev_ts) <= ts_tol_s:
                    chosen = best["path"]

        if chosen is not None:
            used.add(chosen)
        pairs.append((event, chosen))

    return pairs
