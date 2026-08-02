"""Clip extraction using ffmpeg."""

from __future__ import annotations

import re
from pathlib import Path

from soccer_vision.io.video import ffmpeg_extract_clip

# Matches the names produced by ``extract_event_clips`` below:
# ``{prefix}_{index:03d}_{label}_{ts}s.mp4`` where label may contain underscores.
_CLIP_NAME_RE = re.compile(r"^(?P<prefix>.+?)_(?P<index>\d+)_(?P<label>.+)_(?P<ts>\d+)s\.mp4$")


def halo_samples_for(event: dict, halo_tracks: dict[int, list] | None,
                     *, extra_ids: set[int] | None = None) -> list | None:
    """Track boxes to halo for one event, or ``None``.

    Uses ``track_ids`` when the event carries one (on-ball spans do — a player
    fragments across lanes mid-touch, and the spotlight has to follow through the
    handoff or it drops out partway through the clip), else the single
    ``track_id``.

    ``extra_ids`` adds every other lane belonging to the same player. A clip
    opens several seconds before the touch, and the lane the touch happened on
    typically starts *after* the clip does — on the U14G match a 7.9s clip whose
    lane began 4.8s in, so the spotlight was missing for most of it and then
    appeared, which reads as the halo lagging. The player is usually on screen
    that whole time under a different lane id, so halo the player.

    Lanes of one player are disjoint in time by construction, so the merged
    samples read as one continuous track. Where two lanes do overlap (a wrong
    link, or two lanes of the same player alive at once) the earlier sample wins
    for that frame rather than the halo flickering between them.
    """
    if not halo_tracks:
        return None

    ids = list(event.get("track_ids") or [])
    if not ids:
        tid = event.get("track_id")
        ids = [tid] if tid is not None else []
    if extra_ids:
        ids = list(ids) + [t for t in extra_ids if t not in set(ids)]

    merged: list = []
    for tid in ids:
        merged.extend(halo_tracks.get(int(tid)) or [])
    if not merged:
        return None
    merged.sort(key=lambda s: s[0])
    deduped: list = []
    for sample in merged:
        if deduped and deduped[-1][0] == sample[0]:
            continue
        deduped.append(sample)
    return deduped


def extract_event_clips(
    video_path: str | Path,
    events: list[dict],
    out_dir: str | Path,
    *,
    pre_s: float = 5.0,
    post_s: float = 30.0,
    prefix: str = "clip",
    reencode: bool = True,
    halo_tracks: dict[int, list] | None = None,
    halo_color: tuple[int, int, int] = (0, 215, 255),
    halo_style: str = "ellipse",
    halo_max_gap_frames: int = 20,
) -> list[Path]:
    """Extract a clip for each event, return list of output paths.

    When ``halo_tracks`` is given (``{track_id: [(frame, bbox), ...]}`` from
    :func:`soccer_vision.clips.halo.load_track_boxes`), any event carrying a
    matching ``track_id`` is re-rendered with a soft team-coloured spotlight on
    that player; other events fall back to a plain ffmpeg cut.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []

    for i, event in enumerate(events, 1):
        ts = event.get("timestamp_s", event.get("position_ms", 0) / 1000)
        label = event.get("label", "event")
        start = max(0.0, ts - pre_s)
        duration = pre_s + post_s
        out_path = out_dir / f"{prefix}_{i:03d}_{label}_{ts:.0f}s.mp4"

        tid = event.get("track_id")
        samples = halo_samples_for(event, halo_tracks)
        if samples:
            from soccer_vision.clips.halo import render_halo_clip

            print(f"  [{i}/{len(events)}] {label} at {ts:.1f}s → "
                  f"{out_path.name} (halo track {tid})")
            render_halo_clip(
                video_path, out_path, start_s=start, duration_s=duration,
                track_samples=samples, color=halo_color, style=halo_style,
                max_gap_frames=halo_max_gap_frames,
            )
        else:
            if halo_tracks is not None:
                print(f"  [{i}/{len(events)}] {label} at {ts:.1f}s → "
                      f"{out_path.name} (no track — plain cut)")
            else:
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
