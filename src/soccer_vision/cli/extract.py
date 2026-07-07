"""CLI for clip extraction and reel building."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.clips.extract import extract_event_clips
from soccer_vision.clips.reels import build_reel
from soccer_vision.events.select import filter_events
from soccer_vision.io.osl import read_osl
from soccer_vision.io.video import ffmpeg_extract_clip


def _load_events(run_dir: Path) -> list[dict]:
    """Load OSL events with a timestamp_s field ready for extraction."""
    osl_doc = read_osl(run_dir / "annotations.json")
    events = osl_doc.get("events", [])
    for e in events:
        if "timestamp_s" not in e:
            e["timestamp_s"] = e.get("position_ms", 0) / 1000
    return events


def _describe(args) -> str:
    parts = []
    if getattr(args, "event", None):
        parts.append(f"event={args.event}")
    if getattr(args, "events", None):
        parts.append(f"events={args.events}")
    if getattr(args, "team", None):
        parts.append(f"team={args.team}")
    if getattr(args, "track", None) is not None:
        parts.append(f"track={args.track}")
    return ", ".join(parts) or "all events"


def _load_halo(run_dir: Path, style: str | None):
    """Resolve a ``--halo`` request to ``(track_boxes, style, max_gap_frames)``.

    Returns ``(None, None, 0)`` when halos aren't requested or ``tracks.json`` is
    missing (in which case a warning is printed and extraction proceeds plainly).
    """
    if not style:
        return None, None, 0

    import json

    from soccer_vision.clips.halo import load_track_boxes

    tracks_path = run_dir / "tracks.json"
    if not tracks_path.exists():
        print(f"--halo requested but {tracks_path} is missing "
              "(re-run `process` to generate it). Extracting plain clips.")
        return None, None, 0

    meta = json.loads(tracks_path.read_text())
    max_gap = int(meta.get("sample_interval", 5)) * 4
    return load_track_boxes(tracks_path), style, max_gap


def run_extract(args):
    """Extract clips from a processed run directory."""
    run_dir = Path(args.run)
    proxy_path = run_dir / "broadcast_proxy.mp4"
    clips_dir = run_dir / "clips"

    events = _load_events(run_dir)

    # --events takes one or more labels; --team / --track narrow further.
    if args.events:
        events = [e for e in events if e.get("label") in args.events]
    events = filter_events(events, team=args.team, track_id=args.track)

    if not events:
        print(f"No matching events found ({_describe(args)}).")
        return

    halo_tracks, halo_style, halo_max_gap = _load_halo(run_dir, getattr(args, "halo", None))

    clip_paths = extract_event_clips(
        proxy_path, events, clips_dir,
        pre_s=args.pre, post_s=args.post,
        **({"halo_tracks": halo_tracks, "halo_style": halo_style,
            "halo_max_gap_frames": halo_max_gap} if halo_tracks is not None else {}),
    )
    haloed = " with halo" if halo_tracks is not None else ""
    print(f"\n{len(clip_paths)} clip(s) extracted{haloed} to {clips_dir}/ ({_describe(args)})")


def run_reel(args):
    """Build a highlight reel from a processed run, filtered by event/team/player."""
    import tempfile

    run_dir = Path(args.run)
    proxy_path = run_dir / "broadcast_proxy.mp4"

    if args.player:
        print("Note: --player (roster name) needs jersey mapping (future phase); "
              "use --track <id> to select a specific player for now.")

    events = _load_events(run_dir)
    events = filter_events(
        events, label=args.event, team=args.team, track_id=args.track
    )

    if not events:
        print(f"No matching events found ({_describe(args)}).")
        return

    halo_tracks, halo_style, halo_max_gap = _load_halo(run_dir, getattr(args, "halo", None))

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths = []
        for i, event in enumerate(events):
            ts = event.get("timestamp_s", event.get("position_ms", 0) / 1000)
            start = max(0.0, ts - 5.0)
            tmp_path = Path(tmpdir) / f"tmp_{i:03d}.mp4"
            tid = event.get("track_id")
            samples = halo_tracks.get(int(tid)) if halo_tracks and tid is not None else None
            if samples:
                from soccer_vision.clips.halo import render_halo_clip

                render_halo_clip(proxy_path, tmp_path, start_s=start, duration_s=20.0,
                                 track_samples=samples, style=halo_style,
                                 max_gap_frames=halo_max_gap)
            else:
                ffmpeg_extract_clip(proxy_path, start, 20.0, tmp_path)
            clip_paths.append(tmp_path)
        out_path = build_reel(clip_paths, args.out)
    print(f"Reel saved: {out_path} ({len(events)} clips — {_describe(args)})")
