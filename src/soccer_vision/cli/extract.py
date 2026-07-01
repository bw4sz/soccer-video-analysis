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

    clip_paths = extract_event_clips(
        proxy_path, events, clips_dir,
        pre_s=args.pre, post_s=args.post,
    )
    print(f"\n{len(clip_paths)} clip(s) extracted to {clips_dir}/ ({_describe(args)})")


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

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths = []
        for i, event in enumerate(events):
            ts = event.get("timestamp_s", event.get("position_ms", 0) / 1000)
            start = max(0.0, ts - 5.0)
            tmp_path = Path(tmpdir) / f"tmp_{i:03d}.mp4"
            ffmpeg_extract_clip(proxy_path, start, 20.0, tmp_path)
            clip_paths.append(tmp_path)
        out_path = build_reel(clip_paths, args.out)
    print(f"Reel saved: {out_path} ({len(events)} clips — {_describe(args)})")
