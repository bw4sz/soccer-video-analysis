"""CLI for clip extraction and reel building."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.clips.extract import extract_event_clips
from soccer_vision.clips.reels import build_event_reel
from soccer_vision.io.osl import read_osl


def run_extract(args):
    """Extract clips from a processed run directory."""
    run_dir = Path(args.run)
    osl_path = run_dir / "annotations.json"
    proxy_path = run_dir / "broadcast_proxy.mp4"
    clips_dir = run_dir / "clips"

    osl_doc = read_osl(osl_path)
    events = osl_doc.get("events", [])

    if args.events:
        events = [e for e in events if e.get("label") in args.events]

    if not events:
        print("No matching events found.")
        return

    # Convert position_ms to timestamp_s for extraction
    for e in events:
        if "timestamp_s" not in e:
            e["timestamp_s"] = e.get("position_ms", 0) / 1000

    clip_paths = extract_event_clips(
        proxy_path, events, clips_dir,
        pre_s=args.pre, post_s=args.post,
    )
    print(f"\n{len(clip_paths)} clip(s) extracted to {clips_dir}/")


def run_reel(args):
    """Build a highlight reel from a processed run."""
    run_dir = Path(args.run)
    osl_path = run_dir / "annotations.json"
    proxy_path = run_dir / "broadcast_proxy.mp4"

    osl_doc = read_osl(osl_path)
    events = osl_doc.get("events", [])

    for e in events:
        if "timestamp_s" not in e:
            e["timestamp_s"] = e.get("position_ms", 0) / 1000

    out_path = build_event_reel(
        proxy_path, events, args.out,
        event_label=args.event,
    )
    print(f"Reel saved: {out_path}")
