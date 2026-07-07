"""CLI for the ``trim-empty`` editor — cut dead time, keep the original."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.clips.trim import (
    build_ball_track,
    load_ball_track,
    trim_empty,
    write_ball_track,
)


def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"


def run_trim_empty(args):
    """Remove offscreen/stationary dead time from a match video."""
    video = Path(args.video)
    if not video.exists():
        print(f"Video not found: {video}")
        return

    # Obtain a ball track: use the provided one, or build it from the detector.
    if args.track:
        track = load_ball_track(args.track)
    else:
        print(
            "No --track supplied; building one from the RF-DETR ball detector.\n"
            "  (Install the detector extras, or pass --track ball_track.json.\n"
            "   Schema: see soccer_vision.events.deadball.)"
        )
        try:
            track = build_ball_track(
                video,
                sample_fps=args.sample_fps,
                device=args.device,
                smooth=args.smooth,
            )
        except Exception as e:  # detector/weights missing → actionable message
            print(f"\nCould not build a ball track automatically: {e}")
            print(
                "Supply a precomputed track with --track once ball tracking is "
                "available. The trim workflow and schema are ready."
            )
            return
        if args.save_track:
            write_ball_track(track, args.save_track)
            print(f"Ball track saved: {args.save_track}")

    edl = trim_empty(
        video,
        track,
        out_path=args.out,
        edl_path=args.edl,
        min_dead_s=args.min_dead,
        stationary_px=args.stationary_px,
        pad_s=args.pad,
        reencode=not args.copy,
        dry_run=args.dry_run,
    )

    src = edl["source_duration_s"]
    kept = edl["kept_duration_s"]
    removed = edl["removed_duration_s"]
    pct = (removed / src * 100) if src else 0.0
    print(
        f"\nSource {_fmt(src)} → kept {_fmt(kept)} "
        f"(removed {_fmt(removed)}, {pct:.0f}% of the match) "
        f"across {len(edl['removed_segments'])} dead span(s)."
    )
    print(f"Edit-decision list: {edl['edl_path']}")
    if args.dry_run:
        print("Dry run — no video written. Drop --dry-run to render the cut.")
    else:
        print(f"Trimmed video: {edl['output']} (original left untouched)")
