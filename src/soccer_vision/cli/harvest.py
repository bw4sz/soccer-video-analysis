"""CLI for ``soccer-vision harvest`` — pull CC-BY youth-soccer clips."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.harvest import harvest
from soccer_vision.harvest.queries import DEFAULT_QUERIES


def _load_queries(args) -> list[str]:
    if args.queries:
        return args.queries
    if args.queries_file:
        text = Path(args.queries_file).read_text(encoding="utf-8")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    return DEFAULT_QUERIES


def run_harvest(args):
    """Search YouTube, keep CC-BY youth matches, save a midpoint clip each."""
    # Fail fast (once) if yt-dlp/ffmpeg aren't available, rather than emitting
    # the install hint per search query.
    from soccer_vision.harvest.youtube import _import_yt_dlp

    try:
        _import_yt_dlp()
    except RuntimeError as e:
        print(e)
        return

    try:
        result = harvest(
            args.out_dir,
            target=args.n,
            queries=_load_queries(args),
            clip_len_s=args.clip_len,
            position=args.position,
            position_frac=args.position_frac,
            max_per_channel=args.max_per_channel,
            per_query=args.per_query,
            min_duration_s=args.min_duration,
            max_height=args.max_height,
            dry_run=args.dry_run,
        )
    except RuntimeError as e:  # missing yt-dlp/ffmpeg → actionable message
        print(e)
        return

    verb = "would keep" if args.dry_run else "downloaded"
    print(
        f"\nDone. {verb} {result.downloaded} clip(s) "
        f"(scanned {result.scanned} videos).\n"
        f"  skipped — already have: {result.skipped_seen}, "
        f"not CC-BY: {result.skipped_license}, "
        f"too short: {result.skipped_duration}, "
        f"channel cap: {result.skipped_channel_cap}\n"
        f"  manifest: {result.manifest_path}"
    )
    if not args.dry_run and result.downloaded:
        print(f"  attribution: {result.manifest_path.with_name('ATTRIBUTION.md')}")
