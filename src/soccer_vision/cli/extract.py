"""CLI for clip extraction and reel building."""

from __future__ import annotations

from pathlib import Path

from soccer_vision.clips.extract import extract_event_clips, halo_samples_for
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
    if getattr(args, "player", None):
        parts.append(f"player={args.player}")
    if getattr(args, "number", None) is not None:
        parts.append(f"number={args.number}")
    return ", ".join(parts) or "all events"


def _resolve_player_tracks(args, run_dir: Path) -> set[int] | None:
    """Resolve ``--player`` / ``--number`` to a set of track ids, or ``None``.

    ``None`` means no player filter was requested. An empty set means a filter
    was requested but nothing matched (caller should report no clips). Reads
    ``jerseys.json`` from the run (written by `soccer-vision identify`) and, for
    a name, the profile roster.
    """
    import json

    player = getattr(args, "player", None)
    number = getattr(args, "number", None)
    if not player and number is None:
        return None

    jerseys_path = run_dir / "jerseys.json"
    if not jerseys_path.exists():
        print(f"--player/--number needs jersey numbers: run "
              f"`soccer-vision identify --run {run_dir}` first.")
        return set()

    from soccer_vision.identify.resolve import tracks_for
    from soccer_vision.profiles.loader import load_profile

    jerseys_doc = json.loads(jerseys_path.read_text())
    profile = load_profile(args.profile) if getattr(args, "profile", None) else None
    tids = tracks_for(jerseys_doc, number=number, name=player, profile=profile)
    if not tids:
        who = player or f"#{number}"
        print(f"No track resolved to {who} in {jerseys_path.name}.")
    return tids


def _on_ball_events(args, run_dir: Path, target_ids: set[int]) -> list[dict]:
    """On-ball spans for ``target_ids``, as events. Empty list if unavailable.

    Reads ``ball_track.json`` + ``tracks.json`` from the run; both are written by
    `process`. Prints why it came back empty rather than failing silently, since
    this runs as a fallback and a silent empty result looks like "this player did
    nothing" instead of "the run predates ball_track.json".
    """
    import json

    from soccer_vision.events.on_ball import select_on_ball_spans, spans_to_events

    ball_path, tracks_path = run_dir / "ball_track.json", run_dir / "tracks.json"
    missing = [p.name for p in (ball_path, tracks_path) if not p.exists()]
    if missing:
        print(f"  on-ball fallback needs {' and '.join(missing)} — re-run `process` "
              f"to generate them.")
        return []

    ball_track = json.loads(ball_path.read_text())
    tracks = json.loads(tracks_path.read_text())
    spans = select_on_ball_spans(
        ball_track, tracks, target_ids,
        max_ball_dist_px=getattr(args, "on_ball_dist", 90.0),
        min_span_s=getattr(args, "on_ball_min_span", 0.4),
    )
    return spans_to_events(
        spans, track_teams=tracks.get("teams"), team=getattr(args, "team", None)
    )


def _on_ball_fallback(
    args, run_dir: Path, player_tracks: set[int] | None, labels: list[str] | None
) -> list[dict]:
    """On-ball events to use when a player selection matched no detector events.

    The set-piece detector fires a handful of times per match, so "every clip of
    number 6" almost always comes back empty from the event stream alone. When a
    player was named and nothing matched, we fall back to the moments that player
    was nearest the ball — a far denser and, for this question, more faithful
    signal. Returns ``[]`` when the fallback shouldn't or can't run.

    Deliberately *not* triggered when an explicit event label was requested:
    ``--events pass`` returning on-ball touches instead would answer a different
    question than the one asked. ``--on-ball`` forces it anyway; ``--no-on-ball``
    disables it entirely.
    """
    if not getattr(args, "on_ball", True) and not getattr(args, "on_ball_force", False):
        return []

    targets = set(player_tracks or ())
    if getattr(args, "track", None) is not None:
        targets.add(args.track)
    if not targets:
        return []  # nothing to anchor on — a team-only query isn't a player query

    if labels and not getattr(args, "on_ball_force", False):
        print("  (no on-ball fallback: an explicit event label was requested. "
              "Pass --on-ball to cut ball-proximity spans instead.)")
        return []

    return _on_ball_events(args, run_dir, targets)


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

    player_tracks = _resolve_player_tracks(args, run_dir)
    if player_tracks == set():
        return  # a player filter was asked for but resolved to nothing

    events = _load_events(run_dir)

    # --events takes one or more labels; --team / --track / --player narrow further.
    if args.events:
        events = [e for e in events if e.get("label") in args.events]
    events = filter_events(
        events, team=args.team, track_id=args.track, track_ids=player_tracks
    )

    # --on-ball asks for proximity spans outright, so it must win even when the
    # detector *did* match something: a couple of stray set-piece hits on the
    # player's lanes would otherwise silently suppress the spans that were
    # explicitly requested.
    if not events or getattr(args, "on_ball_force", False):
        spans = _on_ball_fallback(args, run_dir, player_tracks, args.events)
        if spans:
            reason = ("--on-ball requested" if events else "No detector events matched")
            print(f"{reason}; cutting {len(spans)} on-ball "
                  f"span(s) ({_describe(args)}).")
            events = spans
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


def _reel_window(
    event: dict, *, pre_s: float = 5.0, default_s: float = 20.0
) -> tuple[float, float]:
    """``(start_s, duration_s)`` for one reel clip.

    A detector event is an instant, so it gets a fixed window. An on-ball span
    has a real duration, so the window covers the touch plus the same lead-in —
    otherwise a 1.2s touch and a 40s dribble would both become 20s of footage,
    and consecutive touches would overlap into near-duplicate clips.
    """
    ts = event.get("timestamp_s", event.get("position_ms", 0) / 1000)
    duration = event.get("duration_s")
    span = default_s if not duration else pre_s + float(duration) + pre_s / 2
    return max(0.0, ts - pre_s), span


def run_reel(args):
    """Build a highlight reel from a processed run, filtered by event/team/player."""
    import tempfile

    run_dir = Path(args.run)
    proxy_path = run_dir / "broadcast_proxy.mp4"

    player_tracks = _resolve_player_tracks(args, run_dir)
    if player_tracks == set():
        return  # a player filter was asked for but resolved to nothing

    events = _load_events(run_dir)
    events = filter_events(
        events, label=args.event, team=args.team, track_id=args.track,
        track_ids=player_tracks,
    )

    # See run_extract: --on-ball is a request, not just a fallback, so it takes
    # precedence over a thin set-piece match on the same lanes.
    if not events or getattr(args, "on_ball_force", False):
        spans = _on_ball_fallback(
            args, run_dir, player_tracks, [args.event] if args.event else None
        )
        if spans:
            reason = ("--on-ball requested" if events else "No detector events matched")
            print(f"{reason}; building a reel from "
                  f"{len(spans)} on-ball span(s) ({_describe(args)}).")
            events = spans
    if not events:
        print(f"No matching events found ({_describe(args)}).")
        return

    halo_tracks, halo_style, halo_max_gap = _load_halo(run_dir, getattr(args, "halo", None))

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths = []
        for i, event in enumerate(events):
            start, duration = _reel_window(event)
            tmp_path = Path(tmpdir) / f"tmp_{i:03d}.mp4"
            samples = halo_samples_for(event, halo_tracks)
            if samples:
                from soccer_vision.clips.halo import render_halo_clip

                render_halo_clip(proxy_path, tmp_path, start_s=start, duration_s=duration,
                                 track_samples=samples, style=halo_style,
                                 max_gap_frames=halo_max_gap)
            else:
                ffmpeg_extract_clip(proxy_path, start, duration, tmp_path)
            clip_paths.append(tmp_path)
        out_path = build_reel(clip_paths, args.out)
    print(f"Reel saved: {out_path} ({len(events)} clips — {_describe(args)})")
