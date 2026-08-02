"""Clip extraction using ffmpeg."""

from __future__ import annotations

import re
from pathlib import Path

from soccer_vision.io.video import ffmpeg_extract_clip

# Matches the names produced by ``extract_event_clips`` below:
# ``{prefix}_{index:03d}_{label}_{ts}s.mp4`` where label may contain underscores.
_CLIP_NAME_RE = re.compile(r"^(?P<prefix>.+?)_(?P<index>\d+)_(?P<label>.+)_(?P<ts>\d+)s\.mp4$")


#: A halo may cross a lane handoff only if a footballer could have covered the
#: distance. ~64 px per metre at this venue and a 10 m/s sprint gives ~640 px/s;
#: left generous because the camera pans, adding apparent speed.
_HALO_MAX_SPEED_PX_S = 900.0


def _lane_span(samples: list) -> tuple[int, int]:
    return samples[0][0], samples[-1][0]


def _foot(bbox) -> tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[3]))


def halo_samples_for(event: dict, halo_tracks: dict[int, list] | None,
                     *, extra_ids: set[int] | None = None,
                     fps: float = 30.0) -> list | None:
    """Track boxes to halo for one event, as **one lane at a time**, or ``None``.

    Uses ``track_ids`` when the event carries one (on-ball spans do — a player
    fragments across lanes mid-touch, and the spotlight has to follow through the
    handoff or it drops out partway through the clip), else the single
    ``track_id``. Those are the anchors: the lanes the event itself happened on.

    ``extra_ids`` offers every other lane ``identify`` gave the same name, so the
    halo can cover the seconds of the clip before the touch, when the player is
    usually on screen under a different lane id.

    **Those extras are candidates, not members**, and that distinction is the
    whole of this function. The previous version merged them all and let the
    earliest sample win each frame, on the stated assumption that lanes of one
    player are disjoint in time. They are not: naming runs per lane with no
    one-player-one-place constraint, so on ``runs/saints-u14g-full`` the name
    "Morgan Lobey" lands on 275 lanes, **two to four of them alive at once on
    18.4% of the frames she is named at**. All but one of those is another
    person, and the merge picked between them arbitrarily, frame by frame —
    which is precisely the halo that jumps between players and settles on empty
    grass.

    So a candidate joins only if it is a *possible continuation* of what has
    already been accepted: no overlap in time with an accepted lane, and close
    enough to one of them that a player could have run between the two in the
    gap. Everything else is dropped. That cannot make the halo correct — the
    name it started from may be wrong — but it does make it coherent: one person
    at a time, moving the way a person moves.
    """
    if not halo_tracks:
        return None

    ids = [int(t) for t in (event.get("track_ids") or [])]
    if not ids:
        tid = event.get("track_id")
        ids = [int(tid)] if tid is not None else []

    accepted: list[list] = []
    for tid in ids:
        lane = halo_tracks.get(tid)
        if lane:
            accepted.append(sorted(lane, key=lambda s: s[0]))
    if not accepted and not extra_ids:
        return None

    candidates = [
        sorted(halo_tracks[int(t)], key=lambda s: s[0])
        for t in (extra_ids or ())
        if int(t) not in set(ids) and halo_tracks.get(int(t))
    ]
    if not accepted and candidates:
        # No anchor lane (a plain --player clip): start from the longest
        # candidate, which is the one most likely to be a real, followable lane.
        candidates.sort(key=len, reverse=True)
        accepted.append(candidates.pop(0))

    # Grow greedily by nearest plausible continuation, re-checking every round:
    # accepting a lane changes which others are reachable.
    changed = True
    while changed and candidates:
        changed = False
        best, best_cost = None, None
        for i, cand in enumerate(candidates):
            cost = _continuation_cost(cand, accepted, fps)
            if cost is not None and (best_cost is None or cost < best_cost):
                best, best_cost = i, cost
        if best is not None:
            accepted.append(candidates.pop(best))
            changed = True

    merged = [s for lane in accepted for s in lane]
    if not merged:
        return None
    merged.sort(key=lambda s: s[0])
    deduped: list = []
    for sample in merged:
        if deduped and deduped[-1][0] == sample[0]:
            continue
        deduped.append(sample)
    return deduped


def _continuation_cost(cand: list, accepted: list[list], fps: float) -> float | None:
    """Implied px/s to reach ``cand`` from an accepted lane, or ``None`` if impossible.

    ``None`` means the candidate overlaps an accepted lane in time (two lanes of
    one player cannot both be live) or no accepted lane is near enough in space
    to be the same person. Otherwise the cost is the slowest such crossing, so
    the tightest continuation is taken first.
    """
    c0, c1 = _lane_span(cand)
    best = None
    for lane in accepted:
        a0, a1 = _lane_span(lane)
        if c0 <= a1 and a0 <= c1:
            return None                      # overlapping in time
        if c0 > a1:                          # candidate follows this lane
            gap_frames, here, there = c0 - a1, lane[-1][1], cand[0][1]
        else:                                # candidate precedes it
            gap_frames, here, there = a0 - c1, lane[0][1], cand[-1][1]
        gap_s = max(gap_frames / fps, 1e-3)
        fh, ft = _foot(here), _foot(there)
        speed = ((fh[0] - ft[0]) ** 2 + (fh[1] - ft[1]) ** 2) ** 0.5 / gap_s
        if speed <= _HALO_MAX_SPEED_PX_S and (best is None or speed < best):
            best = speed
    return best


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
