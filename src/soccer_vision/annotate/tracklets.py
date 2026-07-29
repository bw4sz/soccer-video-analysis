"""Label whole tracklets from one clip, instead of one box per still frame.

The frame workflow costs a decision per box and yields one crop from it. This
one plays a window of the match with every tracked player ringed and numbered,
and asks for a name per number — so one decision harvests every crop in that
lane, and the annotator gets the cues they actually identify children by:
where someone is on the pitch, who they are next to, how they move.

**Numbers are per window, not per track, and they are not jersey numbers.**
Label Studio's labelling config is fixed for a project while ByteTrack ids differ
in every window, so a config naming real track ids is impossible. Each window
instead picks its longest lanes and numbers them 1..N *in the order they first
appear*; the config declares N dropdowns once, and the manifest records which
track id each slot meant. Enrolment joins them back.

**What a lane is worth.** On the U14G RF-DETR run a lane runs 7 detection-frames
(~1.4 s) at the median, so its crops are near-duplicates: one pose, one patch of
pitch, one light. A lane multiplies *crops* far faster than it multiplies the
*diversity* a gallery actually wants, which is why windows are spread across the
match (and why the answer to a thin gallery is more windows and more videos, not
longer clips).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Slot colours (BGR), chosen to stay distinct on green turf and in shadow.
SLOT_COLORS = [
    (66, 133, 244), (60, 200, 255), (80, 220, 80), (200, 120, 255),
    (0, 165, 255), (255, 200, 60), (120, 120, 255), (180, 255, 255),
    (255, 130, 180), (100, 255, 200), (200, 200, 200), (0, 100, 255),
    (255, 255, 120), (140, 200, 90), (255, 160, 220), (90, 180, 255),
]

NOT_OURS = "not ours"
UNSURE = "unsure"


def slot_label(slot: int) -> str:
    """What goes on the chip in the clip. Digits, matching the form's "Player N".

    Letters were tried, to stop a chip reading as a jersey number. They cost more
    than they saved: the form lists "Player 1..N", so a clip showing "J" makes the
    annotator translate on every decision. Whatever is drawn must be the same
    token the dropdown is named after — that matters more than the ambiguity
    with squad numbers, which the instruction header can address instead.
    """
    return str(int(slot))


def choose_windows(
    track_samples: dict[int, list],
    *,
    fps: float,
    window_s: float,
    n_windows: int,
    min_track_frames: int,
    max_lanes: int,
    teams: dict[int, str] | None = None,
    team: str | None = None,
) -> list[dict]:
    """Pick evenly spread windows and the lanes to ring in each.

    Even spread beats picking the busiest windows: a gallery built from one
    passage of play sees one end of the pitch in one light, and the sun moves
    through a youth match. Within a window the *longest* lanes are ringed, since
    a lane that survives longer both yields more crops and is easier to follow
    with the eye.

    Lanes are capped at ``max_lanes`` because the number of dropdowns is fixed
    by the config, and a window ringing thirty players is one nobody will finish.
    """
    if not track_samples:
        return []

    wanted = (team or "").strip().lower()
    eligible = {}
    for tid, samples in track_samples.items():
        if wanted and (teams or {}).get(tid, "").lower() != wanted:
            continue
        if len(samples) >= min_track_frames:
            eligible[tid] = samples
    if not eligible:
        return []

    first = min(s[0][0] for s in eligible.values())
    last = max(s[-1][0] for s in eligible.values())
    span_frames = max(1, last - first)
    window_frames = int(round(window_s * fps))

    starts = []
    if n_windows == 1 or span_frames <= window_frames:
        starts = [first]
    else:
        step = (span_frames - window_frames) / (n_windows - 1)
        starts = [int(round(first + i * step)) for i in range(n_windows)]

    windows = []
    for w, start in enumerate(starts):
        end = start + window_frames
        present = []
        for tid, samples in eligible.items():
            inside = [(f, b) for f, b in samples if start <= f <= end]
            if len(inside) >= min_track_frames:
                present.append((len(inside), tid, inside))
        if not present:
            continue
        # Pick the longest lanes — they yield the most crops and are the easiest
        # to follow — but *number* them in the order they first appear. Numbering
        # by length scatters the sequence across the clip: on the U14G run that
        # put slot 10 on screen at 0.0s and slot 1 at 1.8s, so an annotator
        # scrubbing through met "2, 10, 3" and reasonably concluded the numbering
        # was broken. Chronological order means playing the clip once walks the
        # form from top to bottom.
        present.sort(reverse=True, key=lambda x: (x[0], -x[1]))
        chosen = sorted(present[:max_lanes], key=lambda x: (x[2][0][0], -x[0]))
        lanes = [
            {"slot": i + 1, "track_id": tid, "n_frames": n,
             "first_frame": inside[0][0], "last_frame": inside[-1][0],
             "enters_s": round((inside[0][0] - start) / fps, 1),
             "leaves_s": round((inside[-1][0] - start) / fps, 1)}
            for i, (n, tid, inside) in enumerate(chosen)
        ]
        windows.append({
            "window": w + 1,
            "start_frame": start,
            "end_frame": end,
            "start_s": round(start / fps, 2),
            "duration_s": round(window_frames / fps, 2),
            "lanes": lanes,
        })
    return windows


def render_window_clip(
    video_path: str | Path,
    out_path: str | Path,
    *,
    window: dict,
    track_samples: dict[int, list],
    fps: float,
    max_gap_frames: int = 20,
) -> Path:
    """Write the window with each lane ringed in its slot colour and numbered.

    Encoded through ffmpeg to H.264: OpenCV's ``mp4v`` writes a file most
    browsers refuse to play, and this clip exists to be played in one.
    """
    import cv2

    from soccer_vision.clips.halo import interpolate_bbox
    from soccer_vision.io.video import ffmpeg_run

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix(".raw.mp4")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(window["start_frame"])
    n_frames = int(round(window["duration_s"] * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    try:
        for offset in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            for lane in window["lanes"]:
                bbox = interpolate_bbox(track_samples[lane["track_id"]],
                                        start_frame + offset, max_gap_frames)
                if bbox is None:
                    continue
                _draw_slot(cv2, frame, bbox, lane["slot"])
            writer.write(frame)
    finally:
        writer.release()
        cap.release()

    ffmpeg_run([
        "ffmpeg", "-y",
        "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        # `main` profile and yuv420p are what every browser decodes without
        # argument; the default `high` profile buys nothing on footage nobody is
        # grading.
        "-profile:v", "main", "-level", "4.0", "-pix_fmt", "yuv420p",
        # No audio. A silent AAC track was tried on the theory that Label
        # Studio's documented "H.264 + AAC" meant it needed one; it does not —
        # these clips play video-only — and `anullsrc` is an infinite input that
        # `-shortest` failed to bound, so the encode ran away.
        "-an",
        # Puts the moov atom first so a player can start on the first bytes
        # instead of fetching all 15 MB.
        "-movflags", "+faststart",
        str(out_path),
    ])
    raw_path.unlink(missing_ok=True)
    return out_path


def _draw_slot(cv2, frame, bbox, slot: int):
    """Ring the player and put the slot number in a chip above them.

    Sized against the **frame**, not the box. A label scaled to a 50 px player
    is unreadable the moment a browser fits 1080p into half a screen, and the
    number is the only thing the annotator has to answer against — it has to
    survive that downscale. A filled chip rather than outlined text for the same
    reason: solid colour holds up under video compression where thin strokes
    smear into the turf.
    """
    colour = SLOT_COLORS[(slot - 1) % len(SLOT_COLORS)]
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    cx, w = (x1 + x2) // 2, max(2, x2 - x1)
    fh = frame.shape[0]

    # An ellipse at the feet, the broadcast convention: it marks the player
    # without hiding the body, which is the part being identified.
    cv2.ellipse(frame, (cx, y2), (max(8, w // 2), max(4, w // 4)),
                0, -30, 210, colour, max(2, fh // 400), cv2.LINE_AA)

    label = slot_label(slot)
    scale = max(0.75, fh / 1300.0)
    thick = max(2, int(round(fh / 540.0)))
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    pad = max(4, th // 3)
    cy = max(th + 2 * pad, y1 - pad)
    tl = (cx - tw // 2 - pad, cy - th - 2 * pad)
    br = (cx + tw // 2 + pad, cy + base)
    cv2.rectangle(frame, tl, br, colour, -1, cv2.LINE_AA)
    cv2.rectangle(frame, tl, br, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (cx - tw // 2, cy - pad), cv2.FONT_HERSHEY_DUPLEX,
                scale, (0, 0, 0), thick, cv2.LINE_AA)


def labeling_config(names: list[str], max_lanes: int) -> str:
    """One dropdown per slot, all declared up front.

    A dropdown rather than radio buttons: a squad of thirteen rendered as inline
    choices, times a dozen slots, is a page nobody can scan. ``NOT_OURS`` and
    ``UNSURE`` are first-class options because the alternative is an annotator
    guessing to clear the form, and a guess enrols the wrong appearance.
    """
    from xml.sax.saxutils import escape

    options = "".join(
        f'\n      <Choice value="{escape(n)}"/>' for n in [*names, NOT_OURS, UNSURE]
    )
    blocks = "".join(
        f'\n    <View style="display:inline-block; width:210px; padding:4px">'
        f'\n      <Header value="Player {slot_label(slot)}" size="6"/>'
        f'\n      <Choices name="p{slot}" toName="video" choice="single" '
        f'layout="select" showInline="false">{options}\n      </Choices>'
        f'\n    </View>'
        for slot in range(1, max_lanes + 1)
    )
    return (
        '<View>\n'
        '  <Header value="Name each ringed player. The number on the chip is the '
        '&quot;Player N&quot; below it — it is NOT a jersey number. Players are '
        'numbered in the order they first appear, and each is only on screen for '
        'part of the clip (see the list under the video). Use &quot;not ours&quot; '
        'for opponents, referees and spectators, and &quot;unsure&quot; when you '
        'can\'t tell — both are skipped, never guessed."/>\n'
        '  <Video name="video" value="$video"/>\n'
        '  <Text name="onscreen" value="$onscreen"/>\n'
        f'  <View style="display:flex; flex-wrap:wrap">{blocks}\n  </View>\n'
        '</View>\n'
    )


def clip_url(clip_path: Path, serve_root: Path, base_url: str | None) -> str:
    """URL for a clip: a plain HTTP one when ``base_url`` is given, else local-files.

    Label Studio's ``/data/local-files/`` endpoint needs two environment variables
    set before the server starts and a document root at exactly the right level,
    and when any of that is off the player reports "cannot open video" — which
    looks like a codec problem and isn't. Serving the folder with
    ``python -m http.server`` sidesteps the whole mechanism, so ``--serve-url``
    exists for when the built-in route is fighting you.
    """
    rel = clip_path.resolve().relative_to(Path(serve_root).resolve()).as_posix()
    if base_url:
        return f"{base_url.rstrip('/')}/{rel}"
    return f"/data/local-files/?d={rel}"


def onscreen_guide(window: dict, max_lanes: int) -> str:
    """"Player 3: 4.8-19.8s" per slot — when to look for each one.

    A 20s window holds ten lanes but rarely more than three at once, so an
    annotator who plays it once sees a couple of chips and cannot tell whether
    the rest are missing or merely elsewhere in time. Spelling out each slot's
    span turns that into a scrub target. Slots the window doesn't use are named
    too, because a form with ten dropdowns and six players otherwise looks
    broken.
    """
    parts = [f"Player {lane['slot']}: {lane['enters_s']:.0f}-{lane['leaves_s']:.0f}s"
             for lane in window["lanes"]]
    unused = [str(s) for s in range(len(window["lanes"]) + 1, max_lanes + 1)]
    if unused:
        parts.append(f"(Players {', '.join(unused)} are not in this clip "
                     f"— leave them blank)")
    return "On screen —  " + "  ·  ".join(parts)


def build_tasks(windows: list[dict], clip_urls: dict[int, str], fps: float,
                max_lanes: int = 12) -> list[dict]:
    """One task per window, carrying the slot→track map the export won't."""
    tasks = []
    for w in windows:
        tasks.append({
            "data": {
                "video": clip_urls[w["window"]],
                "onscreen": onscreen_guide(w, max_lanes),
                "window": w["window"],
                "start_frame": w["start_frame"],
                "start_s": w["start_s"],
                "timestamp": _clock(w["start_s"]),
                "slots": {str(lane["slot"]): lane["track_id"] for lane in w["lanes"]},
                "n_lanes": len(w["lanes"]),
            }
        })
    return tasks


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60:d}:{int(seconds) % 60:02d}"


def boxes_from_tracklet_export(
    export: list[dict],
    manifest: dict,
    track_samples: dict[int, list],
    *,
    max_samples_per_lane: int = 20,
) -> tuple[list[tuple[int, np.ndarray, str]], dict]:
    """Turn named slots into ``(frame, bbox, name)`` crops, plus a summary.

    The slot→track map lives in the manifest rather than the export, so a task
    edited or re-imported in Label Studio can't silently renumber anyone. Lanes
    named ``NOT_OURS`` / ``UNSURE`` contribute nothing, and a lane is sampled at
    most ``max_samples_per_lane`` times — 129 near-identical crops from one lane
    would swamp a gallery built from a dozen views.
    """
    windows = {w["window"]: w for w in manifest["windows"]}
    out: list[tuple[int, np.ndarray, str]] = []
    summary = {"lanes_named": 0, "lanes_skipped": 0, "windows": 0, "per_player": {}}

    for task in export:
        data = task.get("data", {}) or {}
        window = windows.get(data.get("window"))
        if window is None:
            continue
        lane_of = {lane["slot"]: lane for lane in window["lanes"]}
        seen_window = False
        for ann in task.get("annotations", []) or []:
            for res in ann.get("result", []) or []:
                choices = (res.get("value", {}) or {}).get("choices") or []
                if not choices:
                    continue
                name = str(choices[0]).strip()
                slot = _slot_of(res.get("from_name"))
                if slot is None or slot not in lane_of:
                    continue
                if name.lower() in (NOT_OURS, UNSURE, ""):
                    summary["lanes_skipped"] += 1
                    continue
                lane = lane_of[slot]
                samples = [
                    (f, b) for f, b in track_samples.get(lane["track_id"], [])
                    if lane["first_frame"] <= f <= lane["last_frame"]
                ]
                if not samples:
                    continue
                step = max(1, len(samples) // max_samples_per_lane)
                kept = samples[::step][:max_samples_per_lane]
                out.extend((f, b, name) for f, b in kept)
                summary["lanes_named"] += 1
                summary["per_player"][name] = summary["per_player"].get(name, 0) + len(kept)
                seen_window = True
        summary["windows"] += int(seen_window)
    return out, summary


def _slot_of(from_name) -> int | None:
    """``"p7"`` → 7. Anything else is a control we didn't write."""
    s = str(from_name or "")
    if s.startswith("p") and s[1:].isdigit():
        return int(s[1:])
    return None


def write_manifest(path: Path, *, run_dir: Path, video: Path, fps: float,
                   windows: list[dict], max_lanes: int) -> Path:
    path.write_text(json.dumps({
        "run": str(run_dir),
        "video": str(video),
        "fps": fps,
        "max_lanes": max_lanes,
        "windows": windows,
    }, indent=1))
    return path
