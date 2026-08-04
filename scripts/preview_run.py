"""Render a run's tracks + ball back onto its video, to eyeball what a detector did.

`process` writes tracks.json and ball_track.json but nothing you can watch, so
judging a detector meant reading numbers. This draws them over the source video:
one box per track, the ball with a short trail, and a HUD carrying the counts.

**Box colour is the kit, not the colour on screen.** A box is coloured by the kit
`process` stamped into tracks.json, and "black" and "white" are unreadable as
drawing colours on a sunlit pitch — so black maps to orange and white to cyan,
with grey for a lane that got no kit at all. Pass `--team black` to have the
legend mark which one is ours.

Every tracked lane is drawn whatever its kit. `identify --team` gates the *name*,
never the box, so an opponent still gets a box — it just carries a bare lane id.

When the run has been through `identify`, jerseys.json is picked up too and each
box carries the name (and jersey number) that lane was given, so you can watch
identity survive — or not survive — a lane handoff. Named lanes are drawn with a
heavier box than anonymous ones; `--named-only` hides the anonymous ones entirely.

Detections are sampled (``sample_interval`` in tracks.json — 6 frames, i.e. 5 fps
at 30 fps video), so a box is *held* across the gap to the next sample rather
than interpolated. The staircase you see is real: it is the rate the pipeline
actually detects at, and smoothing it here would hide the flicker this view
exists to show.

Usage:
    python scripts/preview_run.py --run runs/<match_id> --start-s 10 --dur-s 20
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np

# BGR. The kit names ("black"/"white") are what the team classifier assigns; they
# are useless as *drawing* colours on a dark pitch, so each maps to a legible hue.
TEAM_COLOURS = {
    "black": (0, 140, 255),    # orange
    "white": (255, 255, 0),    # cyan
    "blue": (255, 120, 0),
    "red": (60, 60, 255),
}
UNKNOWN_COLOUR = (150, 150, 150)
BALL_COLOUR = (0, 255, 255)    # yellow
# a ball step this large between samples is a detector jump, not a kick
TELEPORT_PX = 250.0


def load_identities(path: Path) -> dict[int, str]:
    """track id -> display label, for the lanes `identify` was willing to name.

    Lanes it abstained on are left out rather than labelled "unknown": an empty
    label reads as "no claim made", which is what abstention means.
    """
    if not path.exists():
        return {}
    tracks = json.loads(path.read_text()).get("tracks") or {}
    out: dict[int, str] = {}
    for tid, rec in tracks.items():
        name, jersey = rec.get("name"), rec.get("jersey")
        if name:
            label = name.split()[0] + (f" #{jersey}" if jersey is not None else "")
        elif jersey is not None:
            label = f"#{jersey}"
        else:
            continue
        out[int(tid)] = label
    return out


def load_run(run: Path, jerseys: Path | None = None):
    tracks = json.loads((run / "tracks.json").read_text())
    ball = json.loads((run / "ball_track.json").read_text())
    interval = int(tracks.get("sample_interval") or 1)
    teams = tracks.get("teams") or {}
    names = load_identities(jerseys or run / "jerseys.json")

    # frame -> [(track_id, bbox, team, name)], so each rendered frame is one dict hit
    by_frame: dict[int, list] = {}
    for tid, rows in tracks["tracks"].items():
        team = teams.get(str(tid))
        name = names.get(int(tid))
        for row in rows:
            by_frame.setdefault(int(row["frame"]), []).append(
                (int(tid), row["bbox"], team, name)
            )

    ball_by_frame = {int(s["frame"]): s for s in ball["samples"]}
    return by_frame, ball_by_frame, interval, teams, names


def nearest_sample(frame_no: int, keys: list[int], interval: int) -> int | None:
    """The most recent sampled frame at or before frame_no, if within one interval."""
    i = bisect_right(keys, frame_no) - 1
    if i < 0:
        return None
    k = keys[i]
    return k if frame_no - k < interval else None


def draw_hud(img, frame_no, t_s, n_tracks, n_named, ball_vis, teams_seen,
             interval, fps, has_names, ours=None):
    h, w = img.shape[:2]
    pad = 12
    lines = [
        f"frame {frame_no}   t={t_s:6.2f}s",
        f"tracks in frame: {n_tracks}",
        f"ball: {'detected' if ball_vis else 'NOT DETECTED'}",
    ]
    if has_names:
        lines.insert(2, f"named by identify: {n_named}")
    box_h = 26 * len(lines) + 2 * pad
    box_w = 330
    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, line in enumerate(lines):
        colour = (255, 255, 255)
        if line.startswith("ball") and not ball_vis:
            colour = (80, 80, 255)
        cv2.putText(img, line, (pad + 12, pad + 26 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)

    # Legend, bottom-left. The swatch is the drawing hue and the word beside it is
    # the *kit* — an orange chip labelled "black" reads as a bug unless the legend
    # says so, so it spells out "black kit" and marks which one is ours.
    y = h - pad - 12
    items = [(f"{name} kit" + (" (ours)" if name == ours else ""),
              TEAM_COLOURS.get(name, UNKNOWN_COLOUR)) for name in teams_seen]
    items.append(("no kit assigned", UNKNOWN_COLOUR))
    items.append(("ball", BALL_COLOUR))
    lw = sum(40 + 12 * len(name) for name, _ in items) + 24
    overlay = img.copy()
    cv2.rectangle(overlay, (pad, y - 30), (pad + lw, y + 12), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    x = pad + 12
    for name, colour in items:
        cv2.rectangle(img, (x, y - 18), (x + 22, y - 2), colour, -1)
        cv2.putText(img, name, (x + 30, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        x += 40 + 12 * len(name)

    captions = [
        f"RF-DETR + ByteTrack | detections at {fps / interval:.1f} fps, held between samples",
    ]
    if has_names:
        # Every tracked lane is drawn whatever its kit; only the *name* is gated.
        captions.append("box colour = kit assigned by process | label = lane id, "
                        "then the name identify gave it (thick box = named)")
    for i, caption in enumerate(captions):
        cv2.putText(img, caption, (pad + 12, pad + box_h + 26 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--video", default=None,
                    help="source video (default: the 'video' field in tracks.json)")
    ap.add_argument("--start-s", type=float, default=10.0)
    ap.add_argument("--dur-s", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trail", type=int, default=8, help="ball trail length, in samples")
    ap.add_argument("--scale", type=float, default=1.0, help="output scale (0.5 = 720p from 1080p)")
    ap.add_argument("--jerseys", default=None, type=Path,
                    help="identities to label boxes with (default: jerseys.json in the run)")
    ap.add_argument("--named-only", action="store_true",
                    help="draw only the lanes identify was willing to name")
    ap.add_argument("--team", default=None,
                    help="our squad's kit for this match — marks it '(ours)' in the "
                         "legend. Does not filter: every tracked lane is still drawn")
    args = ap.parse_args()

    run = args.run.resolve()
    by_frame, ball_by_frame, interval, teams, names = load_run(run, args.jerseys)
    tracks_meta = json.loads((run / "tracks.json").read_text())
    # tracks.json records the proxy by bare name ("broadcast_proxy.mp4"), which
    # lives in the run dir and is a symlink to the source when --broadcast is off.
    video = Path(args.video) if args.video else Path(tracks_meta.get("video", ""))
    if not video.is_absolute() and not video.exists():
        video = run / video.name
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_f = int(round(args.start_s * fps))
    n_frames = int(round(args.dur_s * fps))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)

    out_path = Path(args.out) if args.out else run / f"preview_{int(args.start_s)}s.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    track_keys = sorted(by_frame)
    ball_keys = sorted(ball_by_frame)
    teams_seen = sorted({t for t in teams.values() if t})

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    trail: list[tuple[int, int]] = []
    n_vis = 0
    n_named_total = n_rows_total = 0

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        fn = start_f + i

        k = nearest_sample(fn, track_keys, interval)
        rows = by_frame.get(k, []) if k is not None else []
        if args.named_only:
            rows = [r for r in rows if r[3]]
        n_named = sum(1 for r in rows if r[3])
        n_named_total += n_named
        n_rows_total += len(rows)
        for tid, bbox, team, name in rows:
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            colour = TEAM_COLOURS.get(team, UNKNOWN_COLOUR) if team else UNKNOWN_COLOUR
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3 if name else 1)
            # lane id always, so a handoff is visible even when the name holds
            label = f"{tid}" + (f" {name}" if name else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1, cv2.LINE_AA)

        bk = nearest_sample(fn, ball_keys, interval)
        sample = ball_by_frame.get(bk) if bk is not None else None
        ball_vis = bool(sample and sample["visible"])
        if ball_vis:
            bx, by = int(round(sample["pixel_x"])), int(round(sample["pixel_y"]))
            if not trail or trail[-1] != (bx, by):
                trail.append((bx, by))
                trail[:] = trail[-args.trail:]
            n_vis += 1
            for j in range(1, len(trail)):
                a = int(255 * j / len(trail))
                p, q = trail[j - 1], trail[j]
                # A raw ball track teleports when the detector latches onto a
                # jersey number; drawn plainly, one such jump is a bright line
                # across the whole pitch and reads as motion. Draw the jumps
                # dim red instead, so they still show but as what they are.
                jump = np.hypot(q[0] - p[0], q[1] - p[1]) > TELEPORT_PX
                colour, width = ((0, 0, 140), 1) if jump else ((0, a, a), 2)
                cv2.line(frame, p, q, colour, width, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 16, BALL_COLOUR, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (bx, by), BALL_COLOUR, cv2.MARKER_CROSS, 12, 1)

        draw_hud(frame, fn, fn / fps, len(rows), n_named, ball_vis, teams_seen,
                 interval, fps, bool(names), ours=args.team)
        if args.scale != 1.0:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"frames rendered : {i + 1}")
    print(f"ball detected   : {n_vis}/{i + 1} ({100 * n_vis / max(1, i + 1):.1f}%)")
    print(f"boxes drawn     : {n_rows_total} ({n_rows_total / max(1, i + 1):.1f}/frame)")
    if names:
        print(f"named boxes     : {n_named_total} "
              f"({100 * n_named_total / max(1, n_rows_total):.1f}% of boxes drawn)")
    print(f"wrote           : {out_path}")


if __name__ == "__main__":
    main()
