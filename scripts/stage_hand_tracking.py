#!/usr/bin/env python
"""Stage a Label Studio project where *you* draw and follow the players.

The tracklet projects ask a different question: our tracker chose the lanes, and
the annotator only names them. That is the right shape for building a gallery —
one decision harvests a whole lane of crops — but it is the wrong shape for
answering "did we track our players at all", because a player the tracker never
found has no ring, no slot and no dropdown. She is invisible to the form.

This dump inverts that. The clips come out **clean** — no rings, no slot chips —
and the annotator draws a box on each of our players and carries it through the
clip with keyframes. Label Studio interpolates between them, so following a
player for 20 s costs a handful of clicks rather than 600. What comes back is
where our players actually were, owing nothing to the tracker, which is the only
kind of answer sheet that can measure a miss.

Frame alignment is exact and it matters: clip frame 0 is the window's
``start_frame`` in the source video, recorded per task and in the manifest, so a
hand-drawn box maps back to a source frame with no seeking guesswork.

Usage:

    python scripts/stage_hand_tracking.py --run runs/saints-u14g-full-30fps \\
        --profile examples/profiles/saints-u14g.yaml \\
        --out runs/saints-u14g-full-30fps/hand_tracking \\
        --at 2400 --window 20 --n-windows 9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from soccer_vision.annotate.hand_tracking import UNSURE
from soccer_vision.annotate.tracklets import clip_url, render_window_clip
from soccer_vision.cli.enroll import label_names
from soccer_vision.profiles.loader import get_roster, load_profile

# Drawn on the pitch, these read apart from each other and from black/white kit
# and green turf. Label Studio colours the box and the timeline track by label,
# so a clip being followed by four hands at once stays readable.
PALETTE = [
    "#E4572E", "#17BEBB", "#FFC914", "#2E86AB", "#A23B72", "#76B041",
    "#F26419", "#5B5F97", "#C33C54", "#3DA5D9", "#EA7AF4", "#8AC926",
    "#FF477E",
]


def labeling_config(names: list[str], fps: float) -> str:
    """Video object tracking: one label per player, boxes carried by keyframe.

    ``framerate`` must match the clip, or Label Studio's frame indices do not
    correspond to source frames and every box lands at the wrong instant.

    The label list carries no ``not ours`` entry, unlike the tracklet config.
    There, every lane was already ringed and had to be dispatched somewhere; here
    nothing is drawn until the annotator draws it, so an opponent is handled by
    not drawing her. ``unsure`` survives, for one of ours whose face never turns.
    """
    from xml.sax.saxutils import escape

    labels = "\n".join(
        f'    <Label value="{escape(n)}" background="{PALETTE[i % len(PALETTE)]}"/>'
        for i, n in enumerate(names)
    )
    # Grey, and deliberately not from the palette: a squad bigger than the
    # palette wraps, and "unsure" sharing a colour with a real player is the one
    # collision that would actually mislead someone mid-clip.
    labels += f'\n    <Label value="{escape(UNSURE)}" background="#8A8A8A"/>'
    return (
        '<View>\n'
        '  <Header value="Draw a box on each of OUR players and follow her '
        'through the clip. Pick her name first, drag a box, then scrub forward '
        'and drag the box back onto her every second or so — Label Studio fills '
        'in between. Ignore opponents, referees and the crowd: leave them '
        'undrawn. Use &quot;unsure&quot; for one of ours you cannot name."/>\n'
        f'  <Labels name="player" toName="video">\n{labels}\n  </Labels>\n'
        f'  <Video name="video" value="$video" framerate="{fps:.6f}"/>\n'
        '  <VideoRectangle name="box" toName="video"/>\n'
        '</View>\n'
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="processed run (for the video and fps)")
    ap.add_argument("--out", required=True, help="project directory to write")
    ap.add_argument("--profile", help="team profile, for the roster label list")
    ap.add_argument("--at", type=float, required=True,
                    help="seconds into the match where the windows start")
    ap.add_argument("--window", type=float, default=20.0, help="window length (s)")
    ap.add_argument("--n-windows", type=int, default=9,
                    help="how many back-to-back windows from --at")
    ap.add_argument("--video", help="override the run's video path")
    ap.add_argument("--serve-root", default=".",
                    help="LOCAL_FILES_DOCUMENT_ROOT the clips are served under")
    ap.add_argument("--base-url", help="serve over plain HTTP from here instead")
    args = ap.parse_args()

    run_dir = Path(args.run)
    tracks = json.loads((run_dir / "tracks.json").read_text())
    fps = float(tracks["fps"])
    video = Path(args.video) if args.video else (run_dir / "broadcast_proxy.mp4")
    if not video.exists():
        video = Path(tracks["video"])

    profile = load_profile(args.profile) if args.profile else None
    names = label_names(get_roster(profile) if profile else [])
    if not names:
        print("  NOTE: no --profile roster, so the label list is only 'unsure'.")

    out_dir = Path(args.out)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    window_frames = int(round(args.window * fps))
    base = int(round(args.at * fps))

    windows, tasks = [], []
    for i in range(args.n_windows):
        start = base + i * window_frames
        start_s = start / fps
        window = {
            "window": i + 1,
            "start_frame": start,
            "start_s": round(start_s, 2),
            "duration_s": round(window_frames / fps, 2),
            # No lanes: this is the whole point. A clean clip owes the annotator
            # nothing and shows her no ring to be anchored by.
            "lanes": [],
        }
        clip = clips_dir / f"window_{i + 1:03d}_{int(round(start_s))}s.mp4"
        print(f"  window {i + 1}/{args.n_windows}  {start_s:.0f}s -> {clip.name}")
        render_window_clip(video, clip, window=window, track_samples={}, fps=fps)
        windows.append(window)
        tasks.append({
            "video": clip_url(clip, Path(args.serve_root), args.base_url),
            "window": window["window"],
            # Carried on the task so an export alone is enough to place a box in
            # the source video; the manifest is a backstop, not the only copy.
            "start_frame": start,
            "fps": fps,
        })

    (out_dir / "labeling_config.xml").write_text(labeling_config(names, fps))
    (out_dir / "label_studio_tasks.json").write_text(json.dumps(tasks, indent=2))
    (out_dir / "hand_tracking.json").write_text(json.dumps({
        "run": str(run_dir),
        "video": str(video),
        "fps": fps,
        "window_s": args.window,
        "windows": windows,
    }, indent=2))

    print(f"\nwrote {out_dir}")
    print(f"  {len(tasks)} clips, {args.window:.0f}s each, no rings drawn")
    print("  import label_studio_tasks.json (not the clips folder) and paste "
          "labeling_config.xml into Labeling Setup -> Code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
