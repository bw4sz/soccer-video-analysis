"""Visualize FOOTPASS player-centric annotations on the broadcast video using
roboflow supervision, writing an annotated mp4.

Each visible player gets a box coloured by **team** (LEFT_TO_RIGHT) and labelled
with **jersey** (SHIRT_NUMBER); when a player performs an **action** the class
name is shown (held for a few frames since actions are single-frame anchors).

Data contract — the tactical h5 stores one row per (player, frame):
    FRAME, PLAYER_ID, LEFT_TO_RIGHT(team 0/1), SHIRT_NUMBER, ROLE_ID,
    X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_W, ROI_H, CLS
ROI_* are fullHD (1920x1080); scale x/3.0, y/3.068 for the 352x640 video.
CLS: 0 = no action; 1..8 = the eight classes below.

IMPORTANT: team + jersey come from the *tracklets* (tracking input), not from
TAAD. TAAD predicts only CLS. So `--source gt` shows ground-truth CLS; once a
model is trained, `--source pred --pred <npz>` overlays predicted CLS on the same
tracklets.

Usage (ground-truth demo on a val game):
    python scripts/footpass_visualize.py --game game_18_H1 --split val \
        --start-frame 15811 --num-frames 350 --out /tmp/footpass_gt.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import supervision as sv

# CLS 1..8 -> name (0 is background/no-action).
CLASSES = ["Pass", "Drive", "Cross", "Shot", "Header", "Throw-in", "Tackle", "Block"]
# columns
FRAME, PID, TEAM, JERSEY, ROLE, XP, YP, XS, YS, RX, RY, RW, RH, CLS = range(14)
SX, SY = 3.0, 1080 / 352  # fullHD -> 352x640 scale (x, y)
TEAM_COLOR = sv.ColorPalette(colors=[sv.Color(56, 130, 246), sv.Color(239, 68, 68)])  # team0 blue, team1 red


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/blue/ewhite/b.weinstein/soccer-vision-data/footpass")
    ap.add_argument("--game", required=True, help="h5 key, e.g. game_18_H1")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--num-frames", type=int, default=350)
    ap.add_argument("--action-hold", type=int, default=12, help="frames to keep an action label visible")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data_root)
    h5_path = root / "data" / f"{args.split}_tactical_data.h5"
    game_n = args.game.split("_")[1]
    video_path = root / "videos" / f"game_{game_n}.mp4"

    with h5py.File(h5_path, "r") as h5:
        rows = h5[args.game][:]
    f0, f1 = args.start_frame, args.start_frame + args.num_frames
    win = rows[(rows[:, FRAME] >= f0) & (rows[:, FRAME] < f1)]

    # Pre-index action anchors so we can "hold" them for a few frames.
    actions = win[win[:, CLS] != 0]  # (frame, player, team, jersey, ..., cls)

    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    box = sv.BoxAnnotator(color=TEAM_COLOR, thickness=2, color_lookup=sv.ColorLookup.CLASS)
    lab = sv.LabelAnnotator(color=TEAM_COLOR, color_lookup=sv.ColorLookup.CLASS,
                            text_scale=0.4, text_thickness=1, text_padding=2)

    n_events = 0
    for t in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        fr = win[(win[:, FRAME] == t) & ~np.isnan(win[:, RX])]
        if len(fr):
            xyxy = np.stack([fr[:, RX] / SX, fr[:, RY] / SY,
                             (fr[:, RX] + fr[:, RW]) / SX, (fr[:, RY] + fr[:, RH]) / SY], axis=1)
            team = fr[:, TEAM].astype(int)
            det = sv.Detections(xyxy=xyxy.astype(np.float32), class_id=team)

            labels = []
            for r in fr:
                jersey = int(r[JERSEY])
                # is this player mid-action within the hold window?
                a = actions[(actions[:, PID] == r[PID]) &
                            (actions[:, FRAME] <= t) & (actions[:, FRAME] > t - args.action_hold)]
                if len(a):
                    cls = int(a[-1, CLS])
                    labels.append(f"#{jersey} {CLASSES[cls - 1].upper()}")
                else:
                    labels.append(f"#{jersey}")
            frame = box.annotate(frame, det)
            frame = lab.annotate(frame, det, labels=labels)

        n_events += int(((actions[:, FRAME] == t)).sum())
        # HUD banner
        cv2.rectangle(frame, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(frame, f"game_{game_n}  f{t}  {t/fps:6.1f}s  GT tracklets  events so far: {n_events}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"wrote {args.out}  ({args.num_frames} frames, {len(actions)} action anchors in window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
