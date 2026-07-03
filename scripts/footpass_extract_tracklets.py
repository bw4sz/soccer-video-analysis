"""Extract FOOTPASS-format tracklets from one of our videos (pipeline "A").

Composes the existing soccer-vision modules — RF-DETR player detection
(`detection/rfdetr.py`) + ByteTrack (`tracking/bytetrack.py`) + jersey-colour
team clustering (`tracking/teams.py`) — and writes a tactical HDF5 in the schema
the FOOTPASS TAAD dataloader expects, so a trained TAAD can predict actions on
our footage.

Schema (one row per (player, frame), matching `utils/TAAD_Dataset.py`):
    FRAME, PLAYER_ID, LEFT_TO_RIGHT(team 0/1), SHIRT_NUMBER, ROLE_ID,
    X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_W, ROI_H, CLS

Gaps vs FOOTPASS (documented, fine for a baseline run):
  - SHIRT_NUMBER = -1 (no jersey OCR yet).
  - ROLE_ID = 0 (no tactical-role model).
  - X/Y_POS = normalised bbox foot-point (no pitch homography); speeds = finite diff.
  - CLS = 0 (unknown — that's what TAAD predicts).
The TAAD baseline only consumes the video clip + ROI boxes (roi_align), so ROI +
team + frame are the fields that matter for inference.

ROI is stored in the video's native pixels. Our footage is 1920x1080 = FOOTPASS's
"fullHD" convention, so TAAD's fullHD->352x640 scaling applies if the clip is fed
at 352x640. The manifest records WxH so the inference adapter can rescale.

Usage (short CPU smoke):
    python scripts/footpass_extract_tracklets.py \
        --video data/match-...mp4 --start-frame 18000 --num-frames 150 --stride 1 \
        --game-key our_match_01_H1 --out-h5 /blue/.../our_tracklets.h5 --preview preview.mp4
Full match should run on GPU (--device cuda) via SLURM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import supervision as sv

from soccer_vision.detection.rfdetr import RFDETRSoccerDetector
from soccer_vision.tracking.bytetrack import create_tracker, track_detections
from soccer_vision.tracking.teams import TeamClassifier

TEAM_COLOR = sv.ColorPalette(colors=[sv.Color(56, 130, 246), sv.Color(239, 68, 68)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=150)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--game-key", default="our_match_01_H1")
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--preview", default=None, help="optional annotated mp4 to sanity-check tracking")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    det = RFDETRSoccerDetector.from_pretrained(device=args.device)
    tracker = create_tracker()
    teams = TeamClassifier()

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    f0, f1 = args.start_frame, args.start_frame + args.num_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    # Pass 1: detect + track, accumulate rows and team-colour samples.
    rows = []  # (frame, track_id, x, y, w, h)
    for t in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        if (t - f0) % args.stride:
            continue
        players = det.predict_players(frame, conf_threshold=args.conf)
        tracked = track_detections(tracker, players)
        if tracked.tracker_id is None:
            continue
        for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            rows.append((t, int(tid), x1, y1, x2 - x1, y2 - y1))
            teams.add_sample(int(tid), frame, xyxy)
    cap.release()
    teams.fit()

    # Map team colour-name -> LEFT_TO_RIGHT 0/1 (stable: first-seen team = 0).
    tname = teams.team_names()  # {"team_a": "blue", "team_b": "red"}
    key_to_lr = {"team_a": 0, "team_b": 1}
    print(f"teams: {tname}  ({len(rows)} player-detections)")

    # Build the FOOTPASS schema array; speeds via per-track finite differences.
    rows.sort(key=lambda r: (r[1], r[0]))
    out = []
    prev = {}
    for (t, tid, x, y, w, h) in rows:
        team_key = teams._track_team.get(tid)  # team_a / team_b / None
        lr = key_to_lr.get(team_key, 0)
        cx, foot = (x + w / 2) / W, (y + h) / H  # normalised foot point
        if tid in prev:
            pt, pcx, pfoot = prev[tid]
            dt = max(1, t - pt)
            xs, ys = (cx - pcx) / dt, (foot - pfoot) / dt
        else:
            xs = ys = 0.0
        prev[tid] = (t, cx, foot)
        out.append([t, tid, lr, -1, 0, cx, foot, xs, ys, x, y, w, h, 0])
    arr = np.array(out, dtype=np.float32)

    Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out_h5, "w") as h5:
        h5.create_dataset(args.game_key, data=arr)
    manifest = {
        "video": args.video, "game_key": args.game_key, "width": W, "height": H,
        "fps": fps, "start_frame": f0, "num_frames": args.num_frames, "stride": args.stride,
        "n_tracks": int(len(set(r[1] for r in rows))), "n_rows": len(out),
        "team_names": tname, "note": "SHIRT_NUMBER=-1, ROLE_ID=0, CLS=0 (unknown)",
    }
    Path(args.out_h5).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_h5} ({len(out)} rows, {manifest['n_tracks']} tracks) + manifest")

    if args.preview:
        _render_preview(args.video, arr, W, H, fps, f0, f1, args.preview, tname)
    return 0


def _render_preview(video, arr, W, H, fps, f0, f1, out, tname):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    box = sv.BoxAnnotator(color=TEAM_COLOR, thickness=2, color_lookup=sv.ColorLookup.CLASS)
    lab = sv.LabelAnnotator(color=TEAM_COLOR, color_lookup=sv.ColorLookup.CLASS,
                            text_scale=0.5, text_thickness=1, text_padding=2)
    FRAME, PID, TEAM, RX, RY, RW, RH = 0, 1, 2, 9, 10, 11, 12
    for t in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        fr = arr[arr[:, FRAME] == t]
        if len(fr):
            xyxy = np.stack([fr[:, RX], fr[:, RY], fr[:, RX] + fr[:, RW], fr[:, RY] + fr[:, RH]], 1)
            det = sv.Detections(xyxy=xyxy.astype(np.float32), class_id=fr[:, TEAM].astype(int))
            labels = [f"t{int(r[PID])} {tname.get('team_a' if int(r[TEAM])==0 else 'team_b','?')}" for r in fr]
            frame = box.annotate(frame, det)
            frame = lab.annotate(frame, det, labels=labels)
        cv2.rectangle(frame, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(frame, f"OUR FOOTAGE  f{t}  {t/fps:.1f}s  RF-DETR+ByteTrack+team  (no TAAD yet)",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"wrote preview {out}")


if __name__ == "__main__":
    raise SystemExit(main())
