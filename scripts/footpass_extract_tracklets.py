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

from soccer_vision.detection.rfdetr import (
    BALL_CLASS_IDS,
    PLAYER_CLASS_IDS,
    RFDETRSoccerDetector,
)
from soccer_vision.tracking.bytetrack import create_tracker, track_detections
from soccer_vision.tracking.teams import TeamClassifier

REFEREE_CLASS_ID = 2  # RFDETR_CLASSES: 0 ball, 1 player, 2 referee, 3 goalkeeper


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

TEAM_COLOR = sv.ColorPalette(colors=[sv.Color(56, 130, 246), sv.Color(239, 68, 68)])


def build_field_polygon(video, f0, f1, W, H, n_samples=25, freq=0.4, min_area=0.15):
    """Turf-segmentation field polygon (no calibration needed).

    Samples ``n_samples`` frames, thresholds green turf in HSV, keeps pixels that
    are green in >= ``freq`` of samples (persistent field, not transient players),
    and returns the convex hull of the largest turf blob. Returns None if the blob
    is smaller than ``min_area`` of the frame (segmentation unreliable — caller
    then keeps all detections rather than nuking everything).
    """
    cap = cv2.VideoCapture(video)
    idxs = np.linspace(f0, max(f0, f1 - 1), n_samples).astype(int)
    acc = np.zeros((H, W), np.float32)
    used = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        acc += (cv2.inRange(hsv, (30, 25, 25), (95, 255, 255)) > 0).astype(np.float32)
        used += 1
    cap.release()
    if used == 0:
        return None
    field = ((acc / used) >= freq).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    field = cv2.morphologyEx(field, cv2.MORPH_CLOSE, k)
    field = cv2.morphologyEx(field, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    if cv2.contourArea(hull) < min_area * W * H:
        return None
    return hull  # (N,1,2) int32, ready for cv2.pointPolygonTest / polylines


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
    ap.add_argument("--field-mask", choices=["auto", "none"], default="auto",
                    help="auto: drop detections whose foot-point is off the turf (removes "
                         "spectators/coaches); none: keep all detections")
    ap.add_argument("--field-margin", type=float, default=20.0,
                    help="px slack outside the field polygon still counted as on-field")
    ap.add_argument("--ball-conf", type=float, default=0.2,
                    help="confidence threshold for the ball (kept lower than players — it's small)")
    ap.add_argument("--drop-referees", action="store_true", default=True,
                    help="drop player boxes that overlap a detected referee (default on)")
    ap.add_argument("--min-track-coverage", type=float, default=0.5,
                    help="drop a track present in fewer than this fraction of frames within its "
                         "own lifespan — flickery tracks are usually supporters/referees")
    ap.add_argument("--min-track-frames", type=int, default=10,
                    help="drop tracks seen in fewer than this many frames total")
    args = ap.parse_args()

    det = RFDETRSoccerDetector.from_pretrained(device=args.device)
    tracker = create_tracker()
    teams = TeamClassifier()

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    f0, f1 = args.start_frame, args.start_frame + args.num_frames

    # Spatial gate: a field polygon from turf segmentation. Detections whose
    # foot-point falls off the turf (spectators, coaches, ball kids) are dropped
    # before tracking/team-clustering — this both cleans the tracklets and stops
    # the jersey-colour clusterer being poisoned by sideline adults.
    field_poly = None
    if args.field_mask == "auto":
        field_poly = build_field_polygon(args.video, f0, f1, W, H)
        if field_poly is None:
            print("[field-mask] turf polygon unreliable — keeping all detections")
        else:
            print(f"[field-mask] field polygon: {len(field_poly)} pts, "
                  f"{cv2.contourArea(field_poly)/(W*H):.0%} of frame")

    def on_field_pt(x, y):
        if field_poly is None:
            return True
        return cv2.pointPolygonTest(field_poly, (float(x), float(y)), True) >= -args.field_margin

    def on_field(x1, y1, x2, y2):
        return on_field_pt((x1 + x2) / 2.0, y2)  # feet, where a player meets the pitch

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    # Pass 1: one detector pass per frame, split into ball / players / referees.
    rows = []       # (frame, track_id, x, y, w, h)
    ball_rows = []  # (frame, px, py, conf) — best on-field ball per frame
    n_off = n_ref = n_ball = 0
    for t in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        if (t - f0) % args.stride:
            continue
        dets = det.predict(frame, conf_threshold=min(args.ball_conf, args.conf))
        if len(dets) == 0:
            continue
        cid = np.asarray(dets.class_id)
        conf = np.asarray(dets.confidence)
        xyxy_all = np.asarray(dets.xyxy)

        # --- ball: highest-confidence ball detection that's on the field ---
        bmask = np.isin(cid, list(BALL_CLASS_IDS)) & (conf >= args.ball_conf)
        for bi in np.argsort(-conf[bmask]) if bmask.any() else []:
            bx, by, bx2, by2 = xyxy_all[bmask][bi]
            cx_b, cy_b = (bx + bx2) / 2.0, (by + by2) / 2.0
            if on_field_pt(cx_b, cy_b):
                ball_rows.append((t, float(cx_b), float(cy_b), float(conf[bmask][bi])))
                n_ball += 1
                break

        # --- referees: their boxes veto overlapping player boxes (RF-DETR often
        # also emits a spurious 'player' box on the same official) ---
        ref_boxes = xyxy_all[cid == REFEREE_CLASS_ID] if args.drop_referees else np.empty((0, 4))

        # --- players (+ goalkeepers), referee- and field-filtered, then tracked ---
        pmask = np.isin(cid, list(PLAYER_CLASS_IDS)) & (conf >= args.conf)
        keep = []
        for i in np.where(pmask)[0]:
            box = xyxy_all[i]
            if any(_iou(box, r) > 0.45 for r in ref_boxes):
                n_ref += 1
                continue
            keep.append(i)
        players = dets[np.array(keep, dtype=int)] if keep else dets[np.zeros(len(dets), bool)]
        tracked = track_detections(tracker, players)
        if tracked.tracker_id is None:
            continue
        for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            if not on_field(x1, y1, x2, y2):
                n_off += 1
                continue
            rows.append((t, int(tid), x1, y1, x2 - x1, y2 - y1))
            teams.add_sample(int(tid), frame, xyxy)
    cap.release()
    if field_poly is not None:
        print(f"[field-mask] dropped {n_off} off-field detections")
    print(f"[referee] dropped {n_ref} player boxes overlapping a referee")
    n_sampled = max(1, (f1 - f0) // args.stride)
    print(f"[ball] detected in {n_ball}/{n_sampled} frames ({n_ball/n_sampled:.0%})")

    # Track-continuity filter: an on-field player is present in most frames of its
    # lifespan; supporters and referees (RF-DETR flipping ref<->player) blink in and
    # out, so their identity is vacant for long stretches. Drop low-coverage / very
    # short tracks — this also keeps the top-13 TAAD slots and team clusters clean.
    per_track = {}
    for r in rows:
        per_track.setdefault(r[1], []).append(r[0])
    flicker = set()
    for tid, fs in per_track.items():
        fs = sorted(set(fs))
        span = (fs[-1] - fs[0]) / args.stride + 1
        coverage = len(fs) / span
        if coverage < args.min_track_coverage or len(fs) < args.min_track_frames:
            flicker.add(tid)
    n_before = len(rows)
    rows = [r for r in rows if r[1] not in flicker]
    print(f"[continuity] dropped {len(flicker)}/{len(per_track)} flickery tracks "
          f"({n_before - len(rows)} detections)")
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
        "field_mask": args.field_mask, "off_field_dropped": int(n_off),
        "field_polygon": field_poly.reshape(-1, 2).tolist() if field_poly is not None else None,
        "referees_dropped": int(n_ref),
        "tracks_dropped_flicker": int(len(flicker)),
        "ball_detected_frames": int(n_ball),
        "ball_track": [[int(t), round(px, 1), round(py, 1), round(c, 3)] for (t, px, py, c) in ball_rows],
    }
    Path(args.out_h5).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_h5} ({len(out)} rows, {manifest['n_tracks']} tracks) + manifest")

    if args.preview:
        _render_preview(args.video, arr, W, H, fps, f0, f1, args.preview, tname, field_poly,
                        {int(r[0]): (r[1], r[2]) for r in ball_rows})
    return 0


def _render_preview(video, arr, W, H, fps, f0, f1, out, tname, field_poly=None, ball_by_frame=None):
    ball_by_frame = ball_by_frame or {}
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
        if field_poly is not None:
            cv2.polylines(frame, [field_poly], True, (0, 255, 0), 2, cv2.LINE_AA)
        fr = arr[arr[:, FRAME] == t]
        if len(fr):
            xyxy = np.stack([fr[:, RX], fr[:, RY], fr[:, RX] + fr[:, RW], fr[:, RY] + fr[:, RH]], 1)
            det = sv.Detections(xyxy=xyxy.astype(np.float32), class_id=fr[:, TEAM].astype(int))
            labels = [f"t{int(r[PID])} {tname.get('team_a' if int(r[TEAM])==0 else 'team_b','?')}" for r in fr]
            frame = box.annotate(frame, det)
            frame = lab.annotate(frame, det, labels=labels)
        if t in ball_by_frame:
            bx, by = ball_by_frame[t]
            cv2.circle(frame, (int(bx), int(by)), 9, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(bx), int(by)), 9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(frame, f"OUR FOOTAGE  f{t}  {t/fps:.1f}s  RF-DETR+ByteTrack+team  (no TAAD yet)",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"wrote preview {out}")


if __name__ == "__main__":
    raise SystemExit(main())
