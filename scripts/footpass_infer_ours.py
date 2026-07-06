"""Run a trained FOOTPASS TAAD baseline on OUR footage tracklets and visualize.

Companion to `footpass_extract_tracklets.py`. That script writes a tactical HDF5
(one row per player-frame) with ROLE_ID=0 (we have no tactical-role model). The
stock `vendor/FOOTPASS/run_TAAD_on_matches.py` groups ROIs by ROLE_ID 1..13, so
it produces empty ROIs on our data. But the TAAD network doesn't use role
*semantics* — the M=26 dimension is just a set of temporally-consistent tracklet
slots fed to roi_align. So here we assign each ByteTrack track to a per-team slot
(1..13, top-13 by frame count), which is all the model needs.

Pipeline: h5 tracklets + our video -> per-clip TAAD logits -> per-(slot,class)
temporal NMS decode -> action events -> annotated mp4 + key-frame PNGs.

Class order is the trainer's (train_TAAD_Baseline.py):
    0 background, 1 drive, 2 pass, 3 cross, 4 throw-in, 5 shot, 6 header,
    7 tackle, 8 block.
NOTE: this differs from the (buggy) CLASSES list in footpass_visualize.py — use
this one.

Usage (footpass env, GPU — the model's forward hardcodes .cuda()):
    /blue/ewhite/b.weinstein/envs/footpass/bin/python scripts/footpass_infer_ours.py \
        --h5 /blue/.../our_saints_setpiece.h5 --game-key our_saints_setpiece \
        --checkpoint /blue/.../runs/taad_03072026_1113/checkpoints/best_model.pt \
        --out-dir /blue/.../taad_ours_smoke
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from decord import VideoReader, cpu

# FOOTPASS repo on sys.path for the model import.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "FOOTPASS"))
from models.model_TAAD_baseline import X3D_TAAD_Baseline  # noqa: E402

NAMES = ["background", "drive", "pass", "cross", "throw-in", "shot", "header", "tackle", "block"]
# BGR colours per action for overlays (background never drawn).
ACT_COLOR = {
    1: (255, 200, 0), 2: (0, 200, 255), 3: (0, 255, 120), 4: (60, 60, 255),
    5: (255, 0, 200), 6: (200, 255, 0), 7: (0, 140, 255), 8: (180, 120, 255),
}
FRAME, PID, LR, SHIRT, ROLE, XP, YP, XS, YS, RX, RY, RW, RH, CLS = range(14)
COEFF = 1.125  # ROI expansion, matches run_TAAD_on_matches.py
SX, SY = 3.0, 1080 / 352.0  # fullHD (1920x1080) -> 352x640


def assign_slots(arr):
    """Return (work array with ROLE=slot 1..13, {slot_global: track_id})."""
    work = arr.copy()
    slot_to_track = {}
    for lr in (0, 1):
        sub = work[work[:, LR] == lr]
        if not len(sub):
            continue
        ids, counts = np.unique(sub[:, PID], return_counts=True)
        order = ids[np.argsort(-counts)][:13]  # top-13 tracks by presence
        smap = {int(t): i + 1 for i, t in enumerate(order)}
        for i, t in enumerate(order):
            slot_to_track[lr * 13 + i] = int(t)   # global slot index -> track
        m = work[:, LR] == lr
        work[m, ROLE] = [smap.get(int(t), 0) for t in work[m, PID]]
    return work, slot_to_track


def get_clip(vr, frame_ids):
    frames = vr.get_batch(np.asarray(frame_ids, dtype=np.int64)).asnumpy()
    h, w, _ = frames[0].shape
    if (w, h) != (640, 352):
        frames = np.stack([cv2.resize(f, (640, 352), interpolation=cv2.INTER_AREA) for f in frames], 0)
    clip = frames.astype(np.float32) / 255.0
    clip = (clip - 0.45) / 0.225
    return clip  # (T,352,640,3)


def get_roi_masks(data, frame_range):
    """(26,T,5) rois + (26,T) masks, grouped by (LR, ROLE=slot 1..13)."""
    T = len(frame_range)
    all_rois, all_masks = [], []
    for lr in (0, 1):
        for role_id in range(1, 14):
            local = data[(data[:, LR] == lr) & (data[:, ROLE] == role_id)]
            roi_seq, mask_seq = [], []
            for tidx, t in enumerate(frame_range):
                bb = local[(local[:, FRAME] == t) & (~np.isnan(local[:, RX]))] if len(local) else []
                if len(bb):
                    tlx = max(min(1920, int(bb[0, RX] - ((COEFF - 1.0) * bb[0, RW] // 2))), 0)
                    tly = max(min(1080, int(bb[0, RY] - ((COEFF - 1.0) * bb[0, RH] // 2))), 0)
                    brx = max(min(1920, int(bb[0, RX] + (COEFF * bb[0, RW]))), 0)
                    bry = max(min(1080, int(bb[0, RY] + (COEFF * bb[0, RH]))), 0)
                    roi_seq.append([tidx, int(tlx / SX), int(tly / SY), int(brx / SX), int(bry / SY)])
                    mask_seq.append(1.0)
                else:
                    roi_seq.append([tidx, 50.0, 50.0, 72.5, 99.0])
                    mask_seq.append(0.0)
            all_rois.append(np.array(roi_seq, dtype=np.float32))
            all_masks.append(np.array(mask_seq, dtype=np.float32))
    return np.stack(all_rois, 0), np.stack(all_masks, 0)  # (26,T,5), (26,T)


def nms_decode(logits, conf, nms, minf):
    """logits (9,26,L) -> list of events (abs_frame, slot, cls, score)."""
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=0).numpy()
    events = []
    L = probs.shape[-1]
    for m in range(probs.shape[1]):
        cls_t = probs[:, m, :].argmax(0)          # (L,)
        score_t = probs[:, m, :].max(0)
        for c in range(1, 9):
            cand = np.where((cls_t == c) & (score_t > conf))[0]
            for t in cand:
                lo, hi = max(0, t - nms), min(L, t + nms + 1)
                if score_t[t] >= score_t[lo:hi][cls_t[lo:hi] == c].max():
                    events.append((int(minf + t), m, c, float(score_t[t])))
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--game-key", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--nms", type=int, default=15)
    ap.add_argument("--clip-length", type=int, default=50)
    ap.add_argument("--max-keyframes", type=int, default=8)
    ap.add_argument("--ball-gate", choices=["soft", "off"], default="soft",
                    help="soft: tag events by ball distance and drop only far+weak ones "
                         "(keeps strong off-ball actions); off: no gating")
    ap.add_argument("--ball-radius", type=float, default=0.18,
                    help="event is 'near ball' if within this fraction of frame width")
    ap.add_argument("--ball-far", type=float, default=0.32,
                    help="soft gate only considers events beyond this fraction of width")
    ap.add_argument("--ball-weak", type=float, default=0.45,
                    help="soft gate only drops far events below this score (strong ones survive)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "keyframes").mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.h5).with_suffix(".manifest.json").read_text())
    video = manifest["video"]
    fps = manifest.get("fps", 29.97)
    tname = manifest.get("team_names", {})
    Wm = manifest.get("width", 1920)
    ball_xy = {int(bt[0]): (float(bt[1]), float(bt[2])) for bt in (manifest.get("ball_track") or [])}

    with h5py.File(args.h5, "r") as f:
        arr = f[args.game_key][:].astype(np.float64)
    work, slot_to_track = assign_slots(arr)

    frames = np.sort(np.unique(work[:, FRAME]))
    minf, maxf = int(frames.min()), int(frames.max())
    T = args.clip_length

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = X3D_TAAD_Baseline()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()

    vr = VideoReader(video, ctx=cpu(0))
    starts = list(range(minf, maxf + 1, T))
    all_logits = np.zeros((9, 26, len(starts) * T), dtype=np.float32)

    for ci, sf in enumerate(starts):
        rng = list(range(sf, sf + T))
        rng = [min(r, maxf) for r in rng]  # clamp tail
        clip = get_clip(vr, rng)                                # (T,352,640,3)
        x = torch.from_numpy(clip).to(device).half()
        x = x.reshape(1, T, 352, 640, 3).permute(0, 4, 1, 2, 3)  # (1,3,T,352,640)
        rois, masks = get_roi_masks(work, rng)
        r = torch.from_numpy(rois[None]).float().to(device)      # (1,26,T,5)
        mk = torch.from_numpy(masks[None]).float().to(device)    # (1,26,T)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model([x, r, mk])                             # (1,9,26,T)
        all_logits[:, :, ci * T:(ci + 1) * T] = pred[0].float().cpu().numpy()

    del vr  # release decord's full-length reader before opening cv2 for render (host-RAM)
    all_logits = all_logits[:, :, : (maxf - minf + 1)]
    events = nms_decode(all_logits, args.conf, args.nms, minf)
    events.sort(key=lambda e: -e[3])

    # ball-distance for each event's acting player (px foot-point -> ball), normalised by width
    foot_at = {(int(r[PID]), int(r[FRAME])): (r[RX] + r[RW] / 2.0, r[RY] + r[RH]) for r in work}

    def ball_dist(fr, tid):
        b = ball_xy.get(fr)
        p = foot_at.get((tid, fr))
        if b is None or p is None:
            return None
        return float(np.hypot(p[0] - b[0], p[1] - b[1]) / Wm)

    # Gentle gate: tag near/off-ball, drop ONLY far + weak events (strong off-ball survive).
    kept, meta, n_near, n_gated = [], {}, 0, 0
    for fr, m, c, sc in events:
        tid = slot_to_track.get(m)
        d = ball_dist(fr, tid) if tid is not None else None
        near = d is not None and d <= args.ball_radius
        if args.ball_gate == "soft" and d is not None and d > args.ball_far and sc < args.ball_weak:
            n_gated += 1
            continue
        kept.append((fr, m, c, sc))
        meta[(fr, m)] = (d, near)
        n_near += int(near)
    events = kept

    # map events -> per-(track,frame) overlay dict, carrying the near-ball flag
    ev_by_frame = {}
    for fr, m, c, sc in events:
        tid = slot_to_track.get(m)
        if tid is None:
            continue
        ev_by_frame.setdefault(fr, []).append((tid, c, sc, meta[(fr, m)][1]))

    counts = {NAMES[c]: 0 for c in range(1, 9)}
    for _, _, c, _ in events:
        counts[NAMES[c]] += 1
    summary = {
        "game_key": args.game_key, "video": video, "window_frames": [minf, maxf],
        "window_s": [round(minf / fps, 1), round(maxf / fps, 1)], "fps": fps,
        "ball_detected_frames": manifest.get("ball_detected_frames", len(ball_xy)),
        "ball_gate": args.ball_gate, "events_gated_far_weak": n_gated,
        "n_events": len(events), "events_near_ball": n_near, "events_per_class": counts,
        "team_names": tname,
        "top_events": [{"frame": fr, "t_s": round(fr / fps, 1), "track": slot_to_track.get(m),
                        "team": ("team_a" if m < 13 else "team_b"), "class": NAMES[c],
                        "score": round(sc, 3),
                        "ball_dist": round(meta[(fr, m)][0], 3) if meta[(fr, m)][0] is not None else None,
                        "near_ball": meta[(fr, m)][1]} for fr, m, c, sc in events[:20]],
    }
    (out / "predictions.json").write_text(json.dumps(summary, indent=2))
    print(f"[infer] {len(events)} events ({n_near} near-ball, {n_gated} gated far+weak)  {counts}")

    _render(video, work, ev_by_frame, events, minf, maxf, fps, out, args.max_keyframes, tname, ball_xy)
    return 0


def _render(video, work, ev_by_frame, events, minf, maxf, fps, out, max_kf, tname, ball_xy=None):
    ball_xy = ball_xy or {}
    cap = cv2.VideoCapture(video)
    W = int(cap.get(3)); H = int(cap.get(4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, minf)
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    team_col = {0: (246, 130, 56), 1: (68, 68, 239)}  # BGR: team0 blue, team1 red
    hold = 12  # frames to keep an action label up

    kf_saved, kf_frames = 0, {fr for fr, _, _, _ in events[:max_kf]}
    for t in range(minf, maxf + 1):
        ok, frame = cap.read()
        if not ok:
            break
        rows = work[(work[:, FRAME] == t) & (~np.isnan(work[:, RX]))]
        # active labels within hold window (carry near-ball flag)
        active = {}
        for f2 in range(max(minf, t - hold), t + 1):
            for tid, c, sc, near in ev_by_frame.get(f2, []):
                active[tid] = (c, sc, near)
        for r in rows:
            x1, y1 = int(r[RX]), int(r[RY])
            x2, y2 = int(r[RX] + r[RW]), int(r[RY] + r[RH])
            tid = int(r[PID]); lr = int(r[LR])
            col = team_col.get(lr, (200, 200, 200))
            lab = f"t{tid}"
            if tid in active:
                c, sc, near = active[tid]
                col = ACT_COLOR.get(c, col)
                lab = f"{NAMES[c].upper()} {sc:.2f}" + ("" if near else " off-ball")
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 4)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, lab, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 2, cv2.LINE_AA)
        if t in ball_xy:
            bx, by = int(ball_xy[t][0]), int(ball_xy[t][1])
            cv2.circle(frame, (bx, by), 9, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(frame, f"TAAD on OUR footage  f{t}  {t/fps:.1f}s  teams:{tname}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        if t in kf_frames:
            cv2.imwrite(str(out / "keyframes" / f"kf_{kf_saved:02d}_f{t}.jpg"), frame)
            kf_saved += 1
    cap.release(); writer.release()
    print(f"[render] wrote annotated.mp4 + {kf_saved} keyframes to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
