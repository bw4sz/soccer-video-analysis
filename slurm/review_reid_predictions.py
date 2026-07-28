"""Render what the gallery actually saw and what it guessed.

Two views, because they answer different questions:

  frame_*.jpg   — the labelled boxes on the full frame, captioned truth -> guess.
                  Shows the *scene*: how big a player is, how alike they look.
  matches.jpg   — each held-out crop beside the three gallery exemplars it scored
                  highest. Shows *why* a guess was made, which is the only way to
                  tell "wrong because blurry" from "wrong because identical".

Everything is leave-one-frame-out: the gallery for a frame is built from every
other frame, so nothing here is a crop matching itself.
"""
from collections import defaultdict
from pathlib import Path
import json

import cv2
import numpy as np

from soccer_vision.cli.enroll import named_boxes
from soccer_vision.identify.enroll import boxes_from_label_studio
from soccer_vision.identify.gallery import build_gallery, match_track
from soccer_vision.identify.reid import crop_player
from soccer_vision.profiles.loader import load_profile

ROOT = Path("/orange/ewhite/b.weinstein/soccer-video-analysis")
SCRATCH = Path("/tmp/claude-4736/-orange-ewhite-b-weinstein-soccer-video-analysis/46a265f2-4403-4f29-a0b2-8f2e504f6b15/scratchpad")
OUT = ROOT / "runs/u14g_label_frames/reid_review"
OUT.mkdir(parents=True, exist_ok=True)

GREEN, RED, AMBER, GREY = (80, 220, 80), (60, 60, 240), (0, 200, 255), (170, 170, 170)

export = json.loads((ROOT / "runs/u14g_label_frames/annotations.json").read_text())
profile = load_profile(ROOT / "examples/profiles/saints-u14g.yaml")
named, _ = named_boxes(boxes_from_label_studio(export), profile)

by_frame = defaultdict(list)
for f, b, n in named:
    by_frame[int(f)].append((b, n))

# Replay the exact order the cached embeddings were written in.
d = np.load(SCRATCH / "u14g_exemplars.npz", allow_pickle=True)
emb, e_frame, e_name = d["emb"], d["frame"], d["name"]
boxes = []
for frame_no in sorted(by_frame):
    for bbox, name in by_frame[frame_no]:
        img_ok = True   # every box cleared crop_player, verified earlier
        if img_ok:
            boxes.append((frame_no, bbox, name))
assert len(boxes) == len(emb), (len(boxes), len(emb))
assert all(b[2] == n for b, n in zip(boxes, e_name))

def short(name):
    return name.split()[0]

rows = []          # (frame, idx, truth, pred, sim, margin, top3 idxs)
for held in sorted(set(e_frame)):
    tr = e_frame != held
    gal_idx = np.where(tr)[0]
    g = build_gallery(emb[tr], list(e_name[tr]))
    for i in np.where(e_frame == held)[0]:
        truth = e_name[i]
        m = match_track(emb[i:i + 1], g)
        sims = emb[tr] @ emb[i]
        top3 = gal_idx[np.argsort(-sims)[:3]]
        rows.append((held, i, truth, m.name, m.similarity, m.margin, top3, sims))

# --- view 1: the boxes on the frame ----------------------------------------
for held in sorted(set(e_frame)):
    frame = cv2.imread(str(ROOT / f"runs/u14g_label_frames/frames/{held:06d}.jpg"))
    for (f, i, truth, pred, sim, mar, _t3, _s) in rows:
        if f != held:
            continue
        x1, y1, x2, y2 = (int(v) for v in boxes[i][1])
        colour = GREY if pred is None else (GREEN if pred == truth else RED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cap = short(truth) if pred is None else f"{short(truth)}>{short(pred)}"
        cv2.putText(frame, cap, (x1 - 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, cap, (x1 - 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 1, cv2.LINE_AA)
    key = "green = correct   red = wrong   grey = abstained (what it does by default)"
    cv2.putText(frame, f"frame {held}   {key}", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"frame {held}   {key}", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"frame_{held:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

# --- view 2: query crop vs the exemplars it scored highest ------------------
CROP_H, LABEL_W, PAD = 190, 300, 8

def tile(idx, height=CROP_H):
    f, bbox, _ = boxes[idx]
    img = cv2.imread(str(ROOT / f"runs/u14g_label_frames/frames/{f:06d}.jpg"))
    c = crop_player(img, bbox)
    scale = height / c.shape[0]
    return cv2.resize(c, (max(1, int(c.shape[1] * scale)), height),
                      interpolation=cv2.INTER_NEAREST)

def caption(strip, text, colour=(235, 235, 235)):
    bar = np.zeros((26, strip.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text[:26], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                colour, 1, cv2.LINE_AA)
    return np.vstack([strip, bar])

strips = []
for (f, i, truth, pred, sim, mar, top3, sims) in rows:
    colour = GREY if pred is None else (GREEN if pred == truth else RED)
    label = np.zeros((CROP_H + 26, LABEL_W, 3), dtype=np.uint8)
    verdict = "ABSTAINED" if pred is None else ("CORRECT" if pred == truth else "WRONG")
    lines = [f"frame {f}", f"truth:  {short(truth)}",
             f"guess:  {short(pred) if pred else '-'}",
             f"sim {sim:.3f}  margin {mar:.3f}", verdict]
    for k, line in enumerate(lines):
        c = colour if k >= 3 else (235, 235, 235)
        cv2.putText(label, line, (10, 34 + k * 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, c, 1, cv2.LINE_AA)
    q = caption(cv2.copyMakeBorder(tile(i), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=colour),
                "QUERY (held out)")
    cells = [label, q, np.zeros((q.shape[0], PAD * 3, 3), dtype=np.uint8)]
    for rank, j in enumerate(top3):
        hit = (e_name[j] == truth)
        t = cv2.copyMakeBorder(tile(j, CROP_H - 6), 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                               value=GREEN if hit else (90, 90, 90))
        cells.append(caption(t, f"{rank+1}. {short(e_name[j])} {sims[np.where(np.where(e_frame != f)[0] == j)[0][0]]:.2f}",
                             (235, 235, 235) if not hit else GREEN))
        cells.append(np.zeros((q.shape[0], PAD, 3), dtype=np.uint8))
    h = max(c.shape[0] for c in cells)
    cells = [np.pad(c, ((0, h - c.shape[0]), (0, 0), (0, 0))) for c in cells]
    strips.append(np.hstack(cells))

w = max(s.shape[1] for s in strips)
strips = [np.pad(s, ((0, 6), (0, w - s.shape[1]), (0, 0))) for s in strips]
per_sheet = 10
for k in range(0, len(strips), per_sheet):
    sheet = np.vstack(strips[k:k + per_sheet])
    cv2.imwrite(str(OUT / f"matches_{k // per_sheet + 1:02d}.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

print(f"{len(rows)} held-out crops rendered -> {OUT}")
print(f"  frames:  {len(set(e_frame))} x frame_*.jpg")
print(f"  matches: {(len(strips) + per_sheet - 1) // per_sheet} x matches_*.jpg")
corr = sum(1 for r in rows if r[3] == r[2])
wrong = sum(1 for r in rows if r[3] is not None and r[3] != r[2])
print(f"  per-crop at defaults: {corr} correct, {wrong} wrong, {len(rows)-corr-wrong} abstained")
