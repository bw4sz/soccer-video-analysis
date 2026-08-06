"""Contact sheets for the lanes the 'not ours' class took a *name* off.

`sheet_negative_class.py` sheets the sweep's sampled lanes. This one asks the
whole-match question instead: over the full run, which lanes did the baseline
name and the negative class reject, and was each rejection right?

That set is small (19 lanes on `runs/saints-u14g-full-30fps`) and it is the only
place the negative class can have *cost* anything — a rejected lane that was
never named loses nothing.

One row per lane, one column per sampled moment across its life, so the row
reads as motion: a player crosses the pitch, a spectator sits still on the same
patch of touchline. Tiles are context, not crops, for the reason in CLAUDE.md.

    python slurm/sheet_negclass_name_losses.py --run runs/saints-u14g-full-30fps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PAD = 130           # px of context around the box
TILE = (300, 300)   # w, h per moment
COLS = 5            # moments sampled across each lane
ROWS = 5            # lanes per sheet


def _tile(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_w, half_h = (x2 - x1) // 2 + PAD, (y2 - y1) // 2 + PAD
    sx1, sy1 = max(0, cx - half_w), max(0, cy - half_h)
    sx2, sy2 = min(w, cx + half_w), min(h, cy + half_h)
    crop = frame[sy1:sy2, sx1:sx2].copy()
    if crop.size == 0:
        return np.zeros((TILE[1], TILE[0], 3), np.uint8)
    cv2.rectangle(crop, (x1 - sx1, y1 - sy1), (x2 - sx1, y2 - sy1), (0, 200, 255), 2)
    scale = min(TILE[0] / crop.shape[1], TILE[1] / crop.shape[0])
    crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    out = np.zeros((TILE[1], TILE[0], 3), np.uint8)
    oy, ox = (TILE[1] - crop.shape[0]) // 2, (TILE[0] - crop.shape[1]) // 2
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--baseline", default="jerseys.pre-negclass.json")
    ap.add_argument("--jerseys", default="jerseys.json")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    run = args.run
    out_dir = args.out_dir or run / "negclass_name_losses"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = json.load(open(run / args.baseline))["tracks"]
    new = json.load(open(run / args.jerseys))["tracks"]
    td = json.load(open(run / "tracks.json"))
    tracks, fps = td["tracks"], td["fps"]

    lost = [
        (tid, base[tid], new[tid])
        for tid in new
        if new[tid].get("excluded") == "not_ours" and base.get(tid, {}).get("name")
    ]
    lost.sort(key=lambda r: -(r[2].get("span_s") or 0))
    print(f"{len(lost)} lane(s) lost a name to the negative class")

    cap = cv2.VideoCapture(str(run / td["video"]))
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    sheet_rows: list[np.ndarray] = []
    sheet_i = 1
    for tid, b, n in lost:
        samples = tracks[tid]
        idx = np.linspace(0, len(samples) - 1, COLS).round().astype(int)
        row = []
        for i in idx:
            s = samples[int(i)]
            cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame"])
            ok, frame = cap.read()
            tile = _tile(frame, s["bbox"]) if ok else np.zeros((TILE[1], TILE[0], 3), np.uint8)
            t = s["frame"] / fps
            cv2.putText(tile, f"{int(t // 60)}:{t % 60:04.1f}", (6, TILE[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            row.append(tile)
        strip = np.hstack(row)

        foot_y = np.median([s["bbox"][3] for s in samples]) / frame_h
        hdr = np.zeros((34, strip.shape[1], 3), np.uint8)
        cv2.putText(hdr, f"lane {tid} | was {b['name']} (sim {b.get('similarity')}, "
                         f"{b.get('source')}) | span {n.get('span_s')}s | "
                         f"foot-y {foot_y:.2f}H | kit {n.get('kit')}",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        sheet_rows.append(np.vstack([hdr, strip]))

        if len(sheet_rows) == ROWS:
            p = out_dir / f"name_loss_{sheet_i:02d}.jpg"
            cv2.imwrite(str(p), np.vstack(sheet_rows)); print("wrote", p)
            sheet_rows, sheet_i = [], sheet_i + 1

    if sheet_rows:
        p = out_dir / f"name_loss_{sheet_i:02d}.jpg"
        cv2.imwrite(str(p), np.vstack(sheet_rows)); print("wrote", p)
    cap.release()


if __name__ == "__main__":
    main()
