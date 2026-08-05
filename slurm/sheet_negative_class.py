"""Contact sheets for hand-labelling the lanes `eval_negative_class.py` scored.

Foot-y is a *stratifier*, not ground truth: our own far-side players stand above
0.40H whenever the camera pans wide, so a rejection rate measured against the
band is measured against a contaminated label. These sheets put each lane's box
back on its frame with enough surrounding pitch to tell a player from somebody
watching, which is the only way to say whether a rejection was correct.

The tile is context, not a crop, for the reason in CLAUDE.md: an overhead camera
renders a player in ~50x21 px, unnameable in isolation, while on the frame you
have the touchline, neighbours and direction of play to go on.

Run after eval_negative_class.py:
    python slurm/sheet_negative_class.py --run runs/saints-u14g-full-30fps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PAD = 110          # px of context kept around the box
TILE = (220, 300)  # w, h of each tile after scaling
COLS = 6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--eval", type=Path, default=None,
                    help="negative_class_eval.json (default: inside --run)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--group", default="all",
                    choices=["all", "disagree", "off", "on", "rej64_on", "rej64_off",
                             "miss64_off"],
                    help="'rej64_on' = on-pitch lanes the capped negative class "
                         "rejected (its cost); 'miss64_off' = off-pitch lanes it "
                         "let through (its recall gap)")
    args = ap.parse_args()

    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.io.video import VideoReader

    rows = json.loads((args.eval or args.run / "negative_class_eval.json").read_text())
    H = 1080

    def keep(r):
        if args.group == "off":
            return r["foot"] / H < 0.40
        if args.group == "on":
            return r["foot"] / H >= 0.45
        if args.group == "disagree":
            return bool(r["base_name"]) and r["neg@all_name"] != r["base_name"]
        if args.group == "rej64_on":
            return r["foot"] / H >= 0.45 and r["neg@64_name"] == "not ours"
        if args.group == "rej64_off":
            return r["foot"] / H < 0.40 and r["neg@64_name"] == "not ours"
        if args.group == "miss64_off":
            return r["foot"] / H < 0.40 and r["neg@64_name"] != "not ours"
        return True

    rows = [r for r in rows if keep(r)]
    print(f"{len(rows)} lanes in group '{args.group}'")
    if not rows:
        return

    samples = load_track_boxes(args.run / "tracks.json")
    # one representative frame per lane: the middle of its sampled span
    pick = {}
    for r in rows:
        got = samples.get(r["tid"]) or []
        if got:
            f, b = got[len(got) // 2]
            pick[r["tid"]] = (int(f), b)

    by_frame: dict[int, list] = {}
    for tid, (f, b) in pick.items():
        by_frame.setdefault(f, []).append((tid, b))

    tiles: dict[int, np.ndarray] = {}
    reader = VideoReader(args.run / "broadcast_proxy.mp4")
    try:
        for frame_no, frame in reader.read_frames(sorted(by_frame)):
            fh, fw = frame.shape[:2]
            for tid, (x1, y1, x2, y2) in by_frame[frame_no]:
                cx1, cy1 = max(0, int(x1) - PAD), max(0, int(y1) - PAD)
                cx2, cy2 = min(fw, int(x2) + PAD), min(fh, int(y2) + PAD)
                sub = frame[cy1:cy2, cx1:cx2].copy()
                if sub.size == 0:
                    continue
                cv2.rectangle(sub, (int(x1) - cx1, int(y1) - cy1),
                              (int(x2) - cx1, int(y2) - cy1), (0, 0, 255), 2)
                tiles[tid] = cv2.resize(sub, TILE)
    finally:
        reader.close()

    out_dir = args.out_dir or (args.run / "negative_class_sheets")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r["tid"] in tiles]

    w, h = TILE
    label_h = 34
    for s in range(0, len(rows), args.per_sheet):
        chunk = rows[s:s + args.per_sheet]
        n_rows = (len(chunk) + COLS - 1) // COLS
        sheet = np.full((n_rows * (h + label_h), COLS * w, 3), 30, np.uint8)
        for i, r in enumerate(chunk):
            rr, cc = divmod(i, COLS)
            y0, x0 = rr * (h + label_h), cc * w
            sheet[y0:y0 + h, x0:x0 + w] = tiles[r["tid"]]
            base = (r["base_name"] or "-").split()[0]
            neg = (r["neg@all_name"] or "-")
            neg = "NOT OURS" if neg == "not ours" else neg.split()[0]
            cv2.putText(sheet, f"{r['tid']} y{r['foot']:.0f} {r['kit'] or '?'}",
                        (x0 + 4, y0 + h + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (200, 200, 200), 1)
            cv2.putText(sheet, f"{base} {r['base_sim']:.2f} -> {neg}",
                        (x0 + 4, y0 + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (120, 220, 255), 1)
        path = out_dir / f"{args.group}_{s // args.per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("wrote", path)


if __name__ == "__main__":
    main()
