"""Propose candidate *spectator* lanes for hand-verification, then enrol them.

`eval_negative_class.py` came back inconclusive for a reason worth recording: the
27 lanes an annotator had already marked ``not ours`` all stand **on the pitch**
(median foot-y 493 of 1080, none above 0.40H). Tracklet windows are rendered from
lanes in play with ``--team black``, so ``not ours`` there means "not on our
team" — an opponent or a referee — and never "not on the field". A negative class
built from them is an on-pitch class, and measurably wins on on-pitch lanes more
often than off-pitch ones.

So the negatives have to be harvested where the spectators are. This proposes
candidates by position and **renders them for a human to confirm**, because
position is exactly the signal under suspicion: our own far-side players stand
above 0.40H whenever the camera pans wide. Enrolling on the heuristic alone would
bank real players as negatives and quietly poison the gallery against them.

    # 1. propose, and render sheets to look at
    python slurm/harvest_offpitch_negatives.py --run runs/saints-u14g-full-30fps

    # 2. after reading the sheets, enrol only the lanes you confirmed
    python slurm/harvest_offpitch_negatives.py --run runs/saints-u14g-full-30fps \
        --confirm 1234,5678,... --out galleries/negatives-u14g.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PAD = 130
TILE = (240, 260)
COLS = 6


def propose(run: Path, *, max_foot_frac: float, min_span_s: float,
            n_candidates: int) -> list[dict]:
    """Long-lived lanes sitting high in the frame, spread across the match."""
    from soccer_vision.clips.halo import load_track_boxes

    tracks = json.loads((run / "tracks.json").read_text())
    fps = tracks["fps"]
    samples = load_track_boxes(run / "tracks.json")

    rows = []
    for tid, s in samples.items():
        if len(s) < 2:
            continue
        span = (s[-1][0] - s[0][0]) / fps
        if span < min_span_s:
            continue
        foot = float(np.median([b[3] for _, b in s]))
        if foot >= max_foot_frac * 1080:
            continue
        rows.append(dict(tid=int(tid), foot=foot, span=span,
                         first=int(s[0][0]), last=int(s[-1][0])))

    # spread over the match so one crowded passage can't supply every negative
    rows.sort(key=lambda r: r["first"])
    if len(rows) > n_candidates:
        idx = np.linspace(0, len(rows) - 1, n_candidates).astype(int)
        rows = [rows[i] for i in idx]
    return rows


def render_sheets(run: Path, rows: list[dict], out_dir: Path, per_sheet: int) -> None:
    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.io.video import VideoReader

    samples = load_track_boxes(run / "tracks.json")
    by_frame: dict[int, list] = {}
    for r in rows:
        got = samples.get(r["tid"]) or []
        if got:
            f, b = got[len(got) // 2]
            by_frame.setdefault(int(f), []).append((r["tid"], b))

    tiles = {}
    reader = VideoReader(run / "broadcast_proxy.mp4")
    try:
        for frame_no, frame in reader.read_frames(sorted(by_frame)):
            fh, fw = frame.shape[:2]
            for tid, (x1, y1, x2, y2) in by_frame[frame_no]:
                cx1, cy1 = max(0, int(x1) - PAD), max(0, int(y1) - PAD)
                cx2, cy2 = min(fw, int(x2) + PAD), min(fh, int(y2) + PAD)
                sub = frame[cy1:cy2, cx1:cx2].copy()
                if sub.size:
                    cv2.rectangle(sub, (int(x1) - cx1, int(y1) - cy1),
                                  (int(x2) - cx1, int(y2) - cy1), (0, 0, 255), 2)
                    tiles[tid] = cv2.resize(sub, TILE)
    finally:
        reader.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r["tid"] in tiles]
    w, h = TILE
    lh = 20
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        nr = (len(chunk) + COLS - 1) // COLS
        sheet = np.full((nr * (h + lh), COLS * w, 3), 30, np.uint8)
        for i, r in enumerate(chunk):
            rr, cc = divmod(i, COLS)
            y0, x0 = rr * (h + lh), cc * w
            sheet[y0:y0 + h, x0:x0 + w] = tiles[r["tid"]]
            cv2.putText(sheet, f"{r['tid']} y{r['foot']:.0f} {r['span']:.0f}s",
                        (x0 + 4, y0 + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (160, 220, 255), 1)
        path = out_dir / f"candidates_{s // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", path)

    (out_dir / "candidates.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out_dir / 'candidates.json'} ({len(rows)} lanes)")


def enrol(run: Path, tids: list[int], out: Path, max_per_lane: int,
          device: str | None) -> None:
    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.identify.reid import ReIDEmbedder, crop_player
    from soccer_vision.io.video import VideoReader

    samples = load_track_boxes(run / "tracks.json")
    by_frame: dict[int, list] = {}
    for tid in tids:
        got = samples.get(tid) or []
        if not got:
            print(f"  lane {tid}: no boxes, skipped")
            continue
        step = max(1, len(got) // max_per_lane)
        for f, b in got[::step][:max_per_lane]:
            by_frame.setdefault(int(f), []).append(b)

    crops = []
    reader = VideoReader(run / "broadcast_proxy.mp4")
    try:
        for frame_no, frame in reader.read_frames(sorted(by_frame)):
            for bbox in by_frame[frame_no]:
                c = crop_player(frame, bbox)
                if c is not None:
                    crops.append(c)
    finally:
        reader.close()

    print(f"embedding {len(crops)} crops from {len(tids)} confirmed lanes")
    embedder = ReIDEmbedder.from_pretrained(device=device)
    emb = embedder.embed(crops)
    np.savez_compressed(out, emb=emb, tids=np.asarray(tids))
    print(f"wrote {out} ({emb.shape})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--max-foot-frac", type=float, default=0.38,
                    help="propose lanes whose median foot-y sits above this "
                         "fraction of frame height")
    ap.add_argument("--min-span-s", type=float, default=3.0)
    ap.add_argument("--n-candidates", type=int, default=72)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--confirm", default=None,
                    help="comma-separated lane ids confirmed as off-pitch")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-per-lane", type=int, default=20)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.run / "offpitch_candidates")

    if args.confirm:
        tids = [int(t) for t in args.confirm.split(",") if t.strip()]
        enrol(args.run, tids, args.out or (out_dir / "negatives.npz"),
              args.max_per_lane, args.device)
        return

    rows = propose(args.run, max_foot_frac=args.max_foot_frac,
                   min_span_s=args.min_span_s, n_candidates=args.n_candidates)
    print(f"{len(rows)} candidate lanes above {args.max_foot_frac:.2f}H "
          f"lasting >= {args.min_span_s}s")
    render_sheets(args.run, rows, out_dir, args.per_sheet)
    print("\nRead the sheets, then re-run with "
          "--confirm <comma-separated ids of the ones that really are off-pitch>")


if __name__ == "__main__":
    main()
