"""Sweep the negative class: which negatives, and how many of them.

`eval_negative_class.py` established that an off-pitch negative class works at
cap 64 and collapses uncapped. Two questions it left open, both answered here on
one cached set of query embeddings:

- **Do mined hard negatives beat randomly-harvested ones?** The class is capped,
  so more negatives never make it bigger — they only change which 64 get drawn.
  Hard negative mining is the standard answer: take the off-pitch lanes the
  current class *failed* to reject and enrol those.
- **Where is the cap's operating point?** 64 was chosen to match a player, not
  because it was measured.

Query embeddings are cached because they are the expensive part (~8k crops off
~4k decoded frames); scoring against a new gallery is milliseconds.

    python slurm/sweep_negative_class.py --run runs/saints-u14g-full-30fps \
        --random galleries/negatives-u14g-offpitch.npz \
        --hard   galleries/negatives-u14g-hard.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NOT_OURS = "not ours"


def load_or_build_queries(run: Path, cache: Path, args) -> tuple[np.ndarray, list]:
    """(embeddings stacked per lane, lane metadata) — cached after the first run."""
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            return z["emb"], json.loads(str(z["meta"]))

    from soccer_vision.identify.reid import ReIDEmbedder, crop_player
    from soccer_vision.io.video import VideoReader
    from eval_negative_class import sample_query_lanes

    lanes, _ = sample_query_lanes(
        run, n_windows=args.n_windows, window_s=args.window_s,
        min_span_s=args.min_span_s, max_crops=args.max_crops)
    by_frame: dict[int, list] = {}
    for lane in lanes.values():
        for f, b in lane["boxes"]:
            by_frame.setdefault(int(f), []).append((lane["tid"], b))

    crops: dict[int, list] = {}
    reader = VideoReader(run / "broadcast_proxy.mp4")
    try:
        for frame_no, frame in reader.read_frames(sorted(by_frame)):
            for tid, bbox in by_frame[frame_no]:
                c = crop_player(frame, bbox)
                if c is not None:
                    crops.setdefault(tid, []).append(c)
    finally:
        reader.close()

    embedder = ReIDEmbedder.from_pretrained(device=args.device)
    embs, meta = [], []
    for tid, cs in crops.items():
        e = embedder.embed(cs)
        meta.append(dict(tid=tid, foot=lanes[tid]["foot"], span=lanes[tid]["span"],
                         start=len(embs and np.concatenate(embs)) if embs else 0,
                         n=len(e)))
        embs.append(e)
    # rebuild offsets cleanly
    off = 0
    for m, e in zip(meta, embs):
        m["start"] = off
        off += len(e)
    emb = np.concatenate(embs)
    np.savez_compressed(cache, emb=emb, meta=json.dumps(meta))
    print(f"cached {emb.shape} query embeddings → {cache}")
    return emb, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--gallery", type=Path,
                    default=Path("galleries/saints-u14g.fullmatch.npz"))
    ap.add_argument("--random", type=Path, required=True)
    ap.add_argument("--hard", type=Path, required=True)
    ap.add_argument("--caps", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--n-windows", type=int, default=10)
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--min-span-s", type=float, default=1.0)
    ap.add_argument("--max-crops", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from soccer_vision.identify.gallery import build_gallery, load_gallery, match_track

    cache = args.run / "query_embeddings.npz"
    emb, meta = load_or_build_queries(args.run, cache, args)

    base = load_gallery(args.gallery)
    base_names = [base["names"][i] for i in base["label"]]

    negs = {}
    for label, path in (("random", args.random), ("hard", args.hard)):
        with np.load(path) as z:
            negs[label] = (np.asarray(z["emb"], np.float32),
                           {int(t) for t in z["tids"]} if "tids" in z else set())
    negs["both"] = (np.concatenate([negs["random"][0], negs["hard"][0]]),
                    negs["random"][1] | negs["hard"][1])

    H = 1080
    print(f"\nnegative pools: "
          + ", ".join(f"{k}={len(v[0])}" for k, v in negs.items()))

    def evaluate(gallery, drop: set):
        off = on = roff = ron = noff = non_ = 0
        for m in meta:
            if m["tid"] in drop:
                continue
            q = emb[m["start"]:m["start"] + m["n"]]
            r = match_track(q, gallery)
            is_off = m["foot"] / H < 0.40
            is_on = m["foot"] / H >= 0.45
            if not (is_off or is_on):
                continue
            if is_off:
                off += 1
                roff += r.name == NOT_OURS
                noff += bool(r.name) and r.name != NOT_OURS
            else:
                on += 1
                ron += r.name == NOT_OURS
                non_ += bool(r.name) and r.name != NOT_OURS
        return off, on, roff, ron, noff, non_

    all_drop = negs["both"][1]
    off, on, _, _, noff, non_ = evaluate(base, all_drop)
    print(f"\nevaluated on {off} off-pitch and {on} on-pitch lanes "
          f"({len(all_drop & {m['tid'] for m in meta})} enrolled lanes held out)")
    print(f"\n{'pool':>8} {'cap':>5} {'reject off':>12} {'reject on':>11} "
          f"{'names off':>10} {'names on':>9}")
    print(f"{'baseline':>8} {'-':>5} {'-':>12} {'-':>11} {noff:>10} {non_:>9}")

    for pool in ("random", "hard", "both"):
        neg_emb, _ = negs[pool]
        for cap in args.caps:
            g = build_gallery(np.concatenate([base["emb"], neg_emb]),
                              base_names + [NOT_OURS] * len(neg_emb),
                              max_per_player=max(cap, 64))
            # keep players at 64, negatives at `cap`
            if cap != 64:
                ni = g["names"].index(NOT_OURS)
                rows = np.flatnonzero(g["label"] == ni)
                keep = np.sort(np.random.default_rng(0).choice(
                    rows, min(cap, len(rows)), replace=False))
                sel = np.concatenate([np.flatnonzero(g["label"] != ni), keep])
                sel.sort()
                g = {"names": g["names"], "emb": g["emb"][sel], "label": g["label"][sel]}
            o, n, ro, rn, no, nn = evaluate(g, all_drop)
            print(f"{pool:>8} {cap:>5} {ro:>6}/{o:<5} {rn:>5}/{n:<5} "
                  f"{no:>10} {nn:>9}")


if __name__ == "__main__":
    main()
