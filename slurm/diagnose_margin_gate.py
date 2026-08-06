"""Why a player's reel is empty: rank-1 without a name, and the margin that ate it.

Morgan's and Morrighan's reels came back at 0.4 and 0.2 minutes and the obvious
reading — "re-id can't find them" — is wrong. Over the eligible lanes in this
run's saved `query_embeddings.npz`, **Morgan is rank-1 more often than any other
real player and is named on none of them**. Every one of those abstentions is a
margin abstention, not a similarity one.

That distinction matters because the two have different fixes. A similarity floor
says the gallery doesn't recognise her; a margin floor says it recognises her but
not *decisively enough*, on a scale where `CLAUDE.md` already records that
absolute similarity is nearly inert because re-id cosines bunch high. The median
margin here is 0.017 against a gate of 0.05 — the gate sits above the 75th
percentile of the distribution it is filtering.

Runs entirely off saved artefacts (no GPU, no crops, no video):

    python slurm/diagnose_margin_gate.py --run runs/saints-u14g-full-30fps \
        --gallery galleries/saints-u14g.fullmatch-neg64.npz

`--compare-gallery` re-scores the *same* lanes under a second gallery, which is
the only way to isolate one gallery change from the gate changes usually shipped
alongside it — the mistake that made job 38746007 unreadable.

**Two things this cannot tell you.** The lanes here are whichever ones the
embedding dump sampled, so counts are a sample of the run, not the run. And there
are no labels, so every number below is recall-shaped: a name recovered by
loosening the gate is not necessarily a *correct* name. Pair it with
`slurm/validate_reid_frames.py` before adopting a threshold.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from soccer_vision.identify.gallery import NEGATIVE_LABEL, load_gallery

MARGINS = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.0]


def lane_scores(run: Path, gallery: dict, jerseys: dict) -> tuple[np.ndarray, np.ndarray]:
    """Score every *eligible* lane the way `match_track` does, one row per lane."""
    with np.load(run / "query_embeddings.npz", allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
        emb = z["emb"].astype(np.float32)

    G = gallery["emb"] / np.linalg.norm(gallery["emb"], axis=1, keepdims=True)
    lab = np.asarray(gallery["label"])
    n_players = len(gallery["names"])

    off, rows, spans = 0, [], []
    for m in meta:
        E = emb[off:off + m["n"]]
        off += m["n"]
        v = jerseys.get(str(m["tid"]))
        # kit and length gates run before any model; keep 'not_ours' so the
        # negative class is measured rather than assumed.
        if v is None or v.get("excluded") in ("kit", "short"):
            continue
        X = E / np.linalg.norm(E, axis=1, keepdims=True)
        s = X @ G.T
        rows.append([np.sort(s[:, lab == p], axis=1)[:, -3:].mean() for p in range(n_players)])
        spans.append(m["span"])
    return np.asarray(rows), np.asarray(spans)


def sweep(S: np.ndarray, spans: np.ndarray, names: list[str], players: list[str],
          min_similarity: float, tag: str) -> None:
    order = np.argsort(-S, axis=1)
    idx = np.arange(len(S))
    best, runner = S[idx, order[:, 0]], S[idx, order[:, 1]]
    margin = best - runner
    win = np.array([names[i] for i in order[:, 0]])

    print(f"\n=== {tag} ===")
    print(f"  {len(S)} eligible lanes | margin p10 {np.percentile(margin, 10):.3f} "
          f"med {np.median(margin):.3f} p90 {np.percentile(margin, 90):.3f}")
    print(f"  abstaining on similarity alone: {(best < min_similarity).sum()}")

    hdr = "  margin  named  rejected " + "".join(f"{p.split()[0]:>11s}" for p in players)
    print(hdr)
    for mm in MARGINS:
        ok = (best >= min_similarity) & (margin >= mm)
        c = collections.Counter(win[ok])
        named = sum(v for k, v in c.items() if k != NEGATIVE_LABEL)
        cells = "".join(f"{c.get(p, 0):11d}" for p in players)
        print(f"  {mm:6.3f} {named:6d} {c.get(NEGATIVE_LABEL, 0):9d}" + cells)

    print("  rank-1 count (no gate at all) — how often the gallery puts each first:")
    c1 = collections.Counter(win)
    for nm, k in c1.most_common():
        secs = spans[win == nm].sum()
        print(f"    {nm:24s} {k:5d} lanes  {secs:8.0f} lane-seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--gallery", required=True, type=Path)
    ap.add_argument("--compare-gallery", type=Path, default=None,
                    help="second gallery scored over the SAME lanes and gates")
    ap.add_argument("--jerseys", default="jerseys.json")
    ap.add_argument("--min-similarity", type=float, default=0.5)
    ap.add_argument("--players", nargs="*",
                    default=["Morgan Lobey", "Morrighan Wright", "Catherine Conroy",
                             "Gia Olson"],
                    help="columns to break the sweep out by")
    args = ap.parse_args()

    jerseys = json.load(open(args.run / args.jerseys))["tracks"]
    for path in [args.gallery] + ([args.compare_gallery] if args.compare_gallery else []):
        g = load_gallery(path)
        S, spans = lane_scores(args.run, g, jerseys)
        sweep(S, spans, g["names"], args.players, args.min_similarity, path.name)


if __name__ == "__main__":
    main()
