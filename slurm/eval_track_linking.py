"""Does a linker put a lane back together after a simulated dropout?

Name-conflict rate is the only validation available on raw output, and it is
weak: it can only see chains where *two* lanes were independently named, and
re-id names 11% of lanes, so most bad links are invisible to it.

This gets real ground truth instead. Take the lanes long enough to cut in half,
delete a few samples from the middle to simulate exactly the detector dropout
that kills lanes in the first place, and drop the two halves back into the full
match alongside all 17k real lanes as distractors. The correct continuation of a
head is known by construction: its own tail. So a wrong link is *measurable*,
not inferred.

What it cannot see: lanes that die for a reason other than dropout (a player
leaving frame, a genuine ID swap in traffic). The split lanes are drawn from
survivors, which are the easier population. Read the numbers as an upper bound.

    python slurm/eval_track_linking.py [--run RUN] [--min-len 20]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from soccer_vision.tracking.link import LinkConfig, link_tracks

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="/orange/ewhite/b.weinstein/soccer-video-analysis/"
                                 "runs/saints-u14g-full")
ap.add_argument("--min-len", type=int, default=20)
ap.add_argument("--max-splits", type=int, default=400)
args = ap.parse_args()

doc = json.loads((Path(args.run) / "tracks.json").read_text())
tracks, teams = doc["tracks"], doc.get("teams", {})
fps, interval = float(doc["fps"]), int(doc["sample_interval"])

eligible = [t for t, v in tracks.items() if len(v) >= args.min_len]
eligible.sort(key=lambda t: -len(tracks[t]))
eligible = eligible[: args.max_splits]
print(f"{len(tracks):,} lanes; {len(eligible)} long enough to split "
      f"(>={args.min_len} samples = {args.min_len * interval / fps:.1f}s)\n")


def build(gap_samples):
    """Full match with `eligible` lanes cut in two, `gap_samples` deleted."""
    out, kits, truth = {}, dict(teams), {}
    for tid, samples in tracks.items():
        if tid not in eligible:
            out[tid] = samples
            continue
        mid = len(samples) // 2
        head, tail = samples[:mid], samples[mid + gap_samples:]
        if len(head) < 2 or len(tail) < 2:
            out[tid] = samples
            continue
        h, t = f"{tid}_h", f"{tid}_t"
        out[h], out[t] = head, tail
        truth[h] = t
        if tid in kits:
            kits[h] = kits[t] = kits.pop(tid)
    return {**doc, "tracks": out, "teams": kits}, truth


STRATEGIES = {
    "naive (position only, greedy)":
        dict(use_motion=False, bidirectional=False, global_assignment=False),
    "motion, forward only":
        dict(use_motion=True, bidirectional=False, global_assignment=False),
    "motion, bidirectional":
        dict(use_motion=True, bidirectional=True, global_assignment=False),
    "motion, bidir + hungarian":
        dict(use_motion=True, bidirectional=True, global_assignment=True),
}

print(f"{'strategy':<30} {'gap':>5} {'correct':>8} {'wrong':>7} {'missed':>7}  "
      f"{'precision':>9}")
for label, kw in STRATEGIES.items():
    for gap_samples in (1, 3, 5, 10):
        synth, truth = build(gap_samples)
        cfg = LinkConfig(max_gap_s=2.5, max_dist_px=150.0, **kw)
        res = link_tracks(synth, cfg)
        made = {l.a: l.b for l in res.links}
        c = Counter()
        for h, t in truth.items():
            got = made.get(h)
            c["correct" if got == t else ("missed" if got is None else "wrong")] += 1
        linked = c["correct"] + c["wrong"]
        prec = f"{100 * c['correct'] / linked:.0f}%" if linked else "-"
        gap_s = gap_samples * interval / fps
        print(f"{label:<30} {gap_s:>4.1f}s {c['correct']:>8} {c['wrong']:>7} "
              f"{c['missed']:>7}  {prec:>9}")
    print()
