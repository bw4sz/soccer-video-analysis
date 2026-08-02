"""Precision against yield, so a default can be picked on evidence.

Two numbers for the same setting, side by side:

* **precision** — of the links it makes, how many join a lane to its own true
  continuation (`slurm/eval_track_linking.py`'s split-lane ground truth).
* **yield** — what it actually buys downstream: lanes named after identity is
  propagated along chains, and on-ball spans for the two players we have been
  cutting reels for.

A setting that links everything at 80% precision is worse than it looks, because
a wrong link doesn't just add a bad clip — it welds another player's whole lane
onto a named chain and inherits the name.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from soccer_vision.events.on_ball import select_on_ball_spans
from soccer_vision.identify.resolve import tracks_for
from soccer_vision.profiles.loader import load_profile
from soccer_vision.tracking.link import (LinkConfig, apply_links, link_tracks,
                                         propagate_names)

ROOT = Path("/orange/ewhite/b.weinstein/soccer-video-analysis")
ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "runs/saints-u14g-full"))
ap.add_argument("--profile", default=str(ROOT / "examples/profiles/saints-u14g.yaml"))
ap.add_argument("--players", nargs="+", default=["Morgan", "Mo"])
args = ap.parse_args()

run = Path(args.run)
doc = json.loads((run / "tracks.json").read_text())
ball = json.loads((run / "ball_track.json").read_text())
jerseys = json.loads((run / "jerseys.json").read_text())
profile = load_profile(args.profile)
tracks = doc["tracks"]
fps, interval = float(doc["fps"]), int(doc["sample_interval"])

# --- ground-truth split set, rebuilt here so both halves share one script ---
eligible = sorted((t for t, v in tracks.items() if len(v) >= 20),
                  key=lambda t: -len(tracks[t]))[:400]


def split_doc(gap_samples=3):
    out, kits, truth = {}, dict(doc.get("teams", {})), {}
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


def precision_at(cfg):
    """Mean precision over 0.2 / 0.6 / 1.0 s simulated dropouts."""
    tot_c = tot_w = 0
    for g in (1, 3, 5):
        synth, truth = split_doc(g)
        res = link_tracks(synth, cfg)
        made = {l.a: l.b for l in res.links}
        for h, t in truth.items():
            got = made.get(h)
            if got == t:
                tot_c += 1
            elif got is not None:
                tot_w += 1
    return tot_c / max(tot_c + tot_w, 1), tot_c, tot_w


def player_spans(tracks_doc, jerseys_doc, who, result=None):
    """Lanes / track-frames / spans for one player in a (possibly linked) doc.

    After linking, a chain lives under its root id only, so a named member lane
    no longer exists as a key. Selection has to be mapped through the parent map
    or the player's own lanes look like they vanished.
    """
    tids = {str(i) for i in tracks_for(jerseys_doc, number=None, name=who,
                                       profile=profile)}
    if result is not None:
        from soccer_vision.tracking.link import _find
        parent = dict(result.parent)
        tids = {_find(parent, t) for t in tids if t in parent}
    tids &= set(tracks_doc["tracks"])
    tf = sum(len(tracks_doc["tracks"][t]) for t in tids)
    spans = select_on_ball_spans(ball, tracks_doc, {int(t) for t in tids})
    return len(tids), tf, len(spans)


base_named = sum(1 for r in jerseys["tracks"].values() if r.get("name"))
print(f"baseline: {len(tracks):,} lanes, {base_named:,} named")
for who in args.players:
    n, tf, sp = player_spans(doc, jerseys, who)
    print(f"  {who:<7} {n:>4} lanes  {tf:>5} t-frames ({tf / 5:>5.0f}s)  {sp:>3} spans")

print(f"\n{'gap':>5} {'px':>5} {'precision':>10} {'chains':>8} {'links':>7} "
      f"{'named':>7} {'conflict':>9}  " +
      "  ".join(f"{w + ' spans':>12}" for w in args.players))
for gap_s, px in [(0.6, 100), (1.0, 100), (1.0, 150), (1.5, 150),
                  (2.0, 150), (2.0, 250), (3.0, 250)]:
    cfg = LinkConfig(max_gap_s=gap_s, max_dist_px=px, use_motion=True,
                     bidirectional=True, global_assignment=False)
    prec, c, w = precision_at(cfg)
    res = link_tracks(doc, cfg)
    linked_doc = apply_links(doc, res, interpolate=True)
    j2, jstats = propagate_names(jerseys, res)
    cells = []
    for who in args.players:
        n, tf, sp = player_spans(linked_doc, j2, who, result=res)
        cells.append(f"{sp:>3} ({tf / 5:>4.0f}s)")
    print(f"{gap_s:>5.1f} {px:>5} {100 * prec:>9.0f}% {res.stats['chains']:>8,} "
          f"{res.stats['links']:>7,} {jstats['named_after']:>7,} "
          f"{jstats['conflicts']:>9,}  " + "  ".join(f"{c:>12}" for c in cells))
