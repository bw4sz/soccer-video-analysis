"""Score track linking against hand-labelled lanes, and say where the gate belongs.

Every linking number we have so far is self-reported. `eval_track_linking.py`
splits real lanes artificially and measures whether the linker puts them back —
which cannot see the failure that matters, two *different* players joined. The
`conflicts` count in `propagate_names` is closer to honest but only fires when
both ends of a chain were independently named, and re-id names 4% of lanes, so it
undercounts by a lot.

This uses real labels instead. `enroll --dump-tracklets --at S --all-lanes` rings
every lane in one contiguous stretch; an annotator names them; two lanes carrying
the same name are the same player. From that:

**Precision** — of the links accepted between two labelled lanes, how many joined
one player to themselves. Wrong links are the expensive failure: one puts another
child into a parent's reel.

**Recall** — of the true continuations (a player's lanes, consecutive in time,
not overlapping), how many ended up in the same chain. Chain membership, not the
direct link, because a chain that reaches the same player through an unlabelled
intermediate lane has still propagated the name, which is the whole point.

Both are reported per gap bucket, because that is the shape of the decision: the
gate is a gap threshold, and the question is where precision falls off.

Usage:
  python slurm/eval_link_ground_truth.py --run runs/<match> \
      --tracklets runs/<match>/tracklets --export annotations.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from soccer_vision.annotate.tracklets import named_lanes_from_export
from soccer_vision.tracking.link import LinkConfig, link_tracks, _find

# The sweep. Each is (max_gap_s, max_dist_px); the current shipped default is
# 2.0/150, and the 5 fps full-match sweep suggested far looser. That suggestion
# was made without ground truth, which is exactly what this script supplies.
GATES = [(1.0, 100), (2.0, 150), (3.0, 200), (5.0, 250), (8.0, 300), (12.0, 400)]

GAP_BUCKETS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9)]


def bucket(gap_s: float) -> str:
    for lo, hi in GAP_BUCKETS:
        if lo <= gap_s < hi:
            return f"{lo:g}-{hi:g}s" if hi < 1e9 else f">{lo:g}s"
    return "?"


def true_continuations(truth: dict[str, str], tracks: dict, fps: float):
    """Consecutive non-overlapping lane pairs of one player — what linking owes us.

    Only *adjacent* pairs count. A player with lanes A, B, C should be joined
    A-B and B-C; scoring A-C as a separate obligation would double-count the one
    thing that has to happen and make recall look worse than it is.
    """
    by_player: dict[str, list[str]] = defaultdict(list)
    for tid, name in truth.items():
        by_player[name].append(tid)

    pairs = []
    for name, tids in by_player.items():
        tids.sort(key=lambda t: tracks[t][0]["frame"])
        for a, b in zip(tids, tids[1:]):
            a_end, b_start = tracks[a][-1]["frame"], tracks[b][0]["frame"]
            if b_start <= a_end:
                continue          # overlapping in time: not a continuation
            pairs.append((a, b, (b_start - a_end) / fps, name))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--tracklets", type=Path,
                    help="the --dump-tracklets dir (default: <run>/tracklets)")
    ap.add_argument("--export", default="annotations.json",
                    help="Label Studio export, relative to --tracklets")
    ap.add_argument("--tracks", default="tracks.json")
    args = ap.parse_args()

    tl_dir = args.tracklets or (args.run / "tracklets")
    manifest = json.loads((tl_dir / "tracklets.json").read_text())
    export = json.loads((tl_dir / args.export).read_text())
    doc = json.loads((args.run / args.tracks).read_text())
    tracks, fps = doc["tracks"], float(doc["fps"])

    named, summary = named_lanes_from_export(export, manifest)
    if summary["unmatched_tasks"]:
        print(f"WARNING: {summary['unmatched_tasks']} tasks matched no window — "
              f"the manifest and the export disagree; results below are partial.")

    # One name per lane. A lane labelled twice with different names (it appeared
    # in two windows) is dropped rather than arbitrated: the annotator saw the
    # same footage twice and disagreed with themselves, so neither label is
    # evidence.
    votes: dict[str, set[str]] = defaultdict(set)
    for lane, name in named:
        votes[str(lane["track_id"])].add(name)
    truth = {t: next(iter(v)) for t, v in votes.items()
             if len(v) == 1 and t in tracks}
    disputed = sum(1 for v in votes.values() if len(v) > 1)

    print(f"labelled lanes: {len(truth)} usable"
          f"{f', {disputed} self-contradictory (dropped)' if disputed else ''}"
          f", {summary['lanes_skipped']} not-ours/unsure")
    print(f"players: {len(set(truth.values()))}  "
          f"({', '.join(f'{n}x{c}' for n, c in Counter(truth.values()).most_common(6))})")

    truth_pairs = true_continuations(truth, tracks, fps)
    print(f"true continuations to recover: {len(truth_pairs)}")
    if not truth_pairs:
        print("Nothing to score — label more lanes, or a longer stretch.")
        return 1
    print("  by gap: " + "  ".join(
        f"{b}:{c}" for b, c in
        sorted(Counter(bucket(g) for _, _, g, _ in truth_pairs).items())))

    print(f"\n{'gap_s':>6}{'dist':>6}{'links':>7}{'judgeable':>10}{'correct':>9}"
          f"{'precision':>11}{'recall':>9}")
    per_gate_detail = {}
    for gap, dist in GATES:
        res = link_tracks(doc, LinkConfig(max_gap_s=gap, max_dist_px=dist))

        # Precision: only links whose BOTH ends are labelled can be judged. The
        # rest are not evidence either way and must not be counted as correct.
        judgeable = [l for l in res.links if l.a in truth and l.b in truth]
        correct = [l for l in judgeable if truth[l.a] == truth[l.b]]

        # Recall over chains, not links — an indirect join still carries the name.
        recovered = [p for p in truth_pairs
                     if _find(res.parent, p[0]) == _find(res.parent, p[1])]

        prec = len(correct) / len(judgeable) if judgeable else float("nan")
        rec = len(recovered) / len(truth_pairs)
        print(f"{gap:>6.1f}{dist:>6}{len(res.links):>7}{len(judgeable):>10}"
              f"{len(correct):>9}{prec:>10.0%}{rec:>9.0%}")
        per_gate_detail[(gap, dist)] = (judgeable, correct, recovered)

    # Where precision breaks, which is what actually sets the threshold.
    print("\nprecision by gap length, at the loosest gate "
          f"({GATES[-1][0]:g}s/{GATES[-1][1]}px):")
    judgeable, correct, _ = per_gate_detail[GATES[-1]]
    ok = Counter(bucket(l.gap_s) for l in correct)
    all_ = Counter(bucket(l.gap_s) for l in judgeable)
    for b, n in sorted(all_.items(), key=lambda x: float(x[0].strip('>s').split('-')[0])):
        print(f"  {b:>8}: {ok[b]:>3}/{n:<3} ({ok[b] / n:.0%})")

    wrong = [l for l in judgeable if l not in correct]
    if wrong:
        print(f"\n{len(wrong)} wrong links at the loosest gate — the expensive "
              f"failure. First few:")
        for l in sorted(wrong, key=lambda l: -l.gap_s)[:8]:
            print(f"  {truth[l.a]} (lane {l.a}) -> {truth[l.b]} (lane {l.b}): "
                  f"gap {l.gap_s:.1f}s, motion err {l.dist_px:.0f}px, "
                  f"{l.speed_px_s:.0f}px/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
