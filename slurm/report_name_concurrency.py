"""How often does one child's name sit on two players at once?

A player is in one place at a time. `identify` does not know that — it names each
lane independently — so a name can land on several lanes that are **alive
simultaneously**, and every one but one of them is somebody else. This is issue
#28, and it is the number to check after any change to naming or linking,
because it bounds how good a player reel can possibly be: a reel built from a
name that is on three people is a reel of three people.

Reports, per named player, the share of her named frames carrying more than one
lane. Linking should reduce it (two lanes joined into a chain stop being two
lanes); re-id abstaining more should reduce it; nothing else in the pipeline
touches it.

Usage:
    python slurm/report_name_concurrency.py --run /path/to/runs/<match_id>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def concurrency(tracks: dict, jerseys: dict, fps: float, interval: int):
    """``{name: (frames, Counter(n_concurrent -> frames))}`` over sampled frames."""
    by_name: dict[str, list[str]] = collections.defaultdict(list)
    for tid, rec in jerseys.get("tracks", {}).items():
        name = rec.get("name")
        if name and tid in tracks:
            by_name[name].append(tid)

    out = {}
    for name, tids in by_name.items():
        per_frame: collections.Counter = collections.Counter()
        for tid in tids:
            for s in tracks[tid]:
                per_frame[s["frame"]] += 1
        hist = collections.Counter(per_frame.values())
        out[name] = (len(tids), len(per_frame), hist)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--tracks", default="tracks.json")
    ap.add_argument("--jerseys", default="jerseys.json")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    doc = json.loads((args.run / args.tracks).read_text())
    jer = json.loads((args.run / args.jerseys).read_text())
    fps = float(doc["fps"])
    interval = int(doc.get("sample_interval") or 1)
    secs = interval / fps

    res = concurrency(doc["tracks"], jer, fps, interval)
    if not res:
        print("  no named lanes")
        return

    rows = sorted(res.items(), key=lambda kv: -kv[1][1])[:args.top]
    print(f"{'player':<24} {'lanes':>6} {'named_s':>8} {'clean_s':>8} "
          f"{'2+_s':>7} {'2+%':>6} {'max':>4}")
    tot_named = tot_multi = 0.0
    for name, (n_lanes, n_frames, hist) in rows:
        multi = sum(v for k, v in hist.items() if k > 1)
        named_s = n_frames * secs
        tot_named += named_s
        tot_multi += multi * secs
        print(f"{name:<24} {n_lanes:6d} {named_s:8.0f} {(n_frames - multi) * secs:8.0f} "
              f"{multi * secs:7.0f} {100 * multi / n_frames:5.1f}% {max(hist):4d}")

    all_multi = 0
    all_frames = 0
    for _n, (_l, f, h) in res.items():
        all_frames += f
        all_multi += sum(v for k, v in h.items() if k > 1)
    print(f"\nover all {len(res)} named players: "
          f"{100 * all_multi / all_frames:.1f}% of named frames carry 2+ lanes "
          f"of the same name ({all_multi * secs:.0f}s of {all_frames * secs:.0f}s)")
    print("A player is in one place at a time, so every one of those frames has "
          "another player under her name — see issue #28.")


if __name__ == "__main__":
    main()
