"""Diff a `identify` run against its pre-negative-class baseline.

The windowed sweep measured *rejection rates*; this measures the thing that
actually reaches a parent: how many lanes still carry a name, whose names they
are, and how much of that name-bearing time sits high in the frame where the
spectators are.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())["tracks"]


def foot_by_lane(run: Path) -> dict[int, float]:
    import numpy as np
    from soccer_vision.clips.halo import load_track_boxes

    out = {}
    for tid, samples in load_track_boxes(run / "tracks.json").items():
        if samples:
            out[int(tid)] = float(np.median([b[3] for _, b in samples]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--high-frac", type=float, default=0.40,
                    help="lanes with median foot-y above this are spectator-rich")
    args = ap.parse_args()

    before = load(args.baseline)
    after = load(args.run / "jerseys.json")
    foot = foot_by_lane(args.run)
    cut = args.high_frac * 1080

    def named(doc):
        return {int(k) for k, v in doc.items() if v.get("name")}

    nb, na = named(before), named(after)
    print(f"named lanes: {len(nb)} → {len(na)}  ({len(na) - len(nb):+d})")

    lost, kept, gained = nb - na, nb & na, na - nb
    print(f"  lost {len(lost)}, kept {len(kept)}, gained {len(gained)}")

    def high(ids):
        return sum(1 for t in ids if foot.get(t, 1e9) < cut)

    print(f"\nof lanes that lost their name: {high(lost)}/{len(lost)} sat above "
          f"{args.high_frac:.2f}H (spectator-rich)")
    print(f"of lanes that kept it:          {high(kept)}/{len(kept)}")

    rej = [int(k) for k, v in after.items() if v.get("excluded") == "not_ours"]
    print(f"\nrejected as 'not ours': {len(rej)} lanes, "
          f"{high(rej)} of them above {args.high_frac:.2f}H")

    print("\nnames before → after:")
    cb = Counter(before[str(t)]["name"] for t in nb)
    ca = Counter(after[str(t)]["name"] for t in na)
    for name in sorted(set(cb) | set(ca)):
        print(f"  {name:>24}: {cb.get(name,0):5d} → {ca.get(name,0):5d}")


if __name__ == "__main__":
    main()
