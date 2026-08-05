"""Report eval_negative_class.py results with the enrolled lanes held out.

The negatives were harvested from the same match they are evaluated on, so a
handful of query lanes are also gallery exemplars and would match themselves.
That is leakage, and on the off-pitch side it inflates exactly the number the
experiment exists to measure — so those lanes are dropped rather than explained
away in a footnote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NOT_OURS = "not ours"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--negatives-from", type=Path, default=None,
                    help="candidates.json / npz whose lane ids were enrolled")
    ap.add_argument("--exclude", default="",
                    help="comma-separated lane ids to hold out")
    args = ap.parse_args()

    rows = json.loads(args.eval.read_text())
    drop = {int(t) for t in args.exclude.split(",") if t.strip()}
    if args.negatives_from and args.negatives_from.suffix == ".npz":
        import numpy as np
        with np.load(args.negatives_from) as z:
            if "tids" in z:
                drop |= {int(t) for t in z["tids"]}

    kept = [r for r in rows if r["tid"] not in drop]
    print(f"{len(rows)} lanes scored, {len(rows) - len(kept)} held out as enrolled "
          f"→ {len(kept)} evaluated\n")

    H = 1080
    off = [r for r in kept if r["foot"] / H < 0.40]
    on = [r for r in kept if r["foot"] / H >= 0.45]

    def named(group, key):
        return sum(1 for r in group if r.get(key) and r[key] != NOT_OURS)

    print(f"{'scoring':>30} {'names off-pitch':>16} {'names on-pitch':>16}")
    print(f"{'baseline (12 players)':>30} "
          f"{named(off,'base_name'):>7}/{len(off):<8} {named(on,'base_name'):>7}/{len(on):<8}")
    for label in ("neg@64", "neg@all"):
        key = label + "_name"
        if key not in kept[0]:
            continue
        print(f"{'+ off-pitch negatives ' + label:>30} "
              f"{named(off, key):>7}/{len(off):<8} {named(on, key):>7}/{len(on):<8}")
    for floor in (0.70, 0.75, 0.80):
        o = sum(1 for r in off if r["base_name"] and r["base_sim"] >= floor)
        n = sum(1 for r in on if r["base_name"] and r["base_sim"] >= floor)
        print(f"{'min_similarity ' + str(floor):>30} {o:>7}/{len(off):<8} {n:>7}/{len(on):<8}")

    for label in ("neg@64", "neg@all"):
        key = label + "_name"
        if key not in kept[0]:
            continue
        ro = sum(1 for r in off if r[key] == NOT_OURS)
        rn = sum(1 for r in on if r[key] == NOT_OURS)
        print(f"\n{label}: rejected as 'not ours' — "
              f"off-pitch {ro}/{len(off)} ({ro/max(len(off),1):.0%}), "
              f"on-pitch {rn}/{len(on)} ({rn/max(len(on),1):.0%})")

    # the decision that matters: of lanes the baseline *named*, which got saved?
    print("\nof lanes the baseline named (a name is what reaches a reel):")
    for label in ("neg@64", "neg@all"):
        key = label + "_name"
        if key not in kept[0]:
            continue
        bo = [r for r in off if r["base_name"]]
        bn = [r for r in on if r["base_name"]]
        ko = sum(1 for r in bo if r[key] == NOT_OURS)
        kn = sum(1 for r in bn if r[key] == NOT_OURS)
        print(f"  {label}: killed {ko}/{len(bo)} off-pitch names, "
              f"{kn}/{len(bn)} on-pitch names")


if __name__ == "__main__":
    main()
