"""What `identify --team` did to a run that was named with no kit gate.

The gate's claim is narrow and checkable: it should remove names from lanes
wearing the *other* squad's kit and leave our own squad's names alone. Two
jerseys.json files — before and after — are enough to test that, plus one thing
they can settle that the counts cannot:

**Did the jersey veto stop firing?** Job 38526407 found 38 re-id names disproved
by legible reads, and the numbers those shirts actually carried (#2, #1, #20,
#23) are mostly worn by nobody on our roster — which is what you would see if the
gallery were pulling *opponents* onto our squad. If that reading is right, gating
on kit should remove most of those conflicts at the source. If the conflict count
holds steady on our own kit, the errors are teammate confusions instead and the
gate was never going to touch them.

Usage:
    python slurm/compare_team_gate.py <run_dir> --before jerseys.pre-teamgate.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text()).get("tracks", {})


def kits(run_dir: Path) -> dict[str, str | None]:
    doc = json.loads((run_dir / "tracks.json").read_text())
    return {str(k): v for k, v in (doc.get("teams") or {}).items()}


def named(tracks: dict) -> set[str]:
    return {tid for tid, r in tracks.items() if r.get("name")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="Run directory")
    ap.add_argument("--before", default="jerseys.pre-teamgate.json",
                    help="jerseys.json from before the gate (relative to run, or a path)")
    ap.add_argument("--after", default="jerseys.json")
    ap.add_argument("--team", default="black", help="The kit that is ours")
    args = ap.parse_args()

    run = Path(args.run)
    before_p = Path(args.before) if Path(args.before).exists() else run / args.before
    after_p = Path(args.after) if Path(args.after).exists() else run / args.after
    before, after = load(before_p), load(after_p)
    kit_of = kits(run)

    b_named, a_named = named(before), named(after)
    print("=== identify --team: what the kit gate changed ===")
    print(f"before: {before_p}  ({len(b_named)} named of {len(before)})")
    print(f"after:  {after_p}  ({len(a_named)} named of {len(after)})")

    # Names removed and kept, split by the kit the lane was wearing. The gate's
    # whole claim is that these two columns look completely different.
    removed = Counter(kit_of.get(t) or "unassigned" for t in b_named - a_named)
    kept = Counter(kit_of.get(t) or "unassigned" for t in b_named & a_named)
    added = Counter(kit_of.get(t) or "unassigned" for t in a_named - b_named)
    print(f"\n{'kit':<12}{'names kept':>12}{'names removed':>15}{'names added':>13}")
    for k in sorted(set(removed) | set(kept) | set(added)):
        print(f"{k:<12}{kept[k]:>12}{removed[k]:>15}{added[k]:>13}")

    ours_removed = removed.get(args.team, 0)
    if ours_removed:
        print(f"\n{ours_removed} name(s) on our own {args.team} kit disappeared — the gate "
              f"should not do that. Expect these to be OCR names whose lane lost its "
              f"cross-check, not gate exclusions.")

    # Per player: who was inflated by the other squad.
    print(f"\n{'player':<24}{'before':>8}{'after':>8}{'delta':>8}")
    b_by = Counter(r["name"] for r in before.values() if r.get("name"))
    a_by = Counter(r["name"] for r in after.values() if r.get("name"))
    for name in sorted(set(b_by) | set(a_by), key=lambda n: -b_by[n]):
        print(f"{name:<24}{b_by[name]:>8}{a_by[name]:>8}{a_by[name] - b_by[name]:>+8}")

    # The prediction under test: conflicts should collapse if they were opponents.
    for label, doc in (("before", before), ("after", after)):
        conf = [r for r in doc.values() if r.get("conflict")]
        nums = Counter(r["conflict"]["ocr_jersey"] for r in conf)
        print(f"\njersey veto, {label}: {len(conf)} re-id name(s) contradicted by "
              f"legible reads")
        if nums:
            print("  numbers read: " + ", ".join(
                f"#{n} x{c}" for n, c in nums.most_common(10)))

    excluded = sum(1 for r in after.values() if r.get("excluded") == "kit")
    print(f"\nlanes held back by the gate: {excluded}")


if __name__ == "__main__":
    main()
