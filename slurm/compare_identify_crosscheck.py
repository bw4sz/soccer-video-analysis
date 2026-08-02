"""What the jersey cross-check did to a re-id-only `identify` run.

Reads the jerseys.json from before the cross-check existed (re-id only) and the
one after (``reid+ocr`` with the veto), and answers three separate questions the
headline "named N tracks" hides:

1. **How wrong was re-id?** Of the lanes it named, how many did legible jersey
   reads *disprove* — and which players were being over-claimed.
2. **Would a tighter re-id threshold have caught the same lanes?** If conflicted
   lanes score no lower than corroborated ones, the veto is finding errors the
   similarity score cannot, which is the case for keeping it.
3. **What did OCR add on its own?** The abstained lanes it named are recall the
   veto did not cost.

Standalone: ``python slurm/compare_identify_crosscheck.py <run_dir>``, or pass
``--before`` / ``--after`` explicitly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load(path: Path) -> dict:
    return json.loads(path.read_text()).get("tracks", {})


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", help="Run directory (finds both files in it)")
    ap.add_argument("--before", help="jerseys.json from the re-id-only run")
    ap.add_argument("--after", help="jerseys.json from the reid+ocr run")
    ap.add_argument("--show", type=int, default=25, help="Conflicts to list (default: 25)")
    args = ap.parse_args()

    run = Path(args.run) if args.run else None
    before_p = Path(args.before) if args.before else run / "jerseys.reid-only.json"
    after_p = Path(args.after) if args.after else run / "jerseys.json"
    before, after = load(before_p), load(after_p)

    print("=== identify cross-check: before vs after ===")
    print(f"before: {before_p}")
    print(f"after:  {after_p}")
    print(f"tracks: {len(before):,} -> {len(after):,}")

    # --- 1. the veto -------------------------------------------------------
    verdicts = Counter(r.get("crosscheck") for r in after.values())
    checked = sum(v for k, v in verdicts.items() if k)
    conflicts = [(tid, r) for tid, r in after.items() if r.get("conflict")]

    print(f"\n--- re-id names cross-checked against jersey reads ---")
    print(f"named by re-id before: {sum(1 for r in before.values() if r.get('name')):,}")
    print(f"cross-checked:         {checked:,}")
    for verdict in ("agree", "conflict", "no_evidence"):
        n = verdicts.get(verdict, 0)
        print(f"  {verdict:<12} {n:>6,}  ({pct(n, checked)} of checked)")

    decided = verdicts.get("agree", 0) + verdicts.get("conflict", 0)
    if decided:
        print(f"\nOf the {decided:,} lanes the reader could rule on, "
              f"{pct(verdicts.get('conflict', 0), decided)} were WRONG.")
        print("(Read this as a sample, not the whole run: lanes carrying a legible "
              "number are the larger, better-lit ones, which re-id also finds easier.)")

    # --- who was being over-claimed, and what the shirts actually said -----
    if conflicts:
        by_name = Counter(r["conflict"]["reid_name"] for _, r in conflicts)
        by_read = Counter(r["conflict"]["ocr_jersey"] for _, r in conflicts)
        claimed = Counter(r.get("name") for r in before.values() if r.get("name"))

        print("\n--- players re-id over-claimed (disproved / lanes it claimed) ---")
        for name, n in by_name.most_common(12):
            print(f"  {name:<22} {n:>5,} disproved of {claimed.get(name, 0):>5,} "
                  f"claimed  ({pct(n, claimed.get(name, 0))})")

        print("\n--- numbers the shirts actually read ---")
        for number, n in by_read.most_common(12):
            print(f"  #{number:<4} {n:>5,}")
        print("A number nobody on the roster wears is an opponent the gallery "
              "matched to one of ours — the failure it cannot see on its own.")

    # --- 2. could similarity alone have caught them? -----------------------
    sims: dict[str, list[float]] = defaultdict(list)
    for r in after.values():
        v = r.get("crosscheck")
        s = r.get("similarity")
        if v and s is not None:
            sims[v].append(float(s))
    if sims.get("conflict") and sims.get("agree"):
        print("\n--- re-id similarity, by what the reads said ---")
        for verdict in ("agree", "conflict", "no_evidence"):
            vals = sims.get(verdict, [])
            if vals:
                print(f"  {verdict:<12} n={len(vals):>6,}  mean {mean(vals):.3f}  "
                      f"min {min(vals):.3f}  max {max(vals):.3f}")
        gap = mean(sims["agree"]) - mean(sims["conflict"])
        if abs(gap) < 0.02:
            print(f"  Gap {gap:+.3f} — the score does NOT separate right from wrong, "
                  "so no threshold would have caught these. The veto earns its keep.")
        else:
            print(f"  Gap {gap:+.3f} — worth testing whether raising --min-reid-margin "
                  "removes the same lanes more cheaply.")

    # --- 3. what OCR added on its own -------------------------------------
    added = [tid for tid, r in after.items()
             if r.get("source") == "ocr" and not before.get(tid, {}).get("name")]
    print(f"\n--- OCR on the lanes re-id abstained on ---")
    print(f"newly named: {len(added):,}")

    named_before = sum(1 for r in before.values() if r.get("name"))
    named_after = sum(1 for r in after.values() if r.get("name"))
    print(f"\nnamed overall: {named_before:,} -> {named_after:,} "
          f"({named_after - named_before:+,})")

    # --- per-player lane counts, which is what a reel actually gets --------
    b_counts = Counter(r["name"] for r in before.values() if r.get("name"))
    a_counts = Counter(r["name"] for r in after.values() if r.get("name"))
    print("\n--- lanes per player (what --player selects) ---")
    for name in sorted(set(b_counts) | set(a_counts), key=lambda n: -b_counts.get(n, 0)):
        b, a = b_counts.get(name, 0), a_counts.get(name, 0)
        flag = "  <-- " + ("dropped" if a < b else "gained") if a != b else ""
        print(f"  {name:<22} {b:>5,} -> {a:>5,}{flag}")

    if conflicts:
        print(f"\n--- {min(args.show, len(conflicts))} conflicts in detail ---")
        for tid, r in sorted(conflicts,
                             key=lambda kv: -(kv[1].get("similarity") or 0))[:args.show]:
            c = r["conflict"]
            print(f"  track {tid:>6}: re-id {c['reid_name']} (#{c['reid_jersey']}) "
                  f"@ sim {r.get('similarity')} vs #{c['ocr_jersey']} from "
                  f"{c['n_obs']} reads (conf {c['ocr_confidence']})")


if __name__ == "__main__":
    main()
