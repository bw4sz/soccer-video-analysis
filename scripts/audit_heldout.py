#!/usr/bin/env python
"""Check a gallery — or a whole galleries/ directory — against ``heldout.yaml``.

`enroll` already refuses to bank a crop from protected footage, so in the normal
course this reports nothing. It exists for the abnormal course: a gallery built
before the registry existed, one built with ``--heldout-mode off``, one appended
to from a stale file, or a block whose bounds moved after the fact. Any of those
turns an evaluation number into a memory test, and none of them leave a mark on
the number itself.

Three verdicts, and the middle one is the point:

``CLEAN``
    Every exemplar is stamped with its footage and frame, and none falls inside a
    protected span.
``UNAUDITABLE``
    Exemplars carry no provenance (or footage the registry doesn't know). This is
    **not** a pass. It means the file cannot answer the question, and a number
    measured against it cannot be defended.
``LEAK``
    Named exemplars sit inside a protected span. The gallery must be rebuilt
    before anything is scored against that block.

    python scripts/audit_heldout.py --gallery galleries/saints-u14g.npz
    python scripts/audit_heldout.py --all              # every gallery in galleries/
    python scripts/audit_heldout.py --run runs/<match> # what a run would leak
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soccer_vision.heldout import load_registry  # noqa: E402
from soccer_vision.identify.gallery import load_gallery, parse_source  # noqa: E402

CLEAN, UNAUDITABLE, LEAK = "CLEAN", "UNAUDITABLE", "LEAK"


def audit_gallery(path: Path, registry) -> dict:
    blank = {"path": path, "verdict": UNAUDITABLE, "n": 0, "unstamped": 0,
             "leaks": [], "frameless": 0, "unknown_footage": Counter()}
    try:
        gallery = load_gallery(path)
    except KeyError:
        # Negative-class dumps (`emb` + `tids`) live in galleries/ but are crop
        # banks, not galleries. They still enrol into one, so they are worth
        # flagging rather than skipping silently.
        return {**blank, "note": "not a gallery (no names/label) — a crop bank; "
                                 "audit the gallery it was enrolled into"}
    n = len(gallery["emb"])
    source = gallery.get("source") or []
    names = [gallery["names"][i] for i in gallery["label"]]

    if not source:
        return {"path": path, "verdict": UNAUDITABLE, "n": n, "unstamped": n,
                "leaks": [], "frameless": 0, "unknown_footage": Counter(),
                "note": "no provenance — built before stamping, or by a route "
                        "that doesn't record it"}

    leaks, frameless, unstamped = [], 0, 0
    unknown_footage = Counter()
    for i, (raw, name) in enumerate(zip(source, names)):
        parsed = parse_source(raw)
        if parsed is None:
            unstamped += 1
            continue
        key, frame = parsed
        spans = registry.spans_for(key)
        if not spans:
            unknown_footage[key] += 1
            continue
        if frame < 0:
            frameless += 1
            continue
        hit = next((s for s in spans if s.contains(frame)), None)
        if hit is not None:
            leaks.append((i, name, key, frame, hit))

    verdict = LEAK if leaks else (UNAUDITABLE if unstamped else CLEAN)
    return {"path": path, "verdict": verdict, "n": n, "unstamped": unstamped,
            "leaks": leaks, "frameless": frameless,
            "unknown_footage": unknown_footage, "note": ""}


def audit_run(run_dir: Path, registry) -> None:
    """How much of a run sits inside a protected span — the cost of the block."""
    import json

    sys.path.insert(0, str(REPO / "src"))
    from soccer_vision.clips.halo import load_track_boxes

    tracks_path = run_dir / "tracks.json"
    doc = json.loads(tracks_path.read_text())
    fps = float(doc.get("fps") or 30.0)
    spans = registry.spans_for(run_dir)

    print(f"\n{run_dir}")
    if not spans:
        print(f"  {registry.describe_coverage(run_dir)}")
        return
    samples = load_track_boxes(tracks_path)
    total = sum(len(s) for s in samples.values())
    inside = sum(1 for s in samples.values() for f, _ in s
                 if any(sp.contains(f) for sp in spans))
    lanes_touching = sum(1 for s in samples.values()
                         if any(sp.contains(f) for f in (x[0] for x in s) for sp in spans))
    for sp in spans:
        print(f"  protected: {sp}")
    print(f"  {inside}/{total} track samples ({100 * inside / max(1, total):.1f}%) "
          f"and {lanes_touching}/{len(samples)} lanes fall inside")
    print(f"  = {inside / fps:.0f} lane-seconds withheld from enrolment")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", type=Path, action="append", default=[],
                    help="Gallery .npz to audit (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="Audit every .npz in galleries/")
    ap.add_argument("--run", type=Path, action="append", default=[],
                    help="Report how much of a run the registry withholds")
    ap.add_argument("--heldout", type=Path, default=None, help="Registry path")
    ap.add_argument("--verbose", action="store_true", help="List every leaking exemplar")
    args = ap.parse_args()

    registry = load_registry(args.heldout)
    print(f"Registry: {registry.path or '<none found>'} — {len(registry.blocks)} block(s)")
    for b in registry.blocks:
        golds = ", ".join(g.gold_id for g in b.gold) or "no gold set yet"
        print(f"  {b.id:<36} {b.flavour:<14} {b.start_s:.0f}-{b.end_s:.0f}s  ({golds})")

    galleries = list(args.gallery)
    if args.all:
        galleries += sorted((REPO / "galleries").glob("*.npz"))
    if not galleries and not args.run:
        galleries = sorted((REPO / "galleries").glob("*.npz"))

    worst = CLEAN
    if galleries:
        print(f"\n{'gallery':<44} {'verdict':<12} exemplars")
        for path in galleries:
            r = audit_gallery(path, registry)
            detail = []
            if r["leaks"]:
                detail.append(f"{len(r['leaks'])} inside a protected span")
            if r["unstamped"]:
                detail.append(f"{r['unstamped']} unstamped")
            if r["frameless"]:
                detail.append(f"{r['frameless']} run-level only")
            for key, n in r["unknown_footage"].most_common():
                detail.append(f"{n} from unregistered '{key}'")
            if r["note"]:
                detail.append(r["note"])
            print(f"{path.name:<44} {r['verdict']:<12} {r['n']:>5}"
                  + (f"   ({'; '.join(detail)})" if detail else ""))
            if r["leaks"] and args.verbose:
                for _, name, key, frame, span in r["leaks"][:40]:
                    print(f"    LEAK {name:<24} {key}@{frame}  in {span}")
            if r["verdict"] == LEAK:
                worst = LEAK
            elif r["verdict"] == UNAUDITABLE and worst != LEAK:
                worst = UNAUDITABLE

    for run_dir in args.run:
        audit_run(run_dir, registry)

    if galleries:
        print(f"\nWorst verdict: {worst}")
        if worst == LEAK:
            print("Rebuild the leaking gallery from its annotations before scoring "
                  "anything against that block. `--heldout-mode exclude` is the "
                  "default, so a plain re-run of `enroll` is enough.")
        elif worst == UNAUDITABLE:
            print("Unauditable is not clean. A gallery without provenance cannot "
                  "support a generalization claim — rebuild it from the source "
                  "annotations so every exemplar carries its frame.")
    return 1 if worst == LEAK else 0


if __name__ == "__main__":
    raise SystemExit(main())
