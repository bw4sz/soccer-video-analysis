"""soccer-vision describe: run SoccerChat over a processed run's event clips.

Reads the clips the pipeline already extracted, asks SoccerChat to caption and
classify each, writes ``soccerchat.json``, and annotates the run's OSL events
with SoccerChat's verdict (marking CONFIRMED events ``verified``). Requires the
``soccerchat`` optional dependency and, in practice, a GPU — launch it on HPC
with ``training/slurm/soccerchat_describe.sbatch``.
"""

from __future__ import annotations

import json
from pathlib import Path


def run_describe(args):
    from soccer_vision.clips.extract import pair_events_with_clips
    from soccer_vision.io.osl import read_osl, write_osl
    from soccer_vision.verify import soccerchat as sc

    run_dir = Path(args.run)
    osl_path = run_dir / "annotations.json"
    if not osl_path.exists():
        print(f"No annotations.json in {run_dir}. Run 'soccer-vision process' first.")
        return
    if not sc.is_available():
        print(
            "SoccerChat runtime not installed. Install the extra:\n"
            "  uv sync --extra soccerchat   (or: pip install 'soccer-vision[soccerchat]')"
        )
        return

    osl = read_osl(osl_path)
    events = osl.get("events", [])
    for e in events:
        e.setdefault("timestamp_s", round(e.get("position_ms", 0) / 1000, 2))

    pairs = [(e, c) for e, c in pair_events_with_clips(events, run_dir / "clips") if c is not None]
    if not pairs:
        print(f"No event clips found in {run_dir / 'clips'}. Nothing to describe.")
        return
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Running SoccerChat on {len(pairs)} clip(s) — loading weights on first clip...")
    model = sc.SoccerChatModel(
        adapter=args.adapter,
        model=args.model,
        max_frames=args.max_frames,
    )
    result = sc.verify_events(pairs, model=model, describe=not args.no_caption)

    # Persist the full per-clip results alongside the run.
    out = {
        "model": {"adapter": args.adapter, "model": args.model, "max_frames": args.max_frames},
        "results": result["results"],
        "verified": result["verified"],
        "rejected": result["rejected"],
    }
    sc_path = run_dir / "soccerchat.json"
    sc_path.write_text(json.dumps(out, indent=2))

    # Annotate OSL events in place with SoccerChat's read on each.
    by_frame = {r.get("frame"): r for r in result["results"]}
    for event in events:
        r = by_frame.get(event.get("frame"))
        if not r:
            continue
        event["soccerchat"] = {
            "sc_class": r.get("sc_class"),
            "sc_label": r.get("sc_label"),
            "verdict": r.get("verdict"),
            "confidence": r.get("confidence"),
            "caption": r.get("caption"),
        }
        if r.get("verdict") == "CONFIRMED":
            event["verified"] = True
    write_osl(osl, osl_path)

    # Summary.
    tally: dict[str, int] = {}
    for r in result["results"]:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("\nSoccerChat verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    for r in result["results"]:
        cap = (r.get("caption") or "").strip()
        print(f"  F{r.get('frame')} {r['label']:<12} {r['verdict']:<9} {r['reason']}")
        if cap:
            print(f"      “{cap}”")
    print(f"\nSaved: {sc_path}")
    print(f"OSL annotated: {osl_path}")
    print("Next: soccer-vision annotate --run", run_dir, "(review in Label Studio)")
