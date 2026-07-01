"""soccer-vision verify and ask CLI commands."""

from __future__ import annotations

import json
from pathlib import Path


def run_verify(args):
    """Verify candidate events from a run using Claude."""
    from soccer_vision.io.osl import read_osl, write_osl
    from soccer_vision.io.project import RunDir
    from soccer_vision.profiles.loader import load_profile
    from soccer_vision.verify.claude import verify_events

    run_dir = Path(args.run)

    # Load OSL events
    osl_path = run_dir / "annotations.json"
    if not osl_path.exists():
        print(f"No annotations.json in {run_dir}. Run 'soccer-vision process' first.")
        return

    osl = read_osl(osl_path)
    candidates = osl.get("events", [])
    if not candidates:
        print("No events to verify.")
        return

    # Load contact sheets
    sheets_dir = run_dir / "sheets"
    sheet_paths = sorted(sheets_dir.glob("sheet_*.jpg")) if sheets_dir.exists() else []
    if not sheet_paths:
        print(f"No contact sheets found in {sheets_dir}. Run pipeline first.")
        return

    # Optional profile
    profile = None
    if args.profile:
        profile = load_profile(Path(args.profile))

    print(f"Verifying {len(candidates)} events across {len(sheet_paths)} sheet(s)...")
    result = verify_events(sheet_paths, candidates, profile=profile)

    verified = result.get("verified", [])
    rejected = result.get("rejected", [])

    print(f"\nVerified: {len(verified)}")
    for e in verified:
        print(f"  F{e.get('frame', '?')} {e.get('label', '?')} — {e.get('reason', '')}")

    print(f"\nRejected: {len(rejected)}")
    for e in rejected:
        print(f"  F{e.get('frame', '?')} {e.get('label', '?')} — {e.get('reason', '')}")

    # Patch OSL: mark verified flag and remove rejected
    verified_frames = {e["frame"] for e in verified if "frame" in e}
    patched_events = []
    for event in osl["events"]:
        fn = event.get("frame")
        if fn in verified_frames:
            event["verified"] = True
            patched_events.append(event)
        elif fn is not None and fn not in {e.get("frame") for e in rejected}:
            patched_events.append(event)
        # else: drop rejected

    osl["events"] = patched_events
    write_osl(osl, osl_path)
    print(f"\nOSL updated: {osl_path} ({len(patched_events)} events retained)")

    # Save raw result
    result_path = run_dir / "verify_result.json"
    with open(result_path, "w") as f:
        json.dump({"verified": verified, "rejected": rejected}, f, indent=2)
    print(f"Verify result saved: {result_path}")


def run_ask(args):
    """Answer a natural language question about a processed run."""
    from soccer_vision.profiles.loader import load_profile
    from soccer_vision.verify.claude import query_match

    run_dir = Path(args.run)
    osl_path = run_dir / "annotations.json"
    stats_path = run_dir / "stats.json"

    profile = None
    if args.profile:
        profile = load_profile(Path(args.profile))

    print(f"Question: {args.question}\n")
    answer = query_match(
        args.question,
        osl_path=osl_path if osl_path.exists() else None,
        stats_path=stats_path if stats_path.exists() else None,
        profile=profile,
    )
    print(answer)
