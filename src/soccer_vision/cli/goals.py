"""CLI: soccer-vision goals — find the goal mouths, then the goals.

Two stages, one command, deliberately separable:

1. **Detect the mouths** (needs a GPU) — SAM3 with the text prompt
   ``"soccer goal net"``, sampled sparsely and median-consolidated into one box
   per end. Cached as ``goals.json``; re-runs reuse it unless ``--redetect``.
2. **Derive the events** (pure geometry, instant) — replay ``ball_track.json``
   through the mouths and emit ``goal`` wherever the ball dwells inside one.

The split is the point. Stage 1 is the expensive, camera-dependent half and
barely changes; stage 2 is where the thresholds live, so tuning ``--min-dwell``
against a match you know the score of costs seconds, not another SAM3 pass.

Events are merged into the run's ``annotations.json`` (replacing any previous
``goal`` events) so ``extract --events goal`` and ``reel`` cut them with no
special-casing.
"""

from __future__ import annotations

import json
from pathlib import Path


def run_goals(args):
    run_dir = Path(args.run)
    ball_track_path = run_dir / "ball_track.json"
    goals_path = run_dir / "goals.json"
    proxy_path = run_dir / "broadcast_proxy.mp4"
    annotations_path = run_dir / "annotations.json"

    if not ball_track_path.exists():
        print(f"No ball_track.json in {run_dir} — run `soccer-vision process` first.")
        return

    print("=== soccer-vision goals ===")
    print(f"Run: {run_dir}")

    # --- Stage 1: goal mouths (cached) --------------------------------------
    goals = None
    if goals_path.exists() and not args.redetect:
        goals = json.loads(goals_path.read_text())
        print(f"Goal mouths: reusing {goals_path} "
              f"({len(goals.get('goals', []))} found; --redetect to rebuild)")
    else:
        from soccer_vision.detection.goal import detect_goal_mouths, is_available

        if not is_available():
            print("SAM3 is not available (needs transformers>=5.12 + CUDA), so the "
                  "goal mouths can't be detected here. Run this step on a GPU node; "
                  "the resulting goals.json is portable and the event stage is CPU-only.")
            return
        if not proxy_path.exists():
            print(f"No broadcast_proxy.mp4 in {run_dir} — run `soccer-vision process` first.")
            return

        print(f"Detecting goal mouths with prompt {args.prompt!r} ...")
        goals = detect_goal_mouths(
            str(proxy_path),
            prompt=args.prompt,
            sample_fps=args.sample_fps,
            max_samples=args.max_samples,
            device=args.device or "cuda",
            min_obs=args.min_obs,
        )
        goals_path.write_text(json.dumps(goals, indent=2))
        print(f"  {goals['n_candidate_boxes']} candidate boxes over "
              f"{goals['n_frames_sampled']} frames -> {goals_path}")

    for g in goals.get("goals", []):
        x1, y1, x2, y2 = g["bbox"]
        print(f"  [{g['side']:>5}] bbox=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) "
              f"score={g['score']:.2f} n_obs={g['n_obs']}")
    if not goals.get("goals"):
        print("  No goal mouth survived consolidation — nothing to detect against.")
        print("  Try --prompt 'goal posts and net', a denser --sample-fps, or check "
              "that both goals are actually in frame.")
        return

    # --- Stage 2: goal events (pure geometry) -------------------------------
    from soccer_vision.events.goal import GOAL_LABEL, detect_goals, load_goal_regions

    ball_track = json.loads(ball_track_path.read_text())
    regions = load_goal_regions(goals)
    events = detect_goals(
        ball_track,
        regions,
        min_dwell_s=args.min_dwell,
        max_dwell_s=args.max_dwell,
        max_gap_s=args.max_gap,
        inset_frac=args.inset,
        require_entry=not args.no_entry_check,
        dedup_window_s=args.dedup_window,
    )
    for e in events:
        e.setdefault("source", "rules")

    print(f"\nGoal events: {len(events)}")
    for e in events:
        mins, secs = divmod(int(e["timestamp_s"]), 60)
        print(f"  {mins:>3}:{secs:02d}  [{e['goal_zone']:>5}]  "
              f"dwell={e['dwell_s']:.1f}s  drift={e['drift_px']:.0f}px  "
              f"conf={e['confidence']:.2f}")
    if not events:
        print("  (nothing crossed the dwell threshold — try a lower --min-dwell, "
              "or check ball-track coverage at the goal ends)")

    if args.dry_run:
        print("\n--dry-run: annotations.json not modified.")
        return

    # Merge into the run's events, replacing any previous goal pass.
    if annotations_path.exists():
        doc = json.loads(annotations_path.read_text())
        prior = [e for e in doc.get("events", []) if e.get("label") == GOAL_LABEL]
        doc["events"] = [e for e in doc.get("events", []) if e.get("label") != GOAL_LABEL]
        doc["events"].extend(events)
        doc["events"].sort(key=lambda e: e.get("timestamp_s", 0.0))
        annotations_path.write_text(json.dumps(doc, indent=2))
        replaced = f", replacing {len(prior)} from a previous run" if prior else ""
        print(f"\nMerged {len(events)} goal events into {annotations_path}{replaced}")
    else:
        print(f"\nNo annotations.json in {run_dir} — goal events not merged. "
              f"(They are reproducible from {goals_path} at any time.)")
