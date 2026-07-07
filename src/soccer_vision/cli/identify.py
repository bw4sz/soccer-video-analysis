"""CLI: soccer-vision identify — read jersey numbers per track.

Powers the individual-player query pathway. Runs as its own opt-in step (not part
of `process`) over an already-processed run: it reads the per-frame player boxes
in ``tracks.json``, crops the number region from the broadcast proxy, recognizes
digits, votes one number per track, and writes ``jerseys.json``. Downstream,
`extract`/`reel --player NAME` / `--number N` resolve to the matching lanes.
"""

from __future__ import annotations

import json
from pathlib import Path


def run_identify(args):
    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.identify.jersey_ocr import JerseyNumberRecognizer, assign_jerseys
    from soccer_vision.io.video import VideoReader
    from soccer_vision.profiles.loader import get_player, load_profile

    run_dir = Path(args.run)
    tracks_path = run_dir / "tracks.json"
    proxy_path = run_dir / "broadcast_proxy.mp4"
    jerseys_path = run_dir / "jerseys.json"

    if not tracks_path.exists():
        print(f"No tracks.json in {run_dir} — run `soccer-vision process` first.")
        return
    if not proxy_path.exists():
        print(f"No broadcast_proxy.mp4 in {run_dir} — run `soccer-vision process` first.")
        return

    profile = load_profile(args.profile) if args.profile else None

    print("=== soccer-vision identify ===")
    print(f"Run:   {run_dir}")
    track_boxes = load_track_boxes(tracks_path)
    print(f"Tracks: {len(track_boxes)}")

    print("Loading jersey-number recognizer...")
    recognizer = JerseyNumberRecognizer.from_pretrained(
        model_id=args.model, device=args.device
    )

    vote_kwargs = {
        "min_votes": args.min_votes,
        "min_share": args.min_share,
        "min_margin": args.min_margin,
    }
    reader = VideoReader(proxy_path)
    try:
        votes = assign_jerseys(
            track_boxes, recognizer, reader,
            max_samples_per_track=args.max_samples,
            vote_kwargs=vote_kwargs,
            progress=True,
        )
    finally:
        reader.close()

    tracks_out = {}
    for tid, v in votes.items():
        name = None
        if profile is not None and v.jersey is not None:
            player = get_player(profile, v.jersey)
            name = player.get("name") if player else None
        tracks_out[str(tid)] = {
            "jersey": v.jersey,
            "confidence": round(v.confidence, 3),
            "n_obs": v.n_obs,
            "legible_frac": round(v.legible_frac, 3),
            "name": name,
        }

    doc = {
        "video": proxy_path.name,
        "model": args.model or "parseq",
        "tracks": tracks_out,
    }
    with open(jerseys_path, "w") as f:
        json.dump(doc, f, indent=2)

    named = sum(1 for t in tracks_out.values() if t["jersey"] is not None)
    print(f"\nIdentified {named}/{len(tracks_out)} tracks. Saved: {jerseys_path}")
    print("Next: soccer-vision reel --run "
          f"{run_dir} --player <name>   (or --number <N>)")
