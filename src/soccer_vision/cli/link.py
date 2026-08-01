"""``soccer-vision link-tracks`` — rejoin fragmented lanes after ``process``.

Runs in seconds against a saved run, so a threshold can be tried and reverted
without paying the ~1.6 h a re-``process`` costs. Writes ``.linked.json`` beside
the originals by default; ``--in-place`` swaps them in and keeps the originals as
``.unlinked.json``, which is what the rest of the CLI reads.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from soccer_vision.tracking.link import (LinkConfig, apply_links, link_tracks,
                                         propagate_names)


def _appearance_fn(run_dir: Path, doc: dict, cfg: LinkConfig, device=None):
    """Cosine similarity between lane A's exit crop and lane B's entry crop.

    Geometry alone cannot tell "my player ran on through the dropout" from "a
    different player crossed that spot", and when a lane's true continuation is
    absent entirely it links to a stranger rather than abstaining — 150 wrong
    links on 400 lanes in `slurm/eval_link_appearance.py`. This is the gate that
    catches those: at 0.70 it rejects 85% of them while keeping 89% of genuine
    links.

    Note this asks something much easier than naming a player from a gallery
    (issue #25): same person, under a second later, same pose and light.
    """
    from soccer_vision.identify.reid import ReIDEmbedder, crop_player
    from soccer_vision.io.video import VideoReader

    video = run_dir / "broadcast_proxy.mp4"
    if not video.exists():
        print(f"  --appearance needs {video}; falling back to geometry only")
        return None

    tracks = doc["tracks"]
    edges: dict[tuple[str, str], tuple[int, list]] = {}
    for tid, samples in tracks.items():
        if not samples:
            continue
        edges[(tid, "end")] = (samples[-1]["frame"], samples[-1]["bbox"])
        edges[(tid, "start")] = (samples[0]["frame"], samples[0]["bbox"])

    by_frame: dict[int, list] = {}
    for key, (frame, bbox) in edges.items():
        by_frame.setdefault(frame, []).append((key, bbox))
    frames = sorted(by_frame)
    print(f"  embedding {len(edges):,} lane edges over {len(frames):,} frames...")

    reader = VideoReader(video)
    embedder = ReIDEmbedder.from_pretrained(device=device)
    table: dict[tuple[str, str], np.ndarray] = {}
    # Embed in chunks and drop the crops as we go: holding all ~35k lane-edge
    # crops to pass to one `embed` call OOMs a 64 GB job (35 GB resident and
    # still climbing when it was killed).
    batch_keys: list = []
    batch_crops: list = []

    def flush():
        if not batch_crops:
            return
        emb = embedder.embed(batch_crops)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        for k, e in zip(batch_keys, emb):
            table[k] = e
        batch_keys.clear()
        batch_crops.clear()

    for fno, frame in reader.read_frames(frames):
        for key, bbox in by_frame.get(fno, []):
            c = crop_player(frame, np.array(bbox, dtype=float))
            if c is not None:
                batch_keys.append(key)
                batch_crops.append(c)
        if len(batch_crops) >= 2048:
            flush()
    flush()
    print(f"  embedded {len(table):,} edges")

    def fn(a: str, b: str):
        ea, eb = table.get((a, "end")), table.get((b, "start"))
        if ea is None or eb is None:
            return None          # no crop: don't veto on missing evidence
        return float(ea @ eb)

    return fn


def run_link_tracks(args):
    run_dir = Path(args.run)
    tracks_path = run_dir / "tracks.json"
    if not tracks_path.exists():
        print(f"No tracks.json in {run_dir} — run `process` first.")
        return
    doc = json.loads(tracks_path.read_text())

    cfg = LinkConfig(
        max_gap_s=args.max_gap,
        max_dist_px=args.max_dist,
        require_kit=not args.ignore_kit,
        use_motion=not args.no_motion,
        bidirectional=not args.no_motion,
        min_appearance=args.min_appearance,
    )

    print("=== soccer-vision link-tracks ===")
    print(f"Run:   {run_dir}")
    print(f"Gate:  gap <= {cfg.max_gap_s}s, motion-predicted miss <= {cfg.max_dist_px}px, "
          f"kit {'must match' if cfg.require_kit else 'ignored'}, "
          f"motion {'on' if cfg.use_motion else 'off'}")

    if args.appearance:
        cfg.appearance_fn = _appearance_fn(run_dir, doc, cfg, device=args.device)
        if cfg.appearance_fn is not None:
            print(f"       appearance veto below {cfg.min_appearance}")

    result = link_tracks(doc, cfg)
    s = result.stats
    print(f"\nLanes {s['lanes']:,} -> {s['chains']:,} chains "
          f"({s['links']:,} links from {s['candidates']:,} candidate pairs)")

    linked = apply_links(doc, result, interpolate=not args.no_interpolate)
    n_interp = linked["linking"]["interpolated_samples"]
    lens = [len(v) for v in linked["tracks"].values()]
    iv, fps = int(doc.get("sample_interval", 1)), float(doc["fps"])
    print(f"Chain length: median {np.median(lens) * iv / fps:.1f}s, "
          f"max {max(lens) * iv / fps:.0f}s "
          f"(was {np.median([len(v) for v in doc['tracks'].values()]) * iv / fps:.1f}s / "
          f"{max(len(v) for v in doc['tracks'].values()) * iv / fps:.0f}s)")
    if not args.no_interpolate:
        print(f"Interpolated {n_interp:,} samples across dropouts "
              "(marked `interpolated: true` — positions, not detections)")

    out_tracks = tracks_path if args.in_place else run_dir / "tracks.linked.json"
    if args.in_place:
        shutil.copy2(tracks_path, run_dir / "tracks.unlinked.json")
    out_tracks.write_text(json.dumps(linked))
    print(f"Saved: {out_tracks}")

    jerseys_path = run_dir / "jerseys.json"
    if jerseys_path.exists():
        jerseys = json.loads(jerseys_path.read_text())
        j2, jstats = propagate_names(jerseys, result)
        print(f"\nNames propagated along chains: "
              f"{jstats['named_before']:,} -> {jstats['named_after']:,} lanes named")
        if jstats["conflicts"]:
            print(f"  {jstats['conflicts']:,} chains carry two different names — "
                  "each is a wrong link or a wrong name; see track_links.json")
        out_j = jerseys_path if args.in_place else run_dir / "jerseys.linked.json"
        if args.in_place:
            shutil.copy2(jerseys_path, run_dir / "jerseys.unlinked.json")
        out_j.write_text(json.dumps(j2))
        print(f"Saved: {out_j}")
    else:
        jstats = {}

    audit = run_dir / "track_links.json"
    audit.write_text(json.dumps({
        "config": {
            "max_gap_s": cfg.max_gap_s, "max_dist_px": cfg.max_dist_px,
            "use_motion": cfg.use_motion, "bidirectional": cfg.bidirectional,
            "require_kit": cfg.require_kit,
            "appearance": bool(cfg.appearance_fn),
            "min_appearance": cfg.min_appearance,
        },
        "stats": s,
        "name_propagation": {k: v for k, v in jstats.items()
                             if k != "conflict_detail"},
        "conflicts": jstats.get("conflict_detail", []),
        "links": [vars(l) for l in result.links],
    }))
    print(f"Saved: {audit}  (every link, with the evidence that justified it)")
    if not args.in_place:
        print("\nThese are written alongside the originals. Re-run with --in-place "
              "to make `reel`/`extract` use them (originals kept as *.unlinked.json).")
