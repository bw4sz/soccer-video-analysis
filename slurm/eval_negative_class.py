"""Does an explicit "not ours" class in the gallery reject spectators?

`match_track` is a closed-set scorer: it ranks our eleven players and abstains
only when the winner fails an absolute floor or a margin test. Handed a
spectator, a referee or an opponent it returns whichever of ours is nearest, and
on `runs/saints-u14g-full-30fps` that puts our players' names on lanes standing
well above the far touchline.

Thresholding the closed-set score is the hard way to fix that — measured on the
same run, the off-pitch and on-pitch similarity distributions overlap almost
completely in the body (medians 0.642 vs 0.668) and separate only in the tail.
The better-posed version is to give the gallery somewhere to put "none of the
above": enrol negatives as a twelfth class, and rejection becomes an ordinary
nearest-neighbour win.

The labels for that already exist and are being thrown away.
`annotate.tracklets.named_lanes_from_export` drops NOT_OURS as "a refusal to
identify, not an identity" — true for enrolling *players*, but those lanes are
exactly the negatives this needs, and the annotator instructions already say to
use it for "opponents, referees and spectators".

Run:
    python slurm/eval_negative_class.py --run runs/saints-u14g-full-30fps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NOT_OURS = "not ours"

# Tracklet exports carrying `not ours` labels, with the run whose tracks.json and
# video the slot->track manifest refers to.
NEGATIVE_SOURCES = [
    ("runs/saints-u14g-full/tracklets/annotations.json",
     "runs/saints-u14g-full/tracklets/tracklets.json"),
    ("runs/u14g_tracklets/annotations.json",
     "runs/u14g_tracklets/tracklets.json"),
]


def negative_lanes(export: list[dict], manifest: dict) -> list[dict]:
    """Lanes an annotator marked ``not ours``.

    Deliberately a near-copy of :func:`named_lanes_from_export` with the filter
    inverted, rather than a parameter on that function: this is an experiment,
    and the production path should keep refusing to enrol a refusal as a player
    until this measurement says the negative class is worth having.
    """
    from soccer_vision.annotate.tracklets import _slot_of, _window_of_clip

    windows = {w["window"]: w for w in manifest["windows"]}
    out = []
    for task in export:
        data = task.get("data", {}) or {}
        key = data.get("window")
        if key not in windows:
            key = _window_of_clip(data.get("video"))
        window = windows.get(key)
        if window is None:
            continue
        lane_of = {lane["slot"]: lane for lane in window["lanes"]}
        for ann in task.get("annotations", []) or []:
            for res in ann.get("result", []) or []:
                choices = (res.get("value", {}) or {}).get("choices") or []
                if not choices or str(choices[0]).strip().lower() != NOT_OURS:
                    continue
                slot = _slot_of(res.get("from_name"))
                if slot is not None and slot in lane_of:
                    out.append(lane_of[slot])
    return out


def embed_negatives(embedder, max_samples_per_lane: int = 20) -> np.ndarray:
    """Embed every ``not ours`` lane across all known tracklet exports."""
    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.identify.reid import crop_player
    from soccer_vision.io.video import VideoReader

    all_crops = []
    for export_path, manifest_path in NEGATIVE_SOURCES:
        export_path, manifest_path = Path(export_path), Path(manifest_path)
        if not export_path.exists():
            print(f"  (skip {export_path} — not present)")
            continue
        manifest = json.loads(manifest_path.read_text())
        lanes = negative_lanes(json.loads(export_path.read_text()), manifest)
        if not lanes:
            continue

        run_dir = Path(manifest["run"])
        samples = load_track_boxes(run_dir / "tracks.json")
        by_frame: dict[int, list] = {}
        kept_lanes = 0
        for lane in lanes:
            got = [(f, b) for f, b in samples.get(lane["track_id"], [])
                   if lane["first_frame"] <= f <= lane["last_frame"]]
            if not got:
                continue
            step = max(1, len(got) // max_samples_per_lane)
            for f, b in got[::step][:max_samples_per_lane]:
                by_frame.setdefault(int(f), []).append(b)
            kept_lanes += 1

        reader = VideoReader(Path(manifest["video"]))
        n_before = len(all_crops)
        try:
            for frame_no, frame in reader.read_frames(sorted(by_frame)):
                for bbox in by_frame[frame_no]:
                    crop = crop_player(frame, bbox)
                    if crop is not None:
                        all_crops.append(crop)
        finally:
            reader.close()
        print(f"  {export_path}: {kept_lanes} 'not ours' lanes "
              f"→ {len(all_crops) - n_before} crops")

    if not all_crops:
        return np.zeros((0, 512), dtype=np.float32)
    return embedder.embed(all_crops)


def sample_query_lanes(run: Path, *, n_windows: int, window_s: float,
                       min_span_s: float, max_crops: int):
    """Lanes alive in a spread of windows, with their crops planned per frame.

    Sampling *windows* rather than lanes keeps the video read to a handful of
    seeks: the whole point is to compare two scoring rules, which needs a few
    hundred honest queries rather than all 17k lanes.
    """
    from soccer_vision.clips.halo import load_track_boxes

    tracks = json.loads((run / "tracks.json").read_text())
    fps = tracks["fps"]
    samples = load_track_boxes(run / "tracks.json")
    teams = tracks.get("teams") or {}

    last = max(f for s in samples.values() for f, _ in s)
    starts = np.linspace(0.08 * last, 0.92 * last, n_windows).astype(int)
    win = int(window_s * fps)

    lanes = {}
    for tid, s in samples.items():
        if not s:
            continue
        f0, f1 = s[0][0], s[-1][0]
        if (f1 - f0) / fps < min_span_s:
            continue
        for w in starts:
            got = [(f, b) for f, b in s if w <= f < w + win]
            if len(got) < 2:
                continue
            step = max(1, len(got) // max_crops)
            kit = teams.get(str(tid))
            kit = kit.get("team") if isinstance(kit, dict) else kit
            lanes[tid] = dict(
                tid=int(tid), boxes=got[::step][:max_crops],
                foot=float(np.median([b[3] for _, b in got])),
                span=float((f1 - f0) / fps), kit=kit,
            )
            break
    return lanes, fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--gallery", type=Path,
                    default=Path("galleries/saints-u14g.fullmatch.npz"))
    ap.add_argument("--n-windows", type=int, default=10)
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--min-span-s", type=float, default=1.0)
    ap.add_argument("--max-crops", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--negatives", type=Path, default=None,
                    help="npz of pre-embedded negatives (from "
                         "harvest_offpitch_negatives.py). Replaces the tracklet "
                         "'not ours' lanes, which are all on-pitch opponents.")
    args = ap.parse_args()

    from soccer_vision.identify.gallery import build_gallery, load_gallery, match_track
    from soccer_vision.identify.reid import ReIDEmbedder, crop_player
    from soccer_vision.io.video import VideoReader

    print("Loading re-id backbone...")
    embedder = ReIDEmbedder.from_pretrained(device=args.device)

    if args.negatives:
        with np.load(args.negatives) as z:
            neg = np.asarray(z["emb"], dtype=np.float32)
        print(f"\nnegatives from {args.negatives}: {len(neg)} exemplars")
    else:
        print("\nEmbedding 'not ours' lanes:")
        neg = embed_negatives(embedder)
    print(f"  total negative exemplars: {len(neg)}")
    if len(neg) == 0:
        raise SystemExit("no negatives found — nothing to test")

    base = load_gallery(args.gallery)
    print(f"\nbase gallery: {len(base['emb'])} exemplars over "
          f"{len(base['names'])} players")

    # Two variants: the class capped like a player (64), and uncapped. Cap size is
    # a real confound — `match_track` averages a crop's top-3 exemplars, so a
    # bigger class finds 3 close neighbours more easily whatever it contains.
    base_names = [base["names"][i] for i in base["label"]]
    variants = {}
    for cap_label, cap in (("neg@64", 64), ("neg@all", max(64, len(neg)))):
        variants[cap_label] = build_gallery(
            np.concatenate([base["emb"], neg]),
            base_names + [NOT_OURS] * len(neg),
            max_per_player=cap,
        )
        n_neg = int((np.asarray(variants[cap_label]["label"])
                     == variants[cap_label]["names"].index(NOT_OURS)).sum())
        print(f"  {cap_label}: {len(variants[cap_label]['emb'])} exemplars "
              f"({n_neg} negative)")

    print("\nSampling query lanes...")
    lanes, fps = sample_query_lanes(
        args.run, n_windows=args.n_windows, window_s=args.window_s,
        min_span_s=args.min_span_s, max_crops=args.max_crops)
    print(f"  {len(lanes)} lanes")

    by_frame: dict[int, list] = {}
    for lane in lanes.values():
        for f, b in lane["boxes"]:
            by_frame.setdefault(int(f), []).append((lane["tid"], b))

    crops: dict[int, list] = {}
    reader = VideoReader(args.run / "broadcast_proxy.mp4")
    try:
        for frame_no, frame in reader.read_frames(sorted(by_frame)):
            for tid, bbox in by_frame[frame_no]:
                crop = crop_player(frame, bbox)
                if crop is not None:
                    crops.setdefault(tid, []).append(crop)
    finally:
        reader.close()
    print(f"  {sum(len(v) for v in crops.values())} crops from "
          f"{len(by_frame)} frames")

    rows = []
    for tid, cs in crops.items():
        emb = embedder.embed(cs)
        lane = lanes[tid]
        r = dict(tid=tid, foot=lane["foot"], span=lane["span"],
                 kit=lane["kit"], n_crops=len(cs))
        m = match_track(emb, base)
        r["base_name"], r["base_sim"], r["base_margin"] = m.name, m.similarity, m.margin
        for label, g in variants.items():
            m = match_track(emb, g)
            r[f"{label}_name"] = m.name
            r[f"{label}_sim"] = m.similarity
            r[f"{label}_margin"] = m.margin
        rows.append(r)

    out = args.out or (args.run / "negative_class_eval.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")
    report(rows)


def report(rows: list[dict]) -> None:
    H = 1080
    off = [r for r in rows if r["foot"] / H < 0.40]
    on = [r for r in rows if r["foot"] / H >= 0.45]
    print(f"\nlanes: {len(off)} above 0.40H (spectator-rich), "
          f"{len(on)} below 0.45H (pitch), {len(rows)} total")
    print("(foot-y is a proxy for on/off pitch, not ground truth — "
          "hand-verify before trusting the split)\n")

    def named(group, key):
        return sum(1 for r in group if r[key] and r[key] != NOT_OURS)

    print(f"{'scoring':>28} {'names off-pitch':>16} {'names on-pitch':>15}")
    print(f"{'baseline (11 players)':>28} "
          f"{named(off,'base_name'):>7}/{len(off):<8} "
          f"{named(on,'base_name'):>7}/{len(on):<7}")
    for label in ("neg@64", "neg@all"):
        print(f"{'+ not-ours class ' + label:>28} "
              f"{named(off, label + '_name'):>7}/{len(off):<8} "
              f"{named(on, label + '_name'):>7}/{len(on):<7}")
    for floor in (0.70, 0.75, 0.80):
        o = sum(1 for r in off if r["base_name"] and r["base_sim"] >= floor)
        n = sum(1 for r in on if r["base_name"] and r["base_sim"] >= floor)
        print(f"{'min_similarity ' + str(floor):>28} {o:>7}/{len(off):<8} {n:>7}/{len(on):<7}")

    for label in ("neg@64", "neg@all"):
        rej_off = sum(1 for r in off if r[label + "_name"] == NOT_OURS)
        rej_on = sum(1 for r in on if r[label + "_name"] == NOT_OURS)
        print(f"\n{label}: 'not ours' won on {rej_off}/{len(off)} off-pitch, "
              f"{rej_on}/{len(on)} on-pitch lanes")


if __name__ == "__main__":
    main()
