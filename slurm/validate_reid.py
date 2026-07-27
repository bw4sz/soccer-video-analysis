"""Leave-one-track-out check: does the re-id gallery name a *new* lane correctly?

The question that matters isn't whether a track matches itself — it's whether a
lane the gallery has never seen gets the right player. So each eligible track is
held out in turn, a gallery is built from the *other* tracks only, and the held
lane is matched against it. That is exactly the production case: enrol from past
matches, name a fresh lane today.

Labels are the high-confidence jersey votes already in ``jerseys.json`` — a
proxy for ground truth, not ground truth (an over-confident OCR read is a wrong
label here). Jersey 1 is excluded because it's this recognizer's known
hallucination class on Veo footage.

Embeddings are cached to ``--cache`` so threshold sweeps don't re-decode video.

    python slurm/validate_reid.py --run runs/saints-u11-sam3-full
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccer_vision.clips.halo import load_track_boxes  # noqa: E402
from soccer_vision.identify.enroll import names_from_jerseys  # noqa: E402
from soccer_vision.identify.gallery import build_gallery, match_track  # noqa: E402


def embed_run(run_dir: Path, track_names: dict[int, str], max_samples: int, device):
    from soccer_vision.identify.reid import ReIDEmbedder, embed_tracks
    from soccer_vision.io.video import VideoReader

    embedder = ReIDEmbedder.from_pretrained(device=device)
    reader = VideoReader(run_dir / "broadcast_proxy.mp4")
    try:
        return embed_tracks(
            load_track_boxes(run_dir / "tracks.json"), embedder, reader,
            max_samples_per_track=max_samples,
            track_ids=set(track_names),
            progress=True,
        )
    finally:
        reader.close()


def leave_one_track_out(per_track, track_names, *, min_similarity, min_margin):
    """Match each track against a gallery built from every *other* track."""
    # Only numbers with >=2 tracks can be tested: hold one out and the player
    # must still be in the gallery.
    counts = Counter(track_names[t] for t in per_track)
    eligible = [t for t in per_track if counts[track_names[t]] >= 2]

    rows = []
    for held in eligible:
        others = [t for t in per_track if t != held]
        gallery = build_gallery(
            np.concatenate([per_track[t] for t in others]),
            [track_names[t] for t in others for _ in range(len(per_track[t]))],
        )
        m = match_track(per_track[held], gallery,
                        min_similarity=min_similarity, min_margin=min_margin)
        rows.append((held, track_names[held], m.name, m.similarity, m.margin))
    return rows


def stranger_rejection(per_track, track_names, *, min_similarity, min_margin):
    """Do players the gallery never enrolled correctly come back unnamed?

    Leave-one-track-out only ever asks about players who *are* in the gallery, but
    in production most tracks are strangers — the opposition, referees, anyone
    unenrolled — and the whole reid+ocr fallback rests on those abstaining rather
    than being forced onto the nearest enrolled player. Numbers seen on exactly one
    track give us real strangers for free: hold that number out of the gallery
    entirely and the correct answer is ``None``.
    """
    counts = Counter(track_names[t] for t in per_track)
    strangers = [t for t in per_track if counts[track_names[t]] == 1]
    if not strangers:
        return None

    enrolled = [t for t in per_track if counts[track_names[t]] >= 2]
    gallery = build_gallery(
        np.concatenate([per_track[t] for t in enrolled]),
        [track_names[t] for t in enrolled for _ in range(len(per_track[t]))],
    )
    matches = [match_track(per_track[t], gallery,
                           min_similarity=min_similarity, min_margin=min_margin)
               for t in strangers]
    rejected = [m for m in matches if m.name is None]
    print(f"  strangers rejected {len(rejected)}/{len(matches)} "
          f"(false names: {[m.name for m in matches if m.name]})")
    return len(rejected) / len(matches)


def report(rows, label):
    named = [r for r in rows if r[2] is not None]
    correct = [r for r in named if r[2] == r[1]]
    n = len(rows)
    print(f"\n--- {label} ---")
    print(f"  tracks tested   {n}")
    print(f"  named           {len(named)}/{n} ({len(named)/n:.0%})  "
          f"[abstained {n - len(named)}]")
    if named:
        print(f"  correct|named   {len(correct)}/{len(named)} ({len(correct)/len(named):.0%})")
    print(f"  correct|all     {len(correct)}/{n} ({len(correct)/n:.0%})")

    # Per-player recall, and the majority-class baseline this has to beat.
    truth = Counter(r[1] for r in rows)
    baseline = max(truth.values()) / n
    print(f"  majority baseline {baseline:.0%} (always predict {truth.most_common(1)[0][0]})")
    per = Counter(r[1] for r in correct)
    print("  per player (correct/total):")
    for name, total in sorted(truth.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<8} {per.get(name, 0):>2}/{total:<3}")
    return len(correct) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", default=None,
                    help="Embedding cache npz (default: <run>/reid_embeddings.npz)")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--min-confidence", type=float, default=0.8)
    ap.add_argument("--min-obs", type=int, default=5)
    ap.add_argument("--exclude-jersey", nargs="+", type=int, default=[1])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cache = Path(args.cache) if args.cache else run_dir / "reid_embeddings.npz"

    doc = json.loads((run_dir / "jerseys.json").read_text())
    track_names = names_from_jerseys(
        doc, None,
        min_confidence=args.min_confidence,
        min_obs=args.min_obs,
        exclude=set(args.exclude_jersey),
    )
    print(f"High-confidence OCR tracks: {len(track_names)} "
          f"across {len(set(track_names.values()))} numbers")

    if cache.exists():
        with np.load(cache) as z:
            per_track = {int(k): z[k] for k in z.files}
        print(f"Loaded cached embeddings for {len(per_track)} tracks ({cache})")
    else:
        per_track = embed_run(run_dir, track_names, args.max_samples, args.device)
        np.savez_compressed(cache, **{str(k): v for k, v in per_track.items()})
        print(f"Embedded {len(per_track)} tracks → {cache}")

    per_track = {t: e for t, e in per_track.items() if t in track_names}

    # Sweep the abstain threshold: it is the one knob that trades coverage for
    # precision, and where it should sit is what this run is meant to answer.
    # Sweep both guards. Coverage and precision among enrolled players is one
    # half; rejecting strangers is the other, and they pull in opposite
    # directions, so the defaults have to be chosen against both at once.
    for min_sim, min_margin in [(0.0, 0.05), (0.5, 0.05), (0.7, 0.05),
                                (0.5, 0.1), (0.5, 0.15), (0.7, 0.15)]:
        report(
            leave_one_track_out(per_track, track_names,
                                min_similarity=min_sim, min_margin=min_margin),
            f"min_similarity={min_sim}, min_margin={min_margin}",
        )
        stranger_rejection(per_track, track_names,
                           min_similarity=min_sim, min_margin=min_margin)


if __name__ == "__main__":
    main()
