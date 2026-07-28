"""Cache the exemplar embeddings, then sweep min_margin on leave-one-frame-out.

The question a single threshold can't answer: are the abstentions a tuning
choice, or is the embedding simply not separating these players? Dropping the
margin to 0 turns matching into pure nearest-neighbour with no abstention — the
accuracy there is the raw discriminability, and no threshold can beat it.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from soccer_vision.identify.gallery import build_gallery, match_track

SCRATCH = Path("/tmp/claude-4736/-orange-ewhite-b-weinstein-soccer-video-analysis/46a265f2-4403-4f29-a0b2-8f2e504f6b15/scratchpad")
CACHE = SCRATCH / "u14g_exemplars.npz"
ROOT = "/orange/ewhite/b.weinstein/soccer-video-analysis"

if not CACHE.exists():
    import cv2
    from soccer_vision.cli.enroll import named_boxes
    from soccer_vision.identify.enroll import boxes_from_label_studio
    from soccer_vision.identify.reid import ReIDEmbedder, crop_player
    from soccer_vision.profiles.loader import load_profile

    export = json.loads(open(f"{ROOT}/runs/u14g_label_frames/annotations.json").read())
    profile = load_profile(f"{ROOT}/examples/profiles/saints-u14g.yaml")
    named, _ = named_boxes(boxes_from_label_studio(export), profile)
    by_frame = defaultdict(list)
    for f, b, n in named:
        by_frame[int(f)].append((b, n))
    print("Embedding...", flush=True)
    embedder = ReIDEmbedder.from_pretrained()
    E, F, N = [], [], []
    for frame_no in sorted(by_frame):
        img = cv2.imread(f"{ROOT}/runs/u14g_label_frames/frames/{frame_no:06d}.jpg")
        crops, names = [], []
        for bbox, name in by_frame[frame_no]:
            c = crop_player(img, bbox)
            if c is not None:
                crops.append(c); names.append(name)
        if crops:
            for e, n in zip(embedder.embed(crops), names):
                E.append(e); F.append(frame_no); N.append(n)
    np.savez(CACHE, emb=np.array(E), frame=np.array(F), name=np.array(N))

d = np.load(CACHE, allow_pickle=True)
emb, frame, name = d["emb"], d["frame"], d["name"]
print(f"{len(emb)} exemplars, {len(set(name))} players, {len(set(frame))} frames\n")

def loo(min_margin, min_similarity=0.5):
    res, errors = Counter(), []
    for held in sorted(set(frame)):
        tr = frame != held
        g = build_gallery(emb[tr], list(name[tr]))
        for truth in sorted(set(name[frame == held])):
            if truth not in g["names"]:
                res["untestable"] += 1
                continue
            m = match_track(emb[(frame == held) & (name == truth)], g,
                            min_similarity=min_similarity, min_margin=min_margin)
            if m.name is None:
                res["abstain"] += 1
            elif m.name == truth:
                res["correct"] += 1
            else:
                res["wrong"] += 1
                errors.append((held, truth, m.name, round(m.margin, 3)))
    return res, errors

print(f"{'min_margin':>10} {'testable':>9} {'named':>7} {'correct':>8} {'wrong':>6}  precision")
for mm in (0.0, 0.02, 0.05, 0.10):
    res, errors = loo(mm)
    testable = res["correct"] + res["wrong"] + res["abstain"]
    named_n = res["correct"] + res["wrong"]
    prec = f"{res['correct']}/{named_n}" if named_n else "-"
    print(f"{mm:>10.2f} {testable:>9} {named_n:>7} {res['correct']:>8} {res['wrong']:>6}  {prec}")

res, errors = loo(0.0)
print(f"\n({res['untestable']} player-frames untestable: that player appears in no other frame)")
print("\nNearest-neighbour confusions with no abstention (min_margin=0):")
for held, truth, pred, mar in errors:
    print(f"  frame {held:>6}: {truth:<22} named as {pred:<22} (margin {mar})")
