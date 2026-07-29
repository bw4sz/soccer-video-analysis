"""Did the tracklet crops help? A/B on the identical held-out query set.

Query is always the same: one labelled frame held out of the u14g_label_frames
export. Only the gallery changes —

  A: the other frames only                      (what we measured at 42%)
  B: the other frames + every tracklet crop     (what we just enrolled)

Anything else held constant, so the difference is the tracklet data and nothing
else.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from soccer_vision.annotate.tracklets import boxes_from_tracklet_export
from soccer_vision.cli.enroll import named_boxes, roster_full_name
from soccer_vision.clips.halo import load_track_boxes
from soccer_vision.identify.enroll import boxes_from_label_studio
from soccer_vision.identify.gallery import build_gallery, match_track
from soccer_vision.identify.reid import ReIDEmbedder, crop_player
from soccer_vision.profiles.loader import load_profile

ROOT = Path("/orange/ewhite/b.weinstein/soccer-video-analysis")
SCRATCH = Path("/tmp/claude-4736/-orange-ewhite-b-weinstein-soccer-video-analysis/"
               "46a265f2-4403-4f29-a0b2-8f2e504f6b15/scratchpad")
profile = load_profile(ROOT / "examples/profiles/saints-u14g.yaml")

# --- query side: the frame crops, cached from the earlier run ---
d = np.load(SCRATCH / "u14g_exemplars.npz", allow_pickle=True)
q_emb, q_frame, q_name = d["emb"], d["frame"], d["name"]
print(f"query pool: {len(q_emb)} frame crops, {len(set(q_name))} players")

# --- gallery side B: tracklet crops ---
export = json.loads((ROOT / "runs/u14g_tracklets/annotations.json").read_text())
manifest = json.loads((ROOT / "runs/u14g_tracklets/tracklets.json").read_text())
tracks = load_track_boxes(ROOT / "runs/u14g-smoke-teamfix/tracks.json")
tl_boxes, summary = boxes_from_tracklet_export(export, manifest, tracks,
                                               max_samples_per_lane=20)
tl_boxes = [(f, b, roster_full_name(profile, n)) for f, b, n in tl_boxes]
print(f"tracklet crops: {len(tl_boxes)} over {len(set(n for _, _, n in tl_boxes))} players")

cache = SCRATCH / "u14g_tracklet_emb.npz"
if cache.exists():
    td = np.load(cache, allow_pickle=True)
    t_emb, t_name = td["emb"], td["name"]
else:
    embedder = ReIDEmbedder.from_pretrained()
    cap = cv2.VideoCapture(str(ROOT / "runs/u14g-smoke-teamfix/broadcast_proxy.mp4"))
    by_frame = defaultdict(list)
    for f, b, n in tl_boxes:
        by_frame[int(f)].append((b, n))
    crops, names = [], []
    for f in sorted(by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        for b, n in by_frame[f]:
            c = crop_player(frame, b)
            if c is not None:
                crops.append(c); names.append(n)
    cap.release()
    t_emb = embedder.embed(crops)
    t_name = np.array(names)
    np.savez(cache, emb=t_emb, name=t_name)
print(f"tracklet embeddings: {len(t_emb)}\n")


def loo(extra_emb=None, extra_name=None, min_margin=0.05):
    res, rank1_hit, rank1_n = Counter(), 0, 0
    for held in sorted(set(q_frame)):
        keep = q_frame != held
        g_emb, g_name = q_emb[keep], list(q_name[keep])
        if extra_emb is not None and len(extra_emb):
            g_emb = np.concatenate([g_emb, extra_emb])
            g_name = g_name + list(extra_name)
        g = build_gallery(g_emb, g_name, max_per_player=64)
        for truth in sorted(set(q_name[q_frame == held])):
            if truth not in g["names"]:
                continue
            query = q_emb[(q_frame == held) & (q_name == truth)]
            rank1_n += 1
            rank1_hit += (match_track(query, g, min_margin=0.0,
                                      min_similarity=0.0).name == truth)
            m = match_track(query, g, min_margin=min_margin)
            res["abstain" if m.name is None else
                ("correct" if m.name == truth else "wrong")] += 1
    return res, rank1_hit, rank1_n


print(f"{'gallery':<34} {'rank-1 (no abstention)':>23} {'named@0.05':>11} {'precision':>10}")
for label, e, n in [("A: frames only", None, None),
                    ("B: frames + tracklet crops", t_emb, t_name)]:
    res, hit, tot = loo(e, n)
    named = res["correct"] + res["wrong"]
    prec = f"{res['correct']}/{named}" if named else "-"
    print(f"{label:<34} {hit:>7}/{tot} ({100 * hit / tot:>3.0f}%)      "
          f"{named:>11} {prec:>10}")
