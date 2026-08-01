"""Did spreading tracklet windows across the whole match help?

The first tracklet batch came from one 3-minute stretch of `u14g_smoke180.mp4`
and moved leave-one-frame-out not at all (19/45 -> 17/45). The diagnosis was
that the crops carried one sun angle and one patch of pitch, so they added
exemplars but almost no *views*. This is the test of that diagnosis: the same
labelling workflow, but 12 windows spread evenly across the full 60-minute match.

Query is held constant across every row: one labelled frame held out of the
u14g_label_frames export. Only the gallery changes —

  A: the other frames only                          (the 42% baseline)
  B: + the smoke-clip tracklet crops                (what we measured before)
  C: + the full-match tracklet crops                (the new batch)
  D: + both

Leakage guard: the frame sampler and the window sampler walked the same match,
so a couple of windows open within a second of a query frame (49236 vs a window
at 49258, 98472 vs 98515). A crop 0.7 s from the query is the same pose in the
same light, which flatters the gallery in a way a future match never will. Every
row is therefore reported twice: all crops, and again with tracklet crops within
--guard-s of the held-out frame dropped from the gallery.
"""
import argparse
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
               "6e252678-ff52-4711-8623-e029d002e9ea/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--guard-s", type=float, default=5.0)
ap.add_argument("--max-samples", type=int, default=20)
ap.add_argument("--min-margin", type=float, default=0.05)
args = ap.parse_args()

profile = load_profile(ROOT / "examples/profiles/saints-u14g.yaml")
embedder = None


def get_embedder():
    global embedder
    if embedder is None:
        embedder = ReIDEmbedder.from_pretrained()
    return embedder


# --- query side: the labelled frames, embedded off the JPEGs beside the export ---
qcache = SCRATCH / "u14g_query_emb.npz"
if qcache.exists():
    d = np.load(qcache, allow_pickle=True)
    q_emb, q_frame, q_name = d["emb"], d["frame"], d["name"]
else:
    export = json.loads((ROOT / "runs/u14g_label_frames/annotations.json").read_text())
    named, _ = named_boxes(boxes_from_label_studio(export), profile)
    by_frame = defaultdict(list)
    for f, b, n in named:
        by_frame[int(f)].append((b, n))
    E, F, N = [], [], []
    for frame_no in sorted(by_frame):
        img = cv2.imread(str(ROOT / f"runs/u14g_label_frames/frames/{frame_no:06d}.jpg"))
        if img is None:
            continue
        crops, names = [], []
        for bbox, name in by_frame[frame_no]:
            c = crop_player(img, bbox)
            if c is not None:
                crops.append(c), names.append(name)
        if crops:
            for e, n in zip(get_embedder().embed(crops), names):
                E.append(e), F.append(frame_no), N.append(n)
    q_emb, q_frame, q_name = np.array(E), np.array(F), np.array(N)
    np.savez(qcache, emb=q_emb, frame=q_frame, name=q_name)

print(f"query pool: {len(q_emb)} frame crops, {len(set(q_name))} players, "
      f"{len(set(q_frame))} frames\n")


def tracklet_embeddings(tag, export_p, manifest_p, tracks_p, video_p):
    """Embed one tracklet batch, keeping the source frame of every crop."""
    cache = SCRATCH / f"u14g_tl_{tag}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["emb"], d["name"], d["frame"]
    export = json.loads(Path(export_p).read_text())
    manifest = json.loads(Path(manifest_p).read_text())
    tracks = load_track_boxes(tracks_p)
    boxes, _ = boxes_from_tracklet_export(export, manifest, tracks,
                                          max_samples_per_lane=args.max_samples)
    boxes = [(f, b, roster_full_name(profile, n)) for f, b, n in boxes]
    by_frame = defaultdict(list)
    for f, b, n in boxes:
        by_frame[int(f)].append((b, n))
    print(f"[{tag}] {len(boxes)} crops over "
          f"{len(set(n for _, _, n in boxes))} players, "
          f"{len(by_frame)} distinct frames — embedding...", flush=True)
    cap = cv2.VideoCapture(str(video_p))
    crops, names, frames = [], [], []
    for f in sorted(by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        for b, n in by_frame[f]:
            c = crop_player(img, b)
            if c is not None:
                crops.append(c), names.append(n), frames.append(f)
    cap.release()
    emb = get_embedder().embed(crops)
    name, frame = np.array(names), np.array(frames)
    np.savez(cache, emb=emb, name=name, frame=frame)
    return emb, name, frame


smoke = tracklet_embeddings(
    "smoke",
    ROOT / "runs/u14g_tracklets/annotations.json",
    ROOT / "runs/u14g_tracklets/tracklets.json",
    ROOT / "runs/u14g-smoke-teamfix/tracks.json",
    ROOT / "runs/u14g-smoke-teamfix/broadcast_proxy.mp4",
)
full = tracklet_embeddings(
    "fullmatch",
    ROOT / "runs/saints-u14g-full/tracklets/annotations.json",
    ROOT / "runs/saints-u14g-full/tracklets/tracklets.json",
    ROOT / "runs/saints-u14g-full/tracks.json",
    ROOT / "runs/saints-u14g-full/broadcast_proxy.mp4",
)

FPS = 29.97
both = (np.concatenate([smoke[0], full[0]]),
        np.concatenate([smoke[1], full[1]]),
        np.concatenate([smoke[2], full[2]]))


def loo(extra=None, guard_s=None, min_margin=0.05):
    """Leave-one-frame-out. `guard_s` drops extra crops near the held-out frame.

    Only the full-match batch shares a timeline with the query frames, so the
    guard is applied by frame number and the smoke batch (a different video,
    different frame numbering) is never near-duplicate by construction.
    """
    res, rank1_hit, rank1_n, dropped = Counter(), 0, 0, 0
    for held in sorted(set(q_frame)):
        keep = q_frame != held
        g_emb, g_name = q_emb[keep], list(q_name[keep])
        if extra is not None and len(extra[0]):
            e_emb, e_name, e_frame = extra
            if guard_s is not None:
                far = np.abs(e_frame - held) > guard_s * FPS
                dropped += int((~far).sum())
                e_emb, e_name = e_emb[far], e_name[far]
            g_emb = np.concatenate([g_emb, e_emb])
            g_name = g_name + list(e_name)
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
    return res, rank1_hit, rank1_n, dropped


rows = [("A: frames only", None),
        ("B: + smoke-clip tracklets", smoke),
        ("C: + full-match tracklets", full),
        ("D: + both tracklet batches", both)]

for guard in (None, args.guard_s):
    head = "all crops" if guard is None else f"guard {guard:g}s around held-out frame"
    print(f"\n=== {head} ===")
    print(f"{'gallery':<30} {'exemplars':>9} {'rank-1 (no abstention)':>23} "
          f"{'named':>6} {'precision':>10}")
    for label, extra in rows:
        res, hit, tot, dropped = loo(extra, guard_s=guard,
                                     min_margin=args.min_margin)
        n_ex = len(q_emb) - 0 if extra is None else len(q_emb) + len(extra[0])
        named = res["correct"] + res["wrong"]
        prec = f"{res['correct']}/{named}" if named else "-"
        note = f"  (-{dropped // max(len(set(q_frame)), 1)}/frame guarded)" if dropped else ""
        print(f"{label:<30} {n_ex:>9} {hit:>7}/{tot} ({100 * hit / tot:>3.0f}%)"
              f"      {named:>6} {prec:>10}{note}")

# per-player exemplar balance, the thing that bit last time
print("\nexemplar counts per player (frames / +smoke / +fullmatch):")
players = sorted(set(q_name))
for p in players:
    print(f"  {p:<24} {int((q_name == p).sum()):>3} "
          f"{int((smoke[1] == p).sum()):>4} {int((full[1] == p).sum()):>5}")
extra_names = sorted(set(full[1]) - set(players))
for p in extra_names:
    print(f"  {p + ' (not in query set)':<24} {'-':>3} "
          f"{int((smoke[1] == p).sum()):>4} {int((full[1] == p).sum()):>5}")
