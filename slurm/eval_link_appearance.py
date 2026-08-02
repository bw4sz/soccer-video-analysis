"""Can an appearance check reject a link whose true continuation isn't there?

The split-lane benchmark flatters the linker: it always *has* a right answer in
range. Real lanes often die with no continuation at all — the player walks out of
frame, or is occluded past the gate. Simulating that (cut a 3.0 s hole and use a
3.0 s gate, so the true tail is just out of reach) the geometry-only linker made
**150 wrong links out of 400 heads instead of abstaining**. Geometry cannot tell
"my player kept running" from "a different player ran through that spot".

Appearance should, and for a reason the gallery work doesn't contradict: naming a
teammate out of 11 needs the embedding to separate people in identical kit, but
this only needs "same person, 0.6 s later, same pose, same light, same patch of
turf" — a far easier question on the same 2.2M-parameter backbone.

Two things get measured on the same links:

* **absent-truth** (3.0 s hole, 3.0 s gate) — every link made is wrong, so any
  rejection is a win. This is the number that matters.
* **present-truth** (0.6 s hole) — rejections here are the cost, since the
  correct link is available and a veto that kills it loses real yield.

    python slurm/eval_link_appearance.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from soccer_vision.identify.reid import ReIDEmbedder, crop_player
from soccer_vision.io.video import VideoReader
from soccer_vision.tracking.link import LinkConfig, link_tracks

ROOT = Path("/orange/ewhite/b.weinstein/soccer-video-analysis")
run = ROOT / "runs/saints-u14g-full"
SCRATCH = Path("/tmp/claude-4736/-orange-ewhite-b-weinstein-soccer-video-analysis/"
               "6e252678-ff52-4711-8623-e029d002e9ea/scratchpad")

doc = json.loads((run / "tracks.json").read_text())
tracks, teams = doc["tracks"], doc.get("teams", {})
fps, interval = float(doc["fps"]), int(doc["sample_interval"])
eligible = sorted((t for t, v in tracks.items() if len(v) >= 20),
                  key=lambda t: -len(tracks[t]))[:400]


def build(gap_samples):
    out, kits, truth = {}, dict(teams), {}
    for tid, s in tracks.items():
        if tid not in eligible:
            out[tid] = s
            continue
        mid = len(s) // 2
        head, tail = s[:mid], s[mid + gap_samples:]
        if len(head) < 2 or len(tail) < 2:
            out[tid] = s
            continue
        h, t = f"{tid}_h", f"{tid}_t"
        out[h], out[t] = head, tail
        truth[h] = t
        if tid in kits:
            kits[h] = kits[t] = kits.pop(tid)
    return {**doc, "tracks": out, "teams": kits}, truth


CFG = LinkConfig(max_gap_s=3.0, max_dist_px=250.0, use_motion=True,
                 bidirectional=True, global_assignment=False)

cases = {}
for label, gap in [("absent-truth (3.0s hole)", 15), ("present-truth (0.6s hole)", 3)]:
    synth, truth = build(gap)
    res = link_tracks(synth, CFG)
    links = [l for l in res.links if l.a in truth]
    cases[label] = (synth, truth, links)
    print(f"{label}: {len(links)} links made on {len(truth)} split heads")

# --- embed the two crops each link is judged on, in one forward pass ---
need = {}   # (tid_in_synth, "end"|"start") -> (frame, bbox), per case
wanted = []
for label, (synth, truth, links) in cases.items():
    for l in links:
        a_s, b_s = synth["tracks"][l.a][-1], synth["tracks"][l.b][0]
        wanted.append((label, l.a, l.b, a_s, b_s))

frames_needed = sorted({s["frame"] for _, _, _, a, b in
                        [(w[0], w[1], w[2], w[3], w[4]) for w in wanted]
                        for s in (a, b)})
print(f"\n{len(wanted)} links -> {len(frames_needed)} distinct frames to decode")

cache = SCRATCH / "link_appearance.npz"
if cache.exists():
    d = np.load(cache, allow_pickle=True)
    sims = {tuple(k.split("|")): float(v) for k, v in
            zip(d["keys"], d["sims"])}
else:
    reader = VideoReader(run / "broadcast_proxy.mp4")
    embedder = ReIDEmbedder.from_pretrained()
    by_frame = {}
    for label, a, b, a_s, b_s in wanted:
        by_frame.setdefault(a_s["frame"], []).append((label, a, b, "a", a_s["bbox"]))
        by_frame.setdefault(b_s["frame"], []).append((label, a, b, "b", b_s["bbox"]))
    crops, tags = [], []
    for fno, frame in reader.read_frames(frames_needed):
        for label, a, b, side, bbox in by_frame.get(fno, []):
            c = crop_player(frame, np.array(bbox, dtype=float))
            if c is not None:
                crops.append(c)
                tags.append((label, a, b, side))
    print(f"embedding {len(crops)} crops...")
    emb = embedder.embed(crops)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    slot = {}
    for e, (label, a, b, side) in zip(emb, tags):
        slot[(label, a, b, side)] = e
    sims = {}
    for label, a, b, _, _ in wanted:
        ea, eb = slot.get((label, a, b, "a")), slot.get((label, a, b, "b"))
        if ea is not None and eb is not None:
            sims[(label, a, b)] = float(ea @ eb)
    np.savez(cache, keys=np.array(["|".join(k) for k in sims]),
             sims=np.array(list(sims.values())))

print(f"\n{'threshold':>10} " + "  ".join(f"{lab:>34}" for lab in cases))
print(f"{'':>10} " + "  ".join(f"{'kept':>10}{'(wrong)':>12}{'(right)':>12}"
                               for _ in cases))
for thr in (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
    row = []
    for label, (synth, truth, links) in cases.items():
        kept = wrong = right = 0
        for l in links:
            s = sims.get((label, l.a, l.b))
            if s is not None and s < thr:
                continue
            kept += 1
            if truth.get(l.a) == l.b:
                right += 1
            else:
                wrong += 1
        row.append(f"{kept:>10}{wrong:>12}{right:>12}")
    print(f"{thr:>10.2f} " + "  ".join(row))
