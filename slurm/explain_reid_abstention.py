"""Print the full scoring for the three crops in question.

The match sheet shows the top-3 *exemplars* (individual crops). match_track
ranks *players*, scoring each as the mean of her top_k=3 best exemplars. Those
are different rankings, and the gap between them is the whole question.
"""
from pathlib import Path

import numpy as np

from soccer_vision.identify.gallery import build_gallery, match_track

SCRATCH = Path("/tmp/claude-4736/-orange-ewhite-b-weinstein-soccer-video-analysis/46a265f2-4403-4f29-a0b2-8f2e504f6b15/scratchpad")
d = np.load(SCRATCH / "u14g_exemplars.npz", allow_pickle=True)
emb, frame, name = d["emb"], d["frame"], d["name"]

def explain(held, truth, top_k=3):
    tr = frame != held
    gi = np.where(tr)[0]
    g = build_gallery(emb[tr], list(name[tr]))
    i = np.where((frame == held) & (name == truth))[0][0]
    sims = emb[tr] @ emb[i]

    print(f"\n=== frame {held}, query = {truth} " + "=" * 30)
    order = np.argsort(-sims)
    print("  nearest EXEMPLARS (what the match sheet shows):")
    for r in order[:5]:
        print(f"    {sims[r]:.3f}  {name[gi[r]]}")

    print(f"  player SCORES (what match_track ranks: mean of top-{top_k} exemplars):")
    scored = []
    for p in sorted(set(name[tr])):
        s = np.sort(sims[name[tr] == p])[::-1][:top_k]
        scored.append((float(s.mean()), p, s, (name[tr] == p).sum()))
    scored.sort(reverse=True)
    for k, (score, p, s, n_ex) in enumerate(scored[:4]):
        mark = "<-- truth" if p == truth else ""
        used = ", ".join(f"{v:.3f}" for v in s)
        print(f"    {score:.3f}  {p:<20} (has {n_ex} exemplars; used {used}) {mark}")

    m = match_track(emb[i:i + 1], g)
    print(f"  margin = {scored[0][0]:.3f} - {scored[1][0]:.3f} = {scored[0][0]-scored[1][0]:.3f}"
          f"   -> needs >= 0.05 -> {'NAMED ' + str(m.name) if m.name else 'ABSTAIN'}")

explain(8952, "Leire Cabral")
explain(8952, "Morrighan Wright")
explain(4476, "Catherine Conroy")
