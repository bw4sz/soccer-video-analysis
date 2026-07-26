"""RF-DETR vs SAM3 ball track, same frames, same scale."""
from __future__ import annotations
import json
from pathlib import Path
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

RUN = Path("/orange/ewhite/b.weinstein/soccer-video-analysis/runs/sam3-ball")
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
S1, S2 = "#2a78d6", "#eb6834"
TP, TS = "#0b0b0b", "#52514e"
SURF = "#fcfcfb"

rf = [s for s in json.loads((RUN/"ball_track.json").read_text())["samples"] if s["visible"]]
sm = [s for s in json.loads((RUN/"sam3_ball_track.json").read_text())["samples"] if s["visible"]]

cap = cv2.VideoCapture(str(RUN/"broadcast_proxy.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
ok, frame = cap.read(); cap.release()
rgb = frame[:, :, ::-1]
cmap = LinearSegmentedColormap.from_list("b", BLUE)

def jumps(s):
    return [float(np.hypot(s[i+1]["pixel_x"]-s[i]["pixel_x"], s[i+1]["pixel_y"]-s[i]["pixel_y"]))
            for i in range(len(s)-1)]

fig = plt.figure(figsize=(15, 11), facecolor=SURF)
gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.18, wspace=0.06)

for col, (name, samples) in enumerate([("RF-DETR (current)", rf), ("SAM3 'soccer ball'", sm)]):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(rgb)
    xs = np.array([s["pixel_x"] for s in samples]); ys = np.array([s["pixel_y"] for s in samples])
    ts = np.arange(len(xs))
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, linewidth=2.0, alpha=.85)
    lc.set_array(ts[:-1]); ax.add_collection(lc)
    ax.scatter(xs, ys, c=ts, cmap=cmap, s=26, edgecolor="white", linewidth=.7, zorder=3)
    ax.set_xlim(0, rgb.shape[1]); ax.set_ylim(rgb.shape[0], 0); ax.axis("off")
    j = jumps(samples)
    ax.set_title(f"{name}\n{len(samples)}/300 detected · median jump {np.median(j):.0f}px · p95 {np.percentile(j,95):.0f}px",
                 color=TP, fontsize=12, loc="left", pad=8)

ax2 = fig.add_subplot(gs[1, :], facecolor=SURF)
for (name, samples, c) in [("RF-DETR", rf, S2), ("SAM3 'soccer ball'", sm, S1)]:
    j = jumps(samples)
    ax2.plot(range(len(j)), j, color=c, linewidth=2.0, label=f"{name}  (p95 {np.percentile(j,95):.0f}px)")
ax2.set_title("Frame-to-frame ball jump — lower and flatter is a real track",
              color=TP, fontsize=12, loc="left", pad=8)
ax2.set_xlabel("detection index", color=TS, fontsize=10)
ax2.set_ylabel("jump (px)", color=TS, fontsize=10)
ax2.legend(frameon=False, fontsize=10, labelcolor=TS)
ax2.grid(axis="y", color="#e6e5e1", linewidth=.8); ax2.set_axisbelow(True)
for s_ in ("top","right"): ax2.spines[s_].set_visible(False)
for s_ in ("left","bottom"): ax2.spines[s_].set_color("#d8d7d2")
ax2.tick_params(colors=TS, labelsize=9)

out = RUN/"ball_compare.png"
fig.savefig(out, dpi=105, bbox_inches="tight", facecolor=SURF)
print("wrote", out)
