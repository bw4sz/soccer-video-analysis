"""Visualise the persisted ball track: spatial path + detection reliability.

Panel 1 — the ball's path drawn on a real frame, coloured light->dark by time
          (sequential blue ramp), so you can read direction of play.
Panel 2 — frame-to-frame jump in pixels, which exposes detector flicker (the
          ball teleporting onto a jersey number or logo and snapping back),
          with a rug showing frames where the ball was not detected at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/orange/ewhite/b.weinstein/soccer-video-analysis/runs/sam3-ball")
OUT = RUN / "ball_track_viz.png"

# Sequential blue ramp (validated palette, steps 100 -> 700)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES_1 = "#2a78d6"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"


def main() -> None:
    track = json.loads((RUN / "ball_track.json").read_text())
    samples = track["samples"]
    vis = [s for s in samples if s["visible"]]
    print(f"samples: {len(samples)}  visible: {len(vis)} "
          f"({100*len(vis)/max(1,len(samples)):.1f}%)")

    # representative frame from the middle of the clip
    cap = cv2.VideoCapture(str(RUN / "broadcast_proxy.mp4"))
    mid = samples[len(samples) // 2]["frame"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("could not read a frame")
    rgb = frame[:, :, ::-1]

    cmap = LinearSegmentedColormap.from_list("ballblue", BLUE_RAMP)
    fig = plt.figure(figsize=(14, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.22)

    # ---- Panel 1: trajectory on the pitch ----
    ax = fig.add_subplot(gs[0])
    ax.imshow(rgb)
    xs = np.array([s["pixel_x"] for s in vis])
    ys = np.array([s["pixel_y"] for s in vis])
    ts = np.array([s["timestamp_s"] for s in vis])

    # connect consecutive detections; colour each segment by time
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=2.0, alpha=0.85)
    lc.set_array(ts[:-1])
    ax.add_collection(lc)
    ax.scatter(xs, ys, c=ts, cmap=cmap, s=34, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlim(0, rgb.shape[1]); ax.set_ylim(rgb.shape[0], 0)
    ax.axis("off")
    ax.set_title(f"Ball path over {ts.max()-ts.min():.0f}s  ·  "
                 f"{len(vis)}/{len(samples)} samples detected "
                 f"({100*len(vis)/len(samples):.0f}%)",
                 color=TEXT_PRIMARY, fontsize=13, loc="left", pad=10)
    cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("elapsed (s)", color=TEXT_SECONDARY, fontsize=9)
    cb.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cb.outline.set_visible(False)

    # ---- Panel 2: frame-to-frame jump (flicker) ----
    ax2 = fig.add_subplot(gs[1], facecolor=SURFACE)
    jumps, jt = [], []
    for a, b in zip(vis[:-1], vis[1:]):
        jumps.append(float(np.hypot(b["pixel_x"]-a["pixel_x"], b["pixel_y"]-a["pixel_y"])))
        jt.append(b["timestamp_s"])
    if jumps:
        ax2.plot(jt, jumps, color=SERIES_1, linewidth=2.0)
        p95 = float(np.percentile(jumps, 95))
        ax2.axhline(p95, color=TEXT_MUTED, linewidth=1, linestyle="--")
        ax2.annotate(f"p95 = {p95:.0f}px", (jt[-1], p95), xytext=(-4, 6),
                     textcoords="offset points", ha="right",
                     color=TEXT_SECONDARY, fontsize=9)
        print(f"jump px: median {np.median(jumps):.0f}  p95 {p95:.0f}  max {max(jumps):.0f}")

    # rug: frames with no detection
    miss = [s["timestamp_s"] for s in samples if not s["visible"]]
    if miss:
        ax2.plot(miss, [0]*len(miss), "|", color=CRITICAL, markersize=9,
                 markeredgewidth=1.5, label=f"ball not detected ({len(miss)})")
        ax2.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")

    ax2.set_title("Frame-to-frame ball jump — large spikes are detector flicker",
                  color=TEXT_PRIMARY, fontsize=12, loc="left", pad=8)
    ax2.set_xlabel("time (s)", color=TEXT_SECONDARY, fontsize=10)
    ax2.set_ylabel("jump (px)", color=TEXT_SECONDARY, fontsize=10)
    ax2.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color("#d8d7d2")
    ax2.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    fig.savefig(OUT, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
