"""Before/after chart for ball-track smoothing.

Three panels, each answering a different question:
  A  ball x over a 30 s window -- what "gyrations" actually look like
  B  ECDF of frame-to-frame jump, against the physical ceiling for a ball
  C  the path drawn on a real frame -- where the false detections go
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccer_vision.tracking.ball_smooth import smooth_samples  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/u14g-fps30-onepass")
OUT = RUN / "ball_smoothing.png"

RAW, FILT = "#eb6834", "#2a78d6"          # categorical slots 2 and 1, validated
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"
PHYS = 35.0            # px/frame for a 30 m/s ball on this framing


def series(samples):
    t = np.array([s["timestamp_s"] for s in samples], dtype=float)
    x = np.array([s["pixel_x"] if (s.get("visible") and s.get("pixel_x") is not None)
                  else np.nan for s in samples], dtype=float)
    y = np.array([s["pixel_y"] if (s.get("visible") and s.get("pixel_x") is not None)
                  else np.nan for s in samples], dtype=float)
    return t, x, y


def jumps(t, x, y):
    d = np.hypot(np.diff(x), np.diff(y))
    ok = np.isfinite(d) & (np.diff(t) < 0.05)
    return d[ok]


def main():
    samples = json.loads((RUN / "ball_track.json").read_text())["samples"]
    out = smooth_samples(samples)

    t0, x0, y0 = series(samples)
    t1, x1, y1 = series(out)
    j0, j1 = jumps(t0, x0, y0), jumps(t1, x1, y1)

    fig = plt.figure(figsize=(15, 12.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.85, 1.5], hspace=0.46,
                          left=0.07, right=0.975, top=0.865, bottom=0.05)

    fig.text(0.07, 0.965, "The ball track is smooth underneath — the detector adds "
             "the gyrations", fontsize=20, color=INK, weight="bold", va="top")
    fig.text(0.07, 0.932,
             "Saints U14G, 3-minute clip at 30 fps (runs/u14g-fps30-onepass). "
             "18% of raw frame-to-frame steps are physically impossible for a ball; "
             "after gating, 1.4%.",
             fontsize=12.5, color=INK2, va="top")

    # ---- A: x over a window ------------------------------------------------
    ax = fig.add_subplot(gs[0])
    lo, hi = 100.0, 130.0
    m0, m1 = (t0 >= lo) & (t0 <= hi), (t1 >= lo) & (t1 <= hi)
    ax.plot(t0[m0], x0[m0], color=RAW, lw=1.6, label="raw detector output")
    ax.plot(t1[m1], x1[m1], color=FILT, lw=2.0, label="after outlier gate")
    ax.set_title("A  Ball x-position over 30 seconds of play",
                 fontsize=13.5, color=INK, loc="left", pad=8)
    ax.set_ylabel("x (px)", fontsize=11, color=INK2)
    ax.set_xlabel("time (s)", fontsize=11, color=INK2)
    ax.legend(frameon=False, fontsize=11, loc="upper left", labelcolor=INK2, ncol=2)
    ax.annotate("each spike is one or two frames on\na jersey, a boot or the far crowd",
                xy=(0.985, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=10.5, color=MUTED, style="italic")

    # ---- B: ECDF of jumps --------------------------------------------------
    ax = fig.add_subplot(gs[1])
    for d, c, lab in ((j0, RAW, "raw"), (j1, FILT, "gated")):
        s = np.sort(d)
        ax.plot(np.maximum(s, 0.05), np.arange(1, len(s) + 1) / len(s),
                color=c, lw=2.2, label=lab)
    ax.axvline(PHYS, color=MUTED, lw=1.4, ls="--")
    ax.text(PHYS * 1.15, 0.28, "a 30 m/s ball\n(~35 px/frame)", fontsize=10.5,
            color=MUTED, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 2500)
    ax.set_ylim(0, 1.02)
    ax.set_title("B  Frame-to-frame jump — the raw track has a whole second mode "
                 "past the physical limit",
                 fontsize=13.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("jump between consecutive frames (px, log scale)", fontsize=11, color=INK2)
    ax.set_ylabel("share of steps", fontsize=11, color=INK2)
    ax.legend(frameon=False, fontsize=11, loc="upper left", labelcolor=INK2)

    # ---- C: path on a frame ------------------------------------------------
    # A short window: 30 s of trajectory overlaid is spaghetti and shows nothing.
    ax = fig.add_subplot(gs[2])
    clo, chi = 112.0, 118.0
    c0, c1 = (t0 >= clo) & (t0 <= chi), (t1 >= clo) & (t1 <= chi)
    cap = cv2.VideoCapture(str(RUN / "broadcast_proxy.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((clo + chi) / 2 * 29.97))
    ok, frame = cap.read()
    cap.release()
    if ok:
        ax.imshow(frame[:, :, ::-1], alpha=0.70)
    ax.plot(x0[c0], y0[c0], color=RAW, lw=1.8, alpha=0.95,
            marker="o", ms=3, mew=0, label="raw")
    ax.plot(x1[c1], y1[c1], color=FILT, lw=3.0, label="gated")
    ax.set_xlim(0, 1920); ax.set_ylim(1080, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"C  Six seconds of play ({clo:.0f}–{chi:.0f}s) drawn on the pitch — "
                 "the orange spurs shoot off to the crowd and back",
                 fontsize=13.5, color=INK, loc="left", pad=8)
    leg = ax.legend(frameon=True, fontsize=11, loc="lower right", labelcolor=INK2,
                    facecolor=SURFACE, edgecolor=GRID)
    leg.get_frame().set_alpha(0.92)

    for a in fig.get_axes():
        a.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=10)
        if a.get_xticks().size and a.images == []:
            a.grid(True, color=GRID, lw=0.8, alpha=0.9)
            a.set_axisbelow(True)

    fig.savefig(OUT, dpi=115, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
