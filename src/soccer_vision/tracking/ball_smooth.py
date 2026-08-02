"""Robust smoothing for the flickery single-ball track.

The RF-DETR ball detector fires once per frame and is right most of the time,
but on youth footage it *flickers*: for a frame or two it latches onto a white
boot, a jersey number, a line marking or someone in the far crowd, so the
reported position teleports across the pitch and snaps back. Measured on a
3-minute U14G clip at 30 fps and reproduced on the full 60-minute match, the
raw track from `process` is bimodal — the median step between frames is 4 px,
but **18% of steps exceed 70 px/frame**, which no ball can do (a 30 m/s ball is
about 35 px/frame on this framing).

Why a *median gate* rather than the Kalman filter in
:mod:`soccer_vision.tracking.ball_kalman`
-------------------------------------------------------------------------------
Two measured facts settle it:

1. **The excursions are short.** Median 2 frames, longest 0.6 s, and 219 of 470
   episodes are a single frame. The true trajectory is never lost for long, so
   it is recoverable from the surrounding frames.
2. **A causal filter cannot use that.** It has to decide *at* the excursion
   whether the ball genuinely relocated or the detector lied, and the only
   evidence that settles it arrives afterwards. ``ball_kalman`` guesses via
   ``reacquire_after``, and on a 3-frame excursion it guesses wrong and re-locks
   onto the flicker — which is why it only cuts unphysical steps from 18% to 8%.

This module is offline (it runs on a saved track, where the future is available)
and reduces unphysical steps to ~2%. A comparison on the same clip, with a
reference set of detections independently within 60 px of their local median:

===============================  ========  ========  ==========
method                           p95 jump  >70 px    coverage
===============================  ========  ========  ==========
raw (what `process` writes)        707 px    18.2%      86.0%
causal Kalman (ball_kalman)        322 px     8.3%      86.0%
this module                         43 px     2.2%      82.2%
===============================  ========  ========  ==========

**The motion model turned out to be worth nothing here** — rejecting outliers is
the whole gain. Adding a Rauch-Tung-Striebel smoother on top of this gate made
every metric slightly *worse* (it injects wobble between detections: median step
2.7 px → 4.5 px) so it is deliberately not done.

What this does *not* fix
------------------------
About 6% of rejected detections are consistent with a straight-line continuation
of the ball's velocity, i.e. some genuine fast motion is being cut along with
the flicker. Recovering it needs the detector to emit **more than one ball
candidate per frame** — today
:func:`soccer_vision.detection.ball.ball_position_from` keeps only the argmax, so
when the top box is a jersey number the real ball is discarded before any filter
sees it. With top-k candidates this gate becomes a shortest-path problem over
them, which is the principled version. Note too that pixel speed includes
**camera motion** — Veo pans and zooms — so a step is not purely the ball's.
"""

from __future__ import annotations

import numpy as np

# Defaults measured on 30 fps Veo footage at 1920x1080; see module docstring.
WINDOW_S = 0.5        # half-width of the local-median window, in seconds
MAX_DEV_PX = 150.0    # a detection this far off the local median is flicker
ITERATIONS = 3        # recompute the median on survivors, so outliers stop voting
MAX_FILL_S = 0.5      # coast across gaps up to this long; leave longer ones absent


def _xy(samples: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Timestamps and detected positions, with NaN where nothing was detected."""
    t = np.array([s["timestamp_s"] for s in samples], dtype=float)
    x = np.array([s["pixel_x"] if (s.get("visible") and s.get("pixel_x") is not None)
                  else np.nan for s in samples], dtype=float)
    y = np.array([s["pixel_y"] if (s.get("visible") and s.get("pixel_x") is not None)
                  else np.nan for s in samples], dtype=float)
    return t, x, y


def _local_median_dev(x: np.ndarray, y: np.ndarray, half: int,
                      min_support: int) -> np.ndarray:
    """Distance of each detection from the median of its +-``half`` samples.

    The median is taken per axis and ignores missing samples. It is the right
    summary here because it tolerates the ~25% contamination that would drag a
    mean straight off the trajectory.
    """
    n = len(x)
    pad = np.full(half, np.nan)
    px = np.concatenate([pad, x, pad])
    py = np.concatenate([pad, y, pad])
    win = 2 * half + 1
    sx = np.lib.stride_tricks.sliding_window_view(px, win)
    sy = np.lib.stride_tricks.sliding_window_view(py, win)

    with np.errstate(invalid="ignore"):
        # A window needs a few real samples before its median means anything.
        enough = np.isfinite(sx).sum(axis=1) >= min_support
        mx = np.full(n, np.nan)
        my = np.full(n, np.nan)
        if enough.any():
            mx[enough] = np.nanmedian(sx[enough], axis=1)
            my[enough] = np.nanmedian(sy[enough], axis=1)
    return np.hypot(x - mx, y - my)


def gate_outliers(
    samples: list[dict],
    *,
    window_s: float = WINDOW_S,
    max_dev_px: float = MAX_DEV_PX,
    iterations: int = ITERATIONS,
) -> np.ndarray:
    """Boolean mask over ``samples``: True where the detection is believable.

    Iterated because the first median is computed on contaminated data; letting
    the survivors vote again tightens it (unphysical steps 2.7% -> 1.0% on the
    validation clip).
    """
    t, x, y = _xy(samples)
    if len(t) < 2:
        return np.isfinite(x)

    # The window is defined in *seconds*, so it spans the same amount of play at
    # any sample rate -- which is what makes the 150 px allowance rate-independent
    # (it is roughly how far the ball travels in half a window).  A floor of 3
    # keeps a sparse track from being judged on two neighbours.
    dt = float(np.median(np.diff(t)))
    half = max(3, int(round(window_s / dt))) if dt > 0 else 3
    # Support scales with the window: an absolute count would demand a *full*
    # window at 5 fps (5 slots) while asking a sixth of one at 30 fps, which
    # collapsed 5 fps tracks to 10% coverage.
    min_support = max(3, (2 * half + 1) // 4)

    keep = np.isfinite(x)
    for _ in range(max(1, iterations)):
        wx = np.where(keep, x, np.nan)
        wy = np.where(keep, y, np.nan)
        dev = _local_median_dev(wx, wy, half, min_support)
        # Support must be positive: a detection whose window holds too few other
        # detections to form a median is *unverified*, and keeping those was
        # measured to put the flicker straight back (unphysical steps 2.2% ->
        # 2.9%, p95 jump 43px -> 52px on the validation clip).  They are the
        # thinly-detected stretches, where a lone detection is as likely to be a
        # boot as a ball.  A genuine one is recovered on the next iteration once
        # its neighbours survive.
        keep = np.isfinite(x) & np.isfinite(dev) & (dev <= max_dev_px)
    return keep


def smooth_samples(
    samples: list[dict],
    *,
    window_s: float = WINDOW_S,
    max_dev_px: float = MAX_DEV_PX,
    iterations: int = ITERATIONS,
    max_fill_s: float = MAX_FILL_S,
) -> list[dict]:
    """Return a copy of ``samples`` (deadball schema) with flicker removed.

    Rejected detections keep their original position under ``raw_pixel_x`` /
    ``raw_pixel_y`` and are marked ``outlier``. A frame between two believable
    detections no more than ``max_fill_s`` apart is filled by interpolation and
    marked ``interpolated`` — the ball was really there, it was just occluded or
    misdetected. **Longer gaps are left ``visible: false``**, because inventing a
    position across them would erase exactly the out-of-play stretches
    ``trim-empty`` exists to find.
    """
    keep = gate_outliers(samples, window_s=window_s, max_dev_px=max_dev_px,
                         iterations=iterations)
    t, x, y = _xy(samples)
    kept = np.flatnonzero(keep)

    out: list[dict] = []
    if len(kept) == 0:
        return [dict(s) for s in samples]

    kt, kx, ky = t[kept], x[kept], y[kept]
    keptset = set(kept.tolist())

    for i, s in enumerate(samples):
        new = dict(s)
        if i in keptset:
            new["smoothed"] = True
            out.append(new)
            continue

        had_detection = s.get("visible") and s.get("pixel_x") is not None
        if had_detection:
            new["raw_pixel_x"] = s["pixel_x"]
            new["raw_pixel_y"] = s["pixel_y"]
            new["outlier"] = True

        # Fill only when bracketed by believable detections that are close in time.
        j = int(np.searchsorted(kt, t[i]))
        if 0 < j < len(kt) and (kt[j] - kt[j - 1]) <= max_fill_s:
            new["visible"] = True
            new["pixel_x"] = round(float(np.interp(t[i], kt, kx)), 1)
            new["pixel_y"] = round(float(np.interp(t[i], kt, ky)), 1)
            new["smoothed"] = True
            new["interpolated"] = True
        else:
            new["visible"] = False
            new["pixel_x"] = None
            new["pixel_y"] = None
        out.append(new)

    return out


def smooth_ball_track(track: dict, **kwargs) -> dict:
    """Return a copy of ``track`` with its ``samples`` gated and gap-filled."""
    smoothed = dict(track)
    smoothed["samples"] = smooth_samples(track.get("samples", []), **kwargs)
    smoothed["smoothed"] = True
    return smoothed
