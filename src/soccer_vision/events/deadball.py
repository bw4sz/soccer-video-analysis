"""Dead-time detection for the ``trim-empty`` editor.

Youth-soccer footage is mostly *dead time*: the ball out of play, or sitting
still while everyone jogs into position. Given a **ball track** (see schema
below) this module decides which spans of the match are dead and which are
worth keeping, so the editor in :mod:`soccer_vision.clips.trim` can splice a
shorter cut. The logic here is pure (no video, no ffmpeg) so it can be unit
tested against synthetic tracks — it works whether the track came from the
RF-DETR ball detector, a future tracker, or a hand-authored fixture.

Ball-track schema (JSON, the input this module reasons over)::

    {
      "video": "match.mp4",
      "fps": 30.0,            # native video frame rate
      "sample_fps": 5.0,      # rate the track was sampled at
      "width": 1920,
      "height": 1080,
      "total_frames": 108000, # optional; used to derive duration
      "samples": [
        {
          "frame": 0,
          "timestamp_s": 0.0,
          "visible": true,     # false = ball offscreen / undetected
          "pixel_x": 950.0,    # null when not visible
          "pixel_y": 540.0,    # null when not visible
          "confidence": 0.82   # 0.0 when not visible
        },
        ...
      ]
    }

A sample is **dead** when the ball is either offscreen (``visible`` false) or
*stationary* — its position has not drifted more than ``stationary_px`` over a
trailing smoothing window. Contiguous dead samples form a run; a run is only
removed once it lasts at least ``min_dead_s`` (the "more than 5 seconds" rule).
"""

from __future__ import annotations

import math

# Defaults tuned for youth footage; overridable from the CLI.
MIN_DEAD_S = 5.0          # a dead run shorter than this is left in
STATIONARY_PX = 40.0      # max positional drift to count as "not moving"
STATIONARY_WINDOW_S = 1.0 # trailing window used to smooth per-sample jitter
PAD_S = 0.5               # context kept on each side of a cut

# Reasons attached to a removed segment.
OFFSCREEN = "offscreen"
STATIONARY = "stationary"
MIXED = "mixed"


def _spread_px(positions: list[tuple[float, float]]) -> float:
    """Bounding-box diagonal of a set of pixel positions."""
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def classify_samples(
    samples: list[dict],
    *,
    stationary_px: float = STATIONARY_PX,
    stationary_window_s: float = STATIONARY_WINDOW_S,
) -> list[str | None]:
    """Label each sample ``OFFSCREEN``, ``STATIONARY``, or ``None`` (live).

    Stationarity is judged over a trailing window of ``stationary_window_s``:
    the ball must be visible across the whole window and its positions must fit
    inside a ``stationary_px`` bounding box. Samples too early to fill the
    window are treated as live (moving), which is the safe default.
    """
    labels: list[str | None] = []
    for i, s in enumerate(samples):
        if not s.get("visible", False):
            labels.append(OFFSCREEN)
            continue

        t = s["timestamp_s"]
        window: list[tuple[float, float]] = []
        window_complete = False
        for j in range(i, -1, -1):
            sj = samples[j]
            if not sj.get("visible", False) or sj.get("pixel_x") is None:
                break
            if t - sj["timestamp_s"] > stationary_window_s:
                window_complete = True
                break
            window.append((sj["pixel_x"], sj["pixel_y"]))
        # j==0 with the whole window visible also counts as complete.
        if not window_complete and samples[0]["timestamp_s"] <= t - stationary_window_s:
            window_complete = True

        if window_complete and len(window) >= 2 and _spread_px(window) < stationary_px:
            labels.append(STATIONARY)
        else:
            labels.append(None)
    return labels


def _run_reason(labels: list[str | None], start: int, end: int) -> str:
    """Majority reason across labels[start:end] (inclusive)."""
    off = sum(1 for k in range(start, end + 1) if labels[k] == OFFSCREEN)
    sta = sum(1 for k in range(start, end + 1) if labels[k] == STATIONARY)
    if off and sta:
        return MIXED
    return OFFSCREEN if off else STATIONARY


def find_removed_segments(
    samples: list[dict],
    *,
    total_duration_s: float | None = None,
    min_dead_s: float = MIN_DEAD_S,
    stationary_px: float = STATIONARY_PX,
    stationary_window_s: float = STATIONARY_WINDOW_S,
    pad_s: float = PAD_S,
) -> list[dict]:
    """Return the time spans to cut out, as ``{start_s, end_s, duration_s, reason}``.

    Contiguous dead samples (offscreen *or* stationary) form a run spanning the
    timestamps of its first and last sample. A run is only cut when its raw
    duration is at least ``min_dead_s``. The actual cut is then shrunk by
    ``pad_s`` on each side so a little context survives around every splice.
    Segments are non-overlapping and sorted by start time.
    """
    if not samples:
        return []

    labels = classify_samples(
        samples,
        stationary_px=stationary_px,
        stationary_window_s=stationary_window_s,
    )

    removed: list[dict] = []
    i = 0
    n = len(samples)
    while i < n:
        if labels[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and labels[j + 1] is not None:
            j += 1

        start = samples[i]["timestamp_s"]
        end = samples[j]["timestamp_s"]
        if end - start >= min_dead_s:
            cut_start = start + pad_s
            cut_end = end - pad_s
            if cut_end - cut_start > 0:
                removed.append({
                    "start_s": round(cut_start, 3),
                    "end_s": round(cut_end, 3),
                    "duration_s": round(cut_end - cut_start, 3),
                    "reason": _run_reason(labels, i, j),
                })
        i = j + 1

    # Never cut past the end of the clip.
    if total_duration_s is not None:
        removed = [r for r in removed if r["start_s"] < total_duration_s]
        for r in removed:
            r["end_s"] = min(r["end_s"], total_duration_s)
            r["duration_s"] = round(r["end_s"] - r["start_s"], 3)
    return removed


def invert_segments(
    removed: list[dict],
    total_duration_s: float,
) -> list[dict]:
    """Complement of ``removed`` over ``[0, total_duration_s]`` — the keep list."""
    keep: list[dict] = []
    cursor = 0.0
    for seg in sorted(removed, key=lambda s: s["start_s"]):
        start = max(0.0, seg["start_s"])
        if start - cursor > 1e-6:
            keep.append({
                "start_s": round(cursor, 3),
                "end_s": round(start, 3),
                "duration_s": round(start - cursor, 3),
            })
        cursor = max(cursor, seg["end_s"])
    if total_duration_s - cursor > 1e-6:
        keep.append({
            "start_s": round(cursor, 3),
            "end_s": round(total_duration_s, 3),
            "duration_s": round(total_duration_s - cursor, 3),
        })
    return keep


def plan_trim(
    samples: list[dict],
    total_duration_s: float,
    *,
    min_dead_s: float = MIN_DEAD_S,
    stationary_px: float = STATIONARY_PX,
    stationary_window_s: float = STATIONARY_WINDOW_S,
    pad_s: float = PAD_S,
) -> dict:
    """One-shot plan: removed + keep segments plus duration bookkeeping.

    Returns the body of the edit-decision list written by the editor.
    """
    removed = find_removed_segments(
        samples,
        total_duration_s=total_duration_s,
        min_dead_s=min_dead_s,
        stationary_px=stationary_px,
        stationary_window_s=stationary_window_s,
        pad_s=pad_s,
    )
    keep = invert_segments(removed, total_duration_s)
    removed_dur = sum(r["duration_s"] for r in removed)
    kept_dur = sum(k["duration_s"] for k in keep)
    return {
        "params": {
            "min_dead_s": min_dead_s,
            "stationary_px": stationary_px,
            "stationary_window_s": stationary_window_s,
            "pad_s": pad_s,
        },
        "source_duration_s": round(total_duration_s, 3),
        "kept_duration_s": round(kept_dur, 3),
        "removed_duration_s": round(removed_dur, 3),
        "keep_segments": keep,
        "removed_segments": removed,
    }
