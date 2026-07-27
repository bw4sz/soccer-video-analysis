"""Goal-mouth detection by text prompt.

SAM3 is already the detector for players and the ball, and it is driven by
*text* — so a goal needs no new model and no new weights, only a new prompt
(``"soccer goal net"``). :class:`soccer_vision.tracking.sam3.SAM3PlayerTracker`
is generic despite its name (a promptable concept tracker), and its ``sharing``
constructor exists precisely so a second concept reuses the loaded weights.

**A goal does not move.** That is the property this module leans on. Per-frame
net segmentation flickers like any other detection, but the true mouth is one
box that is either fixed (a static overhead camera — Veo, Pixellot) or drifting
slowly (a cropped/panning view). So we sample sparsely, throw away implausible
boxes, and take a **per-side median** over everything that survives. The median
is robust to the frames where the prompt latches onto a sideline banner or a
five-a-side rebound board, which no single-frame confidence threshold is.

Output is ``goals.json`` in the run directory, consumed by
:mod:`soccer_vision.events.goal`. The consolidated box is what a fixed camera
needs; the per-frame observations are kept alongside so a panning view can
resolve the mouth near a given frame instead.

Runs as an opt-in step (``soccer-vision goals``) rather than inside ``process``,
matching ``identify`` / ``enroll``: it needs a GPU, it is re-runnable, and most
runs never ask for goal events.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PROMPT = "soccer goal net"

# Plausibility bounds for a goal mouth on wide/overhead youth footage. A goal is
# always wider than it is tall from any camera that can see the field, and it
# occupies a small but not vanishing slice of the frame. These reject the two
# common prompt failures: a thin sideline banner (too wide/flat, or too small)
# and the whole penalty area segmented as one blob (too large).
MIN_AREA_RATIO = 0.0005   # 0.05% of frame
MAX_AREA_RATIO = 0.15     # 15% of frame
MIN_ASPECT = 1.1          # width / height
MAX_ASPECT = 12.0


def is_available() -> bool:
    """Whether the SAM3 runtime needed for goal-mouth detection is present.

    Checks CUDA before touching ``transformers``. That import costs ~4 minutes
    on a cold shared filesystem, which is a long time to wait to be told there
    is no GPU — and the common case here is a CPU box wanting only the event
    stage, which needs neither.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False

    from soccer_vision.tracking.sam3 import is_available as sam3_available

    return sam3_available()


def _plausible(bbox, frame_w: int, frame_h: int) -> bool:
    """Filter a candidate box to goal-mouth-shaped things."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    area_ratio = (w * h) / float(frame_w * frame_h)
    if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
        return False
    return MIN_ASPECT <= (w / h) <= MAX_ASPECT


def assign_side(bbox, frame_w: int) -> str:
    """Which end of the frame a box belongs to.

    Both goals are near the left and right edges of a wide view, so the frame
    midpoint separates them. Reported as ``left``/``right`` to match the
    ``goal_zone`` key ``goal_kick`` already uses.
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    return "left" if cx < frame_w / 2.0 else "right"


def consolidate(
    observations: list[dict],
    *,
    min_obs: int = 3,
    max_spread_frac: float = 0.5,
) -> list[dict]:
    """Fold per-frame boxes into one median mouth per side.

    ``observations`` are ``{frame, bbox, score, side}`` dicts. Pure function —
    this is the part worth testing without a GPU.

    A side is dropped when it has fewer than ``min_obs`` observations (too thin
    to trust), or when its boxes disagree wildly: if the spread of centres
    exceeds ``max_spread_frac`` of the median box width, the prompt was latching
    onto different objects on different frames rather than onto one goal, and a
    median of that is a fiction. Dropping is the right answer — a missing goal
    yields no events, a wrong one yields confident nonsense.
    """
    by_side: dict[str, list[dict]] = {}
    for o in observations:
        by_side.setdefault(o["side"], []).append(o)

    out = []
    for side, obs in sorted(by_side.items()):
        if len(obs) < min_obs:
            continue
        boxes = np.array([o["bbox"] for o in obs], dtype=float)
        med = np.median(boxes, axis=0)
        med_w = med[2] - med[0]

        centres_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
        centres_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
        spread = max(
            float(np.percentile(centres_x, 90) - np.percentile(centres_x, 10)),
            float(np.percentile(centres_y, 90) - np.percentile(centres_y, 10)),
        )
        if med_w > 0 and spread > max_spread_frac * med_w:
            continue

        out.append({
            "side": side,
            "bbox": [round(float(v), 1) for v in med],
            "score": round(float(np.median([o.get("score", 1.0) for o in obs])), 3),
            "n_obs": len(obs),
            "centre_spread_px": round(spread, 1),
            "samples": [
                {"frame": int(o["frame"]),
                 "bbox": [round(float(v), 1) for v in o["bbox"]],
                 "score": round(float(o.get("score", 1.0)), 3)}
                for o in obs
            ],
        })
    return out


def detect_goal_mouths(
    video_path: str,
    *,
    prompt: str = DEFAULT_PROMPT,
    sample_fps: float = 0.2,
    max_samples: int = 60,
    device: str = "cuda",
    min_score: float = 0.3,
    min_obs: int = 3,
    tracker=None,
    progress: bool = True,
) -> dict:
    """Detect both goal mouths in ``video_path``; return a ``goals.json`` payload.

    Sampling is deliberately sparse (``sample_fps`` 0.2 = one frame every five
    seconds, capped at ``max_samples``): the goal is static, so more frames buy
    nothing but GPU time. ``tracker`` accepts an already-loaded
    ``SAM3PlayerTracker`` so a caller inside ``process`` can share the weights
    via ``SAM3PlayerTracker.sharing(tracker, prompt)``.
    """
    from soccer_vision.io.video import VideoReader
    from soccer_vision.tracking.sam3 import SAM3PlayerTracker

    reader = VideoReader(video_path)
    fps = reader.fps or 30.0
    total = reader.total_frames
    interval = max(1, int(round(fps / max(sample_fps, 1e-6))))
    # Spread the cap across the whole match rather than sampling only its start,
    # so a camera that moves at halftime is still represented.
    if total and total / interval > max_samples:
        interval = max(1, int(total / max_samples))

    own_tracker = tracker is None
    if own_tracker:
        tracker = SAM3PlayerTracker(device=device, prompt=prompt, min_score=min_score)
        tracker.start()

    observations: list[dict] = []
    n_frames = 0
    try:
        for fn, frame in reader.sample_frames(interval):
            dets = tracker.track(frame)
            n_frames += 1
            h, w = frame.shape[:2]
            for i in range(len(dets)):
                bbox = [float(v) for v in dets.xyxy[i]]
                if not _plausible(bbox, w, h):
                    continue
                score = float(dets.confidence[i]) if dets.confidence is not None else 1.0
                observations.append({
                    "frame": int(fn), "bbox": bbox, "score": score,
                    "side": assign_side(bbox, w),
                })
            if progress and n_frames % 10 == 0:
                print(f"  goal-mouth sampling: frame {fn}/{total} "
                      f"({len(observations)} candidate boxes)")
    finally:
        if own_tracker:
            tracker.close()
        width, height = reader.width, reader.height
        reader.close()

    goals = consolidate(observations, min_obs=min_obs)
    return {
        "video": str(video_path),
        "prompt": prompt,
        "fps": fps,
        "width": int(width),
        "height": int(height),
        "sample_interval": interval,
        "n_frames_sampled": n_frames,
        "n_candidate_boxes": len(observations),
        "goals": goals,
    }
