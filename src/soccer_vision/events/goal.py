"""Goal detection: the ball dwelling inside a detected goal mouth.

``goal`` has been in the canonical taxonomy (:mod:`soccer_vision.events.labels`)
since the start, but nothing has ever emitted one — the ``rules`` engine only
fires on set pieces. This module is the first producer.

The signal is deliberately simple, and it is *geometry over two sidecars*: the
goal-mouth regions from :mod:`soccer_vision.detection.goal` (``goals.json``) and
the ball trajectory from ``ball_track.json``. Like
:mod:`soccer_vision.events.on_ball`, it works in pixel space, so it needs
neither the field homography (degenerate on overhead footage) nor a trained
action model.

**Dwell, not crossing.** A single ball sample inside the goal-mouth box means
almost nothing: from an elevated camera the net is *behind* a large slice of the
penalty area, so every shot, every goalkeeper pass and half the goal-kick
setups put the ball "inside" that box in 2D for a frame or two. What separates a
goal from a fly-by is that the ball **stays** there — the net catches it and
play stops. So a span only becomes a ``goal`` once the ball has been inside for
``min_dwell_s``; that is the lag time that discards the ball merely passing in
front.

Two further discriminators come cheap and are applied here:

* **Entry side.** The ball must arrive from the field side of the mouth. A ball
  wandering in from behind the goal (retrieved out of play, a warm-up ball) is
  not a goal.
* **Losing the ball counts as staying.** A ball that disappears into the netting
  stops being detected. An *invisible* sample therefore does not break a dwell
  span (up to ``max_gap_s``), while a visible sample clearly outside the mouth
  does. Vanishing inside the goal is evidence for, not against.

**What this does not do.** It cannot see depth. A ball struck hard against the
face of the net, or a keeper standing in the mouth holding the ball, are exactly
the shapes this fires on, and no amount of tuning the dwell fixes that from one
2D box — see the caveat in the module-level docs and the tracking issue. Treat
the output as *candidates worth a look*, which for a parent cutting a highlight
reel is the useful bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GOAL_LABEL = "goal"


@dataclass
class GoalRegion:
    """A detected goal mouth, static box plus its per-sample observations.

    ``bbox`` is the consolidated (median) mouth over the whole video, which is
    what a fixed overhead camera needs. ``samples`` keeps the per-frame boxes so
    a panning/cropped view still resolves the mouth near the frame in question —
    :meth:`bbox_at` picks whichever is appropriate without the caller caring.
    """

    side: str                       # "left" | "right" — which end of the frame
    bbox: tuple[float, float, float, float]
    score: float = 1.0              # detector confidence for the consolidated box
    n_obs: int = 0                  # frames the mouth was detected in
    samples: list[dict] = field(default_factory=list)  # [{frame, bbox, score}]

    def bbox_at(
        self, frame: int, *, max_frame_delta: float = 300.0
    ) -> tuple[float, float, float, float]:
        """The mouth box to use at ``frame``.

        Nearest per-frame observation when one is close enough in time,
        otherwise the consolidated box. On a static camera the two agree; on a
        moving one the nearest observation is what tracks the pan.
        """
        best, best_d = None, max_frame_delta
        for s in self.samples:
            d = abs(int(s["frame"]) - frame)
            if d <= best_d:
                best, best_d = s["bbox"], d
        return tuple(best) if best is not None else self.bbox


def load_goal_regions(goals: dict) -> list[GoalRegion]:
    """Parse a ``goals.json`` payload into :class:`GoalRegion` objects."""
    out = []
    for g in goals.get("goals", []):
        out.append(GoalRegion(
            side=g.get("side", "?"),
            bbox=tuple(g["bbox"]),
            score=float(g.get("score", 1.0)),
            n_obs=int(g.get("n_obs", 0)),
            samples=list(g.get("samples", [])),
        ))
    return out


def _inset(bbox, frac: float) -> tuple[float, float, float, float]:
    """Shrink a box toward its centre by ``frac`` of each dimension.

    The segmented mouth includes the frame, posts and a little slack around
    them. Insetting requires the ball to be *well* inside rather than level with
    a post, which is where a shot going narrowly wide lands.
    """
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


def _inside(x: float, y: float, bbox) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _entered_from_field(prev_sample: dict | None, region: GoalRegion, bbox) -> bool:
    """Whether the last position before the dwell was on the pitch side.

    With no prior visible sample we allow it — the ball being lost right before
    it turns up in the net is a normal way for a goal to look, and demanding
    evidence we don't have would drop real goals.
    """
    if prev_sample is None:
        return True
    x1, _, x2, _ = bbox
    px = prev_sample["pixel_x"]
    # A "left" goal sits at the left of the frame, so the pitch is to its right.
    return px >= x1 if region.side == "left" else px <= x2


def _drift_px(points: list[tuple[float, float]]) -> float:
    """Bounding-box diagonal of a set of points — how far the ball moved."""
    if len(points) < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    return (dx * dx + dy * dy) ** 0.5


def _confidence(
    dwell_s: float, min_dwell_s: float, region_score: float, drift: float,
    still_px: float,
) -> float:
    """Heuristic score in [0, 1] — a ranking aid, not a calibrated probability.

    Longer dwell, a better-detected mouth, and a ball that comes to rest all
    push it up. Stillness is a bonus rather than a gate: youth nets are loose
    and the ball often rebounds straight back out.
    """
    dwell_factor = min(1.0, dwell_s / (2.0 * min_dwell_s)) if min_dwell_s > 0 else 1.0
    score = 0.35 + 0.30 * dwell_factor + 0.20 * min(1.0, region_score)
    if drift <= still_px:
        score += 0.15
    return round(min(0.95, score), 3)


def detect_goals(
    ball_track: dict,
    regions: list[GoalRegion],
    *,
    min_dwell_s: float = 0.6,
    max_dwell_s: float = 10.0,
    max_gap_s: float = 0.7,
    inset_frac: float = 0.12,
    require_entry: bool = True,
    still_px: float = 60.0,
    dedup_window_s: float = 15.0,
) -> list[dict]:
    """Emit ``goal`` events where the ball dwells inside a goal mouth.

    ``ball_track`` is the parsed ``ball_track.json``; ``regions`` come from
    :func:`load_goal_regions`. Returns event dicts in the shape the rest of the
    pipeline expects (``label`` / ``frame`` / ``timestamp_s`` / ``position_ms`` /
    ``confidence``), with ``goal_zone`` naming the end — the same key
    ``goal_kick`` already uses, so ``process`` carries it through unchanged.
    """
    samples = [s for s in ball_track.get("samples", []) if s.get("frame") is not None]
    samples.sort(key=lambda s: int(s["frame"]))
    if not samples or not regions:
        return []

    fps = ball_track.get("fps") or 30.0
    events: list[dict] = []

    for region in regions:
        events.extend(_dwell_spans(
            samples, region, fps=fps, min_dwell_s=min_dwell_s,
            max_dwell_s=max_dwell_s, max_gap_s=max_gap_s,
            inset_frac=inset_frac, require_entry=require_entry, still_px=still_px,
        ))

    events.sort(key=lambda e: e["timestamp_s"])
    return _dedup(events, dedup_window_s)


def _dwell_spans(
    samples: list[dict],
    region: GoalRegion,
    *,
    fps: float,
    min_dwell_s: float,
    max_dwell_s: float,
    max_gap_s: float,
    inset_frac: float,
    require_entry: bool,
    still_px: float,
) -> list[dict]:
    """Group this region's inside-the-mouth samples into candidate goals."""
    out: list[dict] = []
    cur: dict | None = None
    prev_visible: dict | None = None      # last visible sample before the current span

    def close(span):
        if span is None:
            return
        dwell_s = span["last_ts"] - span["first_ts"]
        # Epsilon because the threshold usually lands exactly on a sample
        # boundary: a 5 fps track differencing 0.3 - 0.2 gives 0.0999...,
        # which would drop a dwell the caller asked to keep.
        if dwell_s < min_dwell_s - 1e-6:
            return
        # A ball that sits there for half a minute was never scored — it is
        # parked out of play behind the goal, or the region overlaps dead space
        # the ball rolls into. After a real goal the ball is retrieved within
        # seconds, because play has to restart.
        if max_dwell_s and dwell_s > max_dwell_s:
            return
        if require_entry and not span["entered_from_field"]:
            return
        drift = _drift_px(span["points"])
        out.append({
            "label": GOAL_LABEL,
            "frame": span["first_frame"],
            "timestamp_s": round(span["first_ts"], 2),
            "position_ms": int(span["first_ts"] * 1000),
            "end_s": round(span["last_ts"], 2),
            "duration_s": round(dwell_s, 2),
            "confidence": _confidence(
                dwell_s, min_dwell_s, region.score, drift, still_px),
            "goal_zone": region.side,
            "pixel_x": span["points"][0][0],
            "pixel_y": span["points"][0][1],
            "dwell_s": round(dwell_s, 2),
            "n_samples": span["n"],
            "drift_px": round(drift, 1),
            "ball_lost_frac": round(span["n_lost"] / max(1, span["n"] + span["n_lost"]), 2),
        })

    for s in samples:
        frame = int(s["frame"])
        ts = float(s.get("timestamp_s", frame / fps))
        visible = bool(s.get("visible")) and s.get("pixel_x") is not None

        if not visible:
            # The ball going missing mid-dwell is evidence it is *in* the net,
            # so hold the span open until the gap gets too long.
            if cur is not None:
                if ts - cur["last_ts"] > max_gap_s:
                    close(cur)
                    cur = None
                else:
                    cur["n_lost"] += 1
            continue

        bbox = _inset(region.bbox_at(frame), inset_frac)
        x, y = float(s["pixel_x"]), float(s["pixel_y"])

        if _inside(x, y, bbox):
            if cur is None:
                cur = {
                    "first_frame": frame, "first_ts": ts, "last_ts": ts,
                    "n": 1, "n_lost": 0, "points": [(x, y)],
                    "entered_from_field": _entered_from_field(
                        prev_visible, region, region.bbox_at(frame)),
                }
            else:
                cur["last_ts"] = ts
                cur["n"] += 1
                cur["points"].append((x, y))
        else:
            # Visible and demonstrably outside the mouth — the dwell is over.
            close(cur)
            cur = None
        prev_visible = s

    close(cur)
    return out


def _dedup(events: list[dict], window_s: float) -> list[dict]:
    """Keep the most confident event within each ``window_s`` cluster.

    The ball can rattle in the net, exit the inset box and re-enter, which reads
    as several dwells one after another for what a viewer calls one goal.
    """
    kept: list[dict] = []
    for e in events:
        if kept and e["timestamp_s"] - kept[-1]["timestamp_s"] < window_s:
            if e["confidence"] > kept[-1]["confidence"]:
                kept[-1] = e
            continue
        kept.append(e)
    return kept
