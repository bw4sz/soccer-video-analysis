"""A hand-drawn "this is our pitch" polygon, in pixel space.

At a multi-field complex the detector finds every player on every pitch, and
nothing in the frame says which match is ours. The central-rectangle hull in
:mod:`soccer_vision.detection.field_filter` is a guess (it assumes our pitch
fills the middle of the frame); turf-mask segmentation would find *a* field but
still not say which one is ours; and line-based registration is dead here for
the reasons in :mod:`soccer_vision.pitch` — the home complex has blue, red and
white lines from several overlapping pitches painted at once.

The cheapest correct answer is to ask: a person looks at one frame and names the
corners of our pitch once per match. This module is the plumbing around that
answer — store it, replay it, and keep it aligned as the camera pans.

**Re-id supersedes this.** Once a gallery names our players (see
``soccer-vision enroll``), the target pitch is wherever our players are, and no
polygon is needed. This is the fallback for footage where the gallery is absent
or abstains.

Two ways to follow an XbotGo/Veo camera, which pans but does not travel:

* **Keyframes** — give the polygon at several frames and it is linearly
  interpolated between them. Fully manual, entirely predictable, and the only
  option if you are naming coordinates from a still image without a display.
* **Pan tracking** (``track_pan``) — estimate a similarity transform from the
  keyframe's reference frame to the current one (ORB + RANSAC on a downscaled
  greyscale pair) and carry the polygon through it. Automatic, and it degrades
  to "hold the last good transform" rather than to garbage: implausible
  transforms are rejected outright, so a failed estimate leaves the polygon
  where it was instead of teleporting it off-pitch.

Coordinates are stored **normalised** to the frame (0-1), so a region drawn on a
1080p still still applies to a 720p proxy of the same footage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Reject a motion estimate that implies the camera did something these cameras
# do not do: a jump of more than a quarter-frame, or a zoom beyond +/-25%.
_MAX_SHIFT_FRAC = 0.25
_MAX_SCALE = 1.25
_MIN_SCALE = 0.8
_MIN_INLIERS = 12


class PitchRegionError(ValueError):
    """A region file or polygon spec that cannot be used as given."""


def parse_polygon_spec(spec: str, width: int | None = None,
                       height: int | None = None, mode: str = "auto",
                       min_points: int = 3) -> np.ndarray:
    """Parse ``"x,y x,y ..."`` into a normalised (N, 2) polygon.

    Accepts either normalised coordinates or pixel coordinates, which need
    ``width``/``height`` to normalise. ``mode`` is ``auto`` (values all within
    0-1 are normalised, anything else is pixels), ``normalized``, or ``pixels``.
    Say which when the polygon deliberately runs off the frame — see
    :func:`polygon_below_line`, where x reaches -1 and 2. Points may be separated
    by spaces or semicolons; commas separate x from y.
    """
    tokens = [t for t in spec.replace(";", " ").replace("\n", " ").split(" ") if t.strip()]
    pts: list[tuple[float, float]] = []
    for tok in tokens:
        parts = tok.split(",")
        if len(parts) != 2:
            raise PitchRegionError(
                f"bad point {tok!r}: expected 'x,y' pairs, e.g. "
                "'0.05,0.45 0.95,0.42 0.98,0.99 0.02,0.99'")
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise PitchRegionError(f"bad point {tok!r}: {exc}") from exc

    if len(pts) < min_points:
        raise PitchRegionError(
            f"expected at least {min_points} points, got {len(pts)}")

    arr = np.asarray(pts, dtype=float)
    if mode not in ("auto", "normalized", "pixels"):
        raise PitchRegionError(f"unknown coordinate mode {mode!r}")
    if mode == "normalized" or (mode == "auto"
                                and bool(np.all((arr >= 0.0) & (arr <= 1.0)))):
        return arr
    if not width or not height:
        raise PitchRegionError(
            "pixel coordinates need the frame size to normalise — pass a video "
            "so the frame can be measured, or give coordinates in 0-1")
    return arr / np.array([float(width), float(height)])


def polygon_below_line(p1, p2, extend: float = 1.0) -> np.ndarray:
    """Everything on the near side of a line, extended past the frame edges.

    The dominant case at a multi-pitch venue is a single boundary: our far
    touchline, with the next match and the crowd beyond it. Bounding that region
    left and right is not just unnecessary, it is harmful — the Veo camera zooms
    out, and a polygon that stopped at the frame edge when it was drawn then cuts
    off our own players who were previously out of shot. So the region runs from
    x = -``extend`` to 1 + ``extend`` and down past the bottom of the frame.

    ``p1``/``p2`` are two normalised points on the line, in any order.
    """
    (x1, y1), (x2, y2) = np.asarray(p1, float), np.asarray(p2, float)
    if x1 == x2:
        raise PitchRegionError(
            "the two points of a boundary line must differ in x — a vertical "
            "line has no 'below'")
    slope = (y2 - y1) / (x2 - x1)
    lo, hi = -extend, 1.0 + extend
    return np.array([
        [lo, y1 + slope * (lo - x1)],
        [hi, y1 + slope * (hi - x1)],
        [hi, 1.0 + extend],
        [lo, 1.0 + extend],
    ])


def _scale_about_centroid(poly: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return poly
    c = poly.mean(axis=0)
    return c + (poly - c) * factor


@dataclass
class PitchKeyframe:
    frame: int
    polygon: np.ndarray  # (N, 2) normalised

    def to_json(self) -> dict:
        return {"frame": int(self.frame),
                "polygon": [[round(float(x), 6), round(float(y), 6)] for x, y in self.polygon]}


@dataclass
class PitchRegion:
    """Our pitch, as one or more keyed polygons in normalised frame coordinates."""

    keyframes: list[PitchKeyframe]
    video: str | None = None
    width: int | None = None
    height: int | None = None
    track_pan: bool = True
    margin: float = 0.0
    notes: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.keyframes:
            raise PitchRegionError("a pitch region needs at least one keyframe")
        self.keyframes = sorted(self.keyframes, key=lambda k: k.frame)
        n = len(self.keyframes[0].polygon)
        for kf in self.keyframes:
            if len(kf.polygon) != n:
                raise PitchRegionError(
                    "every keyframe must name the same corners in the same order "
                    f"(frame {self.keyframes[0].frame} has {n} points, "
                    f"frame {kf.frame} has {len(kf.polygon)}) — the polygon is "
                    "interpolated corner by corner between keyframes")

    # -- construction ----------------------------------------------------

    @classmethod
    def from_points(cls, polygon: np.ndarray, frame: int = 0, **kw) -> PitchRegion:
        kf = PitchKeyframe(frame=frame, polygon=np.asarray(polygon, float))
        return cls(keyframes=[kf], **kw)

    @classmethod
    def from_json(cls, data: dict) -> PitchRegion:
        kfs = data.get("keyframes")
        if not kfs:
            raise PitchRegionError("region file has no 'keyframes'")
        return cls(
            keyframes=[PitchKeyframe(frame=int(k["frame"]),
                                     polygon=np.asarray(k["polygon"], dtype=float))
                       for k in kfs],
            video=data.get("video"),
            width=data.get("width"),
            height=data.get("height"),
            track_pan=bool(data.get("track_pan", True)),
            margin=float(data.get("margin", 0.0)),
            notes=data.get("notes"),
            meta=data.get("meta", {}) or {},
        )

    @classmethod
    def load(cls, path: str | Path) -> PitchRegion:
        with open(path) as f:
            return cls.from_json(json.load(f))

    def to_json(self) -> dict:
        return {
            "video": self.video,
            "width": self.width,
            "height": self.height,
            "normalized": True,
            "track_pan": self.track_pan,
            "margin": self.margin,
            "notes": self.notes,
            "keyframes": [k.to_json() for k in self.keyframes],
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        """Write the region file. Points are kept one per line — this is a file
        people read and hand-edit, and an indented pair per coordinate is not."""
        import re

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_json(), indent=2)
        text = re.sub(r"\[\s+(-?[\d.]+),\s+(-?[\d.]+)\s+\]", r"[\1, \2]", text)
        path.write_text(text + "\n")
        return path

    def with_keyframe(self, polygon: np.ndarray, frame: int) -> PitchRegion:
        """Return a copy with another keyframe added (replacing one at the same frame)."""
        kfs = [k for k in self.keyframes if k.frame != frame]
        kfs.append(PitchKeyframe(frame=int(frame), polygon=np.asarray(polygon, float)))
        return PitchRegion(keyframes=kfs, video=self.video, width=self.width,
                           height=self.height, track_pan=self.track_pan,
                           margin=self.margin, notes=self.notes, meta=dict(self.meta))

    # -- evaluation ------------------------------------------------------

    def polygon_at(self, frame_no: int) -> np.ndarray:
        """Normalised polygon at ``frame_no``, interpolated between keyframes.

        Before the first keyframe and after the last, the nearest one is held —
        extrapolating a camera's pan past the range someone actually looked at
        would invent coverage we have no evidence for.
        """
        kfs = self.keyframes
        if len(kfs) == 1 or frame_no <= kfs[0].frame:
            return kfs[0].polygon
        if frame_no >= kfs[-1].frame:
            return kfs[-1].polygon
        for a, b in zip(kfs, kfs[1:]):
            if a.frame <= frame_no <= b.frame:
                span = b.frame - a.frame
                t = 0.0 if span == 0 else (frame_no - a.frame) / span
                return a.polygon * (1 - t) + b.polygon * t
        return kfs[-1].polygon  # unreachable

    def nearest_keyframe(self, frame_no: int) -> PitchKeyframe:
        return min(self.keyframes, key=lambda k: abs(k.frame - frame_no))

    def pixel_polygon(self, frame_shape: tuple[int, int], frame_no: int = 0) -> np.ndarray:
        """Polygon at ``frame_no`` in pixels for a frame of this shape."""
        h, w = frame_shape[:2]
        poly = _scale_about_centroid(self.polygon_at(frame_no), 1.0 + self.margin)
        return poly * np.array([float(w), float(h)])


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray casting: boolean mask of which ``points`` are inside ``polygon``.

    Both arrays are (N, 2) in the same coordinate system. Vectorised over points
    so this can run per detection frame without showing up in the profile.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)

    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)

    inside = np.zeros(len(pts), dtype=bool)
    for i in range(len(poly)):
        ax, ay, bx, by = x1[i], y1[i], x2[i], y2[i]
        if ay == by:
            continue
        straddles = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (bx - ax) * (y - ay) / (by - ay) + ax
        inside ^= straddles & (x < x_cross)
    return inside


# -- camera motion ------------------------------------------------------------


def estimate_similarity(ref_gray: np.ndarray, cur_gray: np.ndarray,
                        work_width: int = 640) -> np.ndarray | None:
    """Similarity transform mapping points in ``ref_gray`` to ``cur_gray``.

    Returns a 2x3 matrix in *full-resolution* coordinates, or None when the
    estimate is missing or implausible. ORB + RANSAC on a downscaled pair: the
    pitch, its lines and the fixed clutter around it give plenty of features, and
    the players moving inside it are outliers RANSAC is happy to discard.
    """
    import cv2

    h, w = ref_gray.shape[:2]
    s = min(1.0, work_width / float(w))
    if s < 1.0:
        ref_s = cv2.resize(ref_gray, (int(w * s), int(h * s)))
        cur_s = cv2.resize(cur_gray, (int(w * s), int(h * s)))
    else:
        ref_s, cur_s = ref_gray, cur_gray

    orb = cv2.ORB_create(1500)
    kp1, des1 = orb.detectAndCompute(ref_s, None)
    kp2, des2 = orb.detectAndCompute(cur_s, None)
    if des1 is None or des2 is None or len(kp1) < _MIN_INLIERS or len(kp2) < _MIN_INLIERS:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)[:400]
    if len(matches) < _MIN_INLIERS:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
    if M is None or inliers is None or int(inliers.sum()) < _MIN_INLIERS:
        return None

    # Undo the downscale: the linear part is scale-free, the translation is not.
    M = M.astype(float).copy()
    M[:, 2] /= max(s, 1e-9)

    scale = float(np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])))
    if not (_MIN_SCALE <= scale <= _MAX_SCALE):
        return None
    if abs(M[0, 2]) > _MAX_SHIFT_FRAC * w or abs(M[1, 2]) > _MAX_SHIFT_FRAC * h:
        return None
    return M


def apply_affine(poly_px: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly_px, dtype=float)
    return pts @ M[:, :2].T + M[:, 2]


def compose_affine(first: np.ndarray, then: np.ndarray) -> np.ndarray:
    """The 2x3 transform equivalent to applying ``first`` and then ``then``."""
    out = np.empty((2, 3), dtype=float)
    out[:, :2] = then[:, :2] @ first[:, :2]
    out[:, 2] = then[:, :2] @ first[:, 2] + then[:, 2]
    return out


class PitchRegionTracker:
    """Serves the pitch polygon for any frame, following the camera if asked.

    **Why it chains.** Matching every frame straight back to the keyframe would
    be drift-free, and it does not work: measured against a single reference on
    our own footage, ORB stops matching after ~5 s on the XbotGo (which pans
    ~130 px/s) and after ~5-10 s on the Veo. A 90-minute match is not one view.

    So the polygon is carried forward frame to frame, where consecutive
    detection samples are 0.2 s apart and match easily, and **re-anchored** to
    the keyframe whenever that match succeeds (every ``reanchor_every`` calls).
    Chaining accumulates drift; the re-anchor resets it whenever the camera
    happens to come back near where the region was drawn. Extra keyframes are
    the manual answer to the same problem and can be used together with this.

    Failure is inert, not destructive: an unusable estimate (too few inliers,
    >1/4-frame jump, >±25% zoom) is discarded and the last good transform is
    held, so a bad match leaves the polygon where it was.
    """

    def __init__(self, region: PitchRegion, video_path: str | Path | None = None,
                 track_pan: bool | None = None, refresh_every: int = 1,
                 reanchor_every: int = 25):
        self.region = region
        self.video_path = Path(video_path) if video_path else None
        self.track_pan = region.track_pan if track_pan is None else bool(track_pan)
        self.refresh_every = max(1, int(refresh_every))
        self.reanchor_every = max(0, int(reanchor_every))
        self._refs: dict[int, np.ndarray | None] = {}
        self._last_M: dict[int, np.ndarray] = {}
        self._prev_gray: np.ndarray | None = None
        self._prev_key: int | None = None
        self._since_anchor = 0
        self._last_frame_no: int | None = None
        self.n_tracked = 0
        self.n_chained = 0
        self.n_reanchored = 0
        self.n_failed = 0

    def _reference_gray(self, kf_frame: int) -> np.ndarray | None:
        """Greyscale of the frame the polygon was drawn on (read once, cached)."""
        if kf_frame in self._refs:
            return self._refs[kf_frame]

        gray = None
        if self.video_path is not None:
            import cv2

            from soccer_vision.io.video import VideoReader

            try:
                with VideoReader(self.video_path) as r:
                    img = r.read_frame(kf_frame)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = None
        self._refs[kf_frame] = gray
        return gray

    def polygon_for(self, frame_no: int, frame: np.ndarray | None = None,
                    frame_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Pixel polygon for ``frame_no``; pass ``frame`` to enable pan tracking."""
        shape = frame.shape if frame is not None else frame_shape
        if shape is None:
            raise ValueError("polygon_for needs either a frame or a frame_shape")
        poly = self.region.pixel_polygon(shape, frame_no)

        if not self.track_pan or frame is None:
            return poly

        kf = self.region.nearest_keyframe(frame_no)
        M = self._last_M.get(kf.frame)
        due = (self._last_frame_no is None
               or abs(frame_no - self._last_frame_no) >= self.refresh_every)
        if due:
            M = self._update(kf.frame, frame)
            self._last_frame_no = frame_no

        return apply_affine(poly, M) if M is not None else poly

    def _update(self, kf_frame: int, frame: np.ndarray) -> np.ndarray | None:
        """Advance the keyframe→current transform using this frame."""
        import cv2

        cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        M = self._last_M.get(kf_frame)
        switched = self._prev_key != kf_frame
        self._prev_key = kf_frame

        # Re-anchor: match straight back to the frame the polygon was drawn on.
        # Drift-free when it lands, which is only while the camera is still
        # looking somewhere near there — hence the chain below.
        want_anchor = (M is None or switched or self.reanchor_every == 0
                       or self._since_anchor >= self.reanchor_every)
        if want_anchor:
            ref = self._reference_gray(kf_frame)
            if ref is not None:
                if ref.shape[:2] != cur.shape[:2]:
                    ref = cv2.resize(ref, (cur.shape[1], cur.shape[0]))
                    self._refs[kf_frame] = ref
                anchored = estimate_similarity(ref, cur)
                if anchored is not None:
                    self._last_M[kf_frame] = anchored
                    self._prev_gray = cur
                    self._since_anchor = 0
                    self.n_tracked += 1
                    self.n_reanchored += 1
                    return anchored
            # Anchor didn't land — wait a full interval before paying for it
            # again rather than retrying a hopeless match every frame.
            self._since_anchor = 0

        # Chain: consecutive detection samples are ~0.2 s apart, so this match is
        # easy where the anchor match is hopeless. Drift accumulates until the
        # next successful re-anchor.
        if M is not None and self._prev_gray is not None:
            step = estimate_similarity(self._prev_gray, cur)
            self._prev_gray = cur  # next hop should be short even if this one failed
            if step is not None:
                M = compose_affine(M, step)
                self._last_M[kf_frame] = M
                self._since_anchor += 1
                self.n_tracked += 1
                self.n_chained += 1
                return M

        # Nothing usable: hold the last good transform. A failed match means we
        # cannot see where the camera went, not that it went home.
        self.n_failed += 1
        return M

    def contains(self, points: np.ndarray, frame_no: int,
                 frame: np.ndarray | None = None,
                 frame_shape: tuple[int, int] | None = None) -> np.ndarray:
        poly = self.polygon_for(frame_no, frame=frame, frame_shape=frame_shape)
        return points_in_polygon(points, poly)


def load_tracker(spec: str | Path | PitchRegion, video_path: str | Path | None = None,
                 track_pan: bool | None = None) -> PitchRegionTracker:
    region = spec if isinstance(spec, PitchRegion) else PitchRegion.load(spec)
    return PitchRegionTracker(region, video_path=video_path, track_pan=track_pan)


# -- drawing ------------------------------------------------------------------


def draw_region(frame: np.ndarray, poly_px: np.ndarray, color=(255, 200, 0),
                alpha: float = 0.15, label: str | None = None) -> np.ndarray:
    """Return a copy of ``frame`` with the region shaded and outlined."""
    import cv2

    out = frame.copy()
    pts = np.asarray(poly_px, dtype=np.int32).reshape(-1, 1, 2)
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=3)
    for i, (x, y) in enumerate(np.asarray(poly_px, dtype=int)):
        cv2.circle(out, (int(x), int(y)), 7, color, -1)
        cv2.putText(out, str(i + 1), (int(x) + 10, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if label:
        cv2.putText(out, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
    return out


def draw_coordinate_grid(frame: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Overlay a labelled normalised-coordinate grid on a reference frame.

    This is the headless path: nobody can click on a frame over SSH, but anyone
    (or a vision model) can read "the near touchline runs along y=0.55" off a
    gridded still and hand back four ``x,y`` pairs.
    """
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    n = int(round(1.0 / step))
    for i in range(n + 1):
        f = i * step
        x, y = int(f * (w - 1)), int(f * (h - 1))
        major = i % 5 == 0
        col = (255, 255, 255) if major else (170, 170, 170)
        thick = 2 if major else 1
        cv2.line(out, (x, 0), (x, h), col, thick, cv2.LINE_AA)
        cv2.line(out, (0, y), (w, y), col, thick, cv2.LINE_AA)
        if major:
            # Nudge the y=0.0 label below the x-axis row so the two don't collide
            # in the corner, where they are least readable and most needed.
            y_org = (6, y - 6 if y > 40 else y + 46)
            for text, org in ((f"x={f:.1f}", (min(x + 6, w - 90), 26)), (f"y={f:.1f}", y_org)):
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
    return out
