"""Team assignment by jersey colour.

v1 team identity: cluster tracked players into two teams by the dominant colour
of their torso region, then map each cluster to a human colour name (blue, white,
red, ...) so events can be filtered by e.g. ``--team blue``.

This needs no extra models — it runs on the RF-DETR player boxes and ByteTrack
IDs already produced by the pipeline. Stable per-player identity (jersey OCR,
sn-gamestate, SAM3 masklets) is a later phase; see ``tracking/sam3.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

# Torso window inside a player bounding box (fractions of box height/width).
# Avoids head/shorts/legs and grabs the shirt.
_TORSO_TOP = 0.20
_TORSO_BOTTOM = 0.55
_TORSO_SIDE = 0.20


def sample_jersey_bgr(frame: np.ndarray, bbox) -> np.ndarray | None:
    """Median BGR colour of a player's torso region.

    ``bbox`` is (x1, y1, x2, y2) in pixel coordinates. Returns ``None`` when the
    box is too small or falls outside the frame.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 8:
        return None

    tx1 = int(round(x1 + bw * _TORSO_SIDE))
    tx2 = int(round(x2 - bw * _TORSO_SIDE))
    ty1 = int(round(y1 + bh * _TORSO_TOP))
    ty2 = int(round(y1 + bh * _TORSO_BOTTOM))

    tx1, tx2 = max(0, tx1), min(w, tx2)
    ty1, ty2 = max(0, ty1), min(h, ty2)
    if tx2 - tx1 < 2 or ty2 - ty1 < 2:
        return None

    patch = frame[ty1:ty2, tx1:tx2].reshape(-1, 3)
    return np.median(patch, axis=0)


def name_bgr_colour(bgr: np.ndarray) -> str:
    """Map a BGR colour to a coarse human colour name."""
    px = np.uint8([[bgr]])  # 1x1x3
    h, s, v = (int(c) for c in cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0])

    if v < 50:
        return "black"
    if s < 45:
        return "white" if v > 150 else "gray"

    # OpenCV hue is 0-179.
    if h < 10 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 160:
        return "purple"
    return "pink"


class TeamClassifier:
    """Assign track IDs to one of two teams by accumulated jersey colour."""

    def __init__(self, min_samples: int = 3):
        self.min_samples = min_samples
        self._samples: dict[int, list[np.ndarray]] = {}
        self._track_team: dict[int, str] = {}
        self._team_names: dict[str, str] = {}
        self._fitted = False

    def add_sample(self, track_id: int, frame: np.ndarray, bbox) -> None:
        """Record one torso-colour observation for a track."""
        colour = sample_jersey_bgr(frame, bbox)
        if colour is not None:
            self._samples.setdefault(int(track_id), []).append(colour)

    def _track_colours(self) -> tuple[list[int], np.ndarray]:
        ids, colours = [], []
        for tid, samples in self._samples.items():
            if len(samples) >= self.min_samples:
                ids.append(tid)
                colours.append(np.median(np.stack(samples), axis=0))
        return ids, (np.stack(colours) if colours else np.empty((0, 3)))

    def fit(self) -> "TeamClassifier":
        """Cluster tracks into two teams (k-means in Lab colour space)."""
        ids, colours = self._track_colours()
        self._fitted = True
        if len(ids) == 0:
            return self

        if len(ids) == 1:
            name = name_bgr_colour(colours[0])
            self._track_team[ids[0]] = "team_a"
            self._team_names = {"team_a": name}
            return self

        lab = cv2.cvtColor(colours.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(lab, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()

        for tid, lbl in zip(ids, labels):
            self._track_team[tid] = "team_a" if lbl == 0 else "team_b"

        # Name each cluster by the mean BGR of its members.
        for cluster, key in ((0, "team_a"), (1, "team_b")):
            members = colours[labels == cluster]
            if len(members):
                self._team_names[key] = name_bgr_colour(members.mean(axis=0))
        return self

    def predict(self, track_id: int) -> str | None:
        """Return the team colour name for a track, or ``None`` if unknown."""
        if not self._fitted:
            raise RuntimeError("TeamClassifier.fit() must be called before predict()")
        key = self._track_team.get(int(track_id))
        return self._team_names.get(key) if key else None

    def team_names(self) -> dict[str, str]:
        """Map internal cluster keys (team_a/team_b) to colour names."""
        return dict(self._team_names)
