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


# Reference BGR colours for kit names a team profile may declare. Naming a
# cluster against these in Lab space is robust to camera white balance: a black
# kit that photographs as dark navy is still far closer to the black reference
# than to the white one, even though its raw hue reads "blue".
KIT_REFERENCE_BGR: dict[str, tuple[int, int, int]] = {
    "black": (10, 10, 10),
    "white": (245, 245, 245),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "red": (40, 30, 200),
    "maroon": (30, 20, 110),
    "orange": (30, 130, 240),
    "yellow": (40, 210, 230),
    "green": (60, 160, 40),
    "cyan": (200, 200, 60),
    "blue": (200, 90, 30),
    "navy": (110, 45, 20),
    "purple": (170, 50, 120),
    "pink": (200, 130, 240),
}


def _bgr_to_lab(colours: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) BGR array to Lab float32."""
    as_img = np.clip(colours, 0, 255).reshape(-1, 1, 3).astype(np.uint8)
    return cv2.cvtColor(as_img, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)


def assign_kit_names(centroids_bgr: np.ndarray, kits: list[str]) -> list[str] | None:
    """Name each cluster centroid after the declared kit nearest in Lab space.

    For the two-team case the assignment is the distinct pairing that minimises
    total Lab distance, so both clusters can never receive the same kit name.
    Returns ``None`` when a kit name is unknown or counts don't line up, in
    which case the caller should fall back to heuristic naming.
    """
    if len(kits) != len(centroids_bgr) or len(kits) == 0:
        return None
    if any(kit not in KIT_REFERENCE_BGR for kit in kits):
        return None

    refs = _bgr_to_lab(np.array([KIT_REFERENCE_BGR[kit] for kit in kits], dtype=np.float32))
    cents = _bgr_to_lab(np.asarray(centroids_bgr, dtype=np.float32))
    # distances[i][j] = Lab distance from centroid i to kit j
    distances = np.linalg.norm(cents[:, None, :] - refs[None, :, :], axis=2)

    if len(kits) == 1:
        return [kits[0]]
    # Two clusters, two kits: pick the cheaper of the two pairings.
    straight = distances[0, 0] + distances[1, 1]
    crossed = distances[0, 1] + distances[1, 0]
    return [kits[0], kits[1]] if straight <= crossed else [kits[1], kits[0]]


class TeamClassifier:
    """Assign track IDs to one of two teams by accumulated jersey colour."""

    _PREVIEW_CROPS_PER_TRACK = 2
    _PREVIEW_CROP_SIZE = (48, 64)  # (width, height)

    def __init__(self, min_samples: int = 3, kits: list[str] | None = None):
        self.min_samples = min_samples
        self.kits = [k.strip().lower() for k in kits] if kits else []
        self._samples: dict[int, list[np.ndarray]] = {}
        self._crops: dict[int, list[np.ndarray]] = {}
        self._track_team: dict[int, str] = {}
        self._team_names: dict[str, str] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._fitted = False

    def add_sample(self, track_id: int, frame: np.ndarray, bbox) -> None:
        """Record one torso-colour observation for a track."""
        colour = sample_jersey_bgr(frame, bbox)
        if colour is None:
            return
        tid = int(track_id)
        self._samples.setdefault(tid, []).append(colour)
        crops = self._crops.setdefault(tid, [])
        if len(crops) < self._PREVIEW_CROPS_PER_TRACK:
            crop = self._torso_crop(frame, bbox)
            if crop is not None:
                crops.append(crop)

    @staticmethod
    def _torso_crop(frame: np.ndarray, bbox) -> np.ndarray | None:
        """Small torso thumbnail for the teams preview montage."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox)
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 8:
            return None
        tx1 = max(0, int(round(x1 + bw * _TORSO_SIDE)))
        tx2 = min(w, int(round(x2 - bw * _TORSO_SIDE)))
        ty1 = max(0, int(round(y1 + bh * _TORSO_TOP)))
        ty2 = min(h, int(round(y1 + bh * _TORSO_BOTTOM)))
        if tx2 - tx1 < 2 or ty2 - ty1 < 2:
            return None
        return cv2.resize(frame[ty1:ty2, tx1:tx2], TeamClassifier._PREVIEW_CROP_SIZE)

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
            kit_names = assign_kit_names(colours[:1], self.kits[:1]) if self.kits else None
            name = kit_names[0] if kit_names else name_bgr_colour(colours[0])
            self._track_team[ids[0]] = "team_a"
            self._team_names = {"team_a": name}
            self._centroids = {"team_a": colours[0]}
            return self

        lab = cv2.cvtColor(colours.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(lab, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()

        for tid, lbl in zip(ids, labels):
            self._track_team[tid] = "team_a" if lbl == 0 else "team_b"

        # Compute each cluster's centroid colour, then name the clusters.
        keys = []
        centroid_list = []
        for cluster, key in ((0, "team_a"), (1, "team_b")):
            members = colours[labels == cluster]
            if len(members):
                keys.append(key)
                centroid_list.append(members.mean(axis=0))
                self._centroids[key] = members.mean(axis=0)

        # Declared profile kits are authoritative; the HSV heuristic is only a
        # fallback for runs without a profile (see issue #11 — a black kit that
        # photographs as navy must not be named "blue").
        kit_names = assign_kit_names(np.array(centroid_list), self.kits) if self.kits else None
        for index, key in enumerate(keys):
            if kit_names:
                self._team_names[key] = kit_names[index]
            else:
                self._team_names[key] = name_bgr_colour(centroid_list[index])
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

    def render_preview(self, max_per_team: int = 8) -> np.ndarray | None:
        """Montage of torso crops per team with the assigned name and swatch.

        One row per team: up to ``max_per_team`` sampled torso thumbnails from
        distinct tracks, the cluster's centroid colour as a swatch, and the
        assigned name. Lets a headless run be verified at a glance.
        """
        if not self._fitted or not self._team_names:
            return None

        crop_w, crop_h = self._PREVIEW_CROP_SIZE
        label_w = 160
        pad = 6
        row_w = label_w + (crop_w + pad) * max_per_team + pad
        row_h = crop_h + 2 * pad
        canvas = np.full((row_h * len(self._team_names), row_w, 3), 24, dtype=np.uint8)

        for row, key in enumerate(sorted(self._team_names)):
            y0 = row * row_h
            centroid = self._centroids.get(key)
            if centroid is not None:
                swatch = tuple(int(c) for c in centroid)
                cv2.rectangle(canvas, (pad, y0 + pad), (42, y0 + pad + 36), swatch, -1)
            cv2.putText(
                canvas,
                f"{key}: {self._team_names[key]}",
                (48, y0 + row_h // 2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            member_ids = [tid for tid, k in self._track_team.items() if k == key]
            x = label_w
            shown = 0
            for tid in member_ids:
                for crop in self._crops.get(tid, [])[:1]:
                    if shown >= max_per_team:
                        break
                    canvas[y0 + pad : y0 + pad + crop_h, x : x + crop_w] = crop
                    x += crop_w + pad
                    shown += 1
                if shown >= max_per_team:
                    break
        return canvas
