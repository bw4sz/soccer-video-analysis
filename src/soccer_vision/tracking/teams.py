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


def sample_jersey_bgr(
    frame: np.ndarray, bbox, mask: np.ndarray | None = None
) -> np.ndarray | None:
    """Median BGR colour of a player's torso region.

    ``bbox`` is (x1, y1, x2, y2) in pixel coordinates. Returns ``None`` when the
    box is too small or falls outside the frame.

    ``mask`` is an optional full-frame boolean segmentation mask for this player
    (SAM3 supplies one per track). When given, the median is taken over *player
    pixels only*. On overhead footage a player is small and the torso rectangle
    is mostly turf, so the rectangular median drags every jersey toward green
    and the two team clusters collapse into one colour — job 37877642 named both
    teams "blue". Masking the sample removes the background entirely.
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

    patch = frame[ty1:ty2, tx1:tx2]
    if mask is not None:
        m = mask[ty1:ty2, tx1:tx2]
        # Need a few real player pixels to be meaningful; otherwise fall back
        # to the rectangular median rather than returning noise.
        if int(m.sum()) >= 4:
            return np.median(patch[m], axis=0)
    return np.median(patch.reshape(-1, 3), axis=0)


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


# Representative BGR for common kit colour names. Used to name a cluster from a
# team profile's declared kits (``kits: [black, white]``) instead of the
# camera-dependent HSV heuristic in ``name_bgr_colour`` — the black gate in that
# heuristic (value < 50) mislabels a black kit that a camera renders as mid-value
# navy (see GitHub issue #11). Kit lists are short and usually far apart in
# colour space, so nearest-in-Lab assignment against these references is robust
# even when the raw torso pixels are off (e.g. black photographed as navy).
_KIT_REF_BGR: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (190, 190, 190),
    "red": (36, 36, 190),
    "maroon": (40, 40, 110),
    "blue": (190, 90, 40),
    "navy": (70, 40, 20),
    "sky": (210, 170, 90),
    "cyan": (200, 200, 40),
    "green": (50, 150, 50),
    "yellow": (40, 220, 225),
    "gold": (40, 190, 220),
    "orange": (30, 120, 230),
    "purple": (140, 40, 120),
    "pink": (150, 120, 230),
}


def kit_reference_bgr(name: str) -> tuple[int, int, int] | None:
    """Reference BGR for a kit-colour name, or ``None`` if unknown."""
    return _KIT_REF_BGR.get(name.strip().lower())


def _bgr_to_lab(bgr) -> np.ndarray:
    px = np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)


def assign_kits_to_clusters(
    centroids_bgr: list, kits: list[str] | None
) -> dict[int, str]:
    """Map cluster centroid colours to declared kit names by nearest Lab match.

    ``centroids_bgr`` is one BGR triple per cluster; ``kits`` is the profile's
    declared kit-name list. Returns ``{cluster_index: kit_name}`` using the
    globally lowest-cost one-to-one assignment, so two clusters never collapse to
    the same name. Kit names with no reference colour are ignored; an empty result
    means the caller should fall back to the HSV heuristic.
    """
    from itertools import permutations

    known = [k for k in (kits or []) if kit_reference_bgr(k) is not None]
    n = len(centroids_bgr)
    if not known or n == 0:
        return {}
    cen_lab = [_bgr_to_lab(c) for c in centroids_bgr]
    kit_lab = {k: _bgr_to_lab(kit_reference_bgr(k)) for k in known}

    def dist(i: int, k: str) -> float:
        return float(np.linalg.norm(cen_lab[i] - kit_lab[k]))

    if len(known) >= n:
        best: tuple[float, tuple[str, ...]] | None = None
        for combo in permutations(known, n):
            cost = sum(dist(i, combo[i]) for i in range(n))
            if best is None or cost < best[0]:
                best = (cost, combo)
        assert best is not None
        return {i: best[1][i] for i in range(n)}
    # Fewer declared kits than clusters: nearest kit per cluster (reuse allowed).
    return {i: min(known, key=lambda k: dist(i, k)) for i in range(n)}


class TeamClassifier:
    """Assign track IDs to one of two teams by accumulated jersey colour."""

    def __init__(self, min_samples: int = 3, keep_crops: int = 4):
        self.min_samples = min_samples
        self.keep_crops = keep_crops
        self._samples: dict[int, list[np.ndarray]] = {}
        self._crops: dict[int, list[np.ndarray]] = {}
        self._track_team: dict[int, str] = {}
        self._team_names: dict[str, str] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._fitted = False

    def add_sample(self, track_id: int, frame: np.ndarray, bbox,
                   mask: np.ndarray | None = None) -> None:
        """Record one torso-colour observation for a track.

        ``mask`` (optional, from a segmentation detector) restricts the colour
        sample to player pixels — see :func:`sample_jersey_bgr`.
        """
        colour = sample_jersey_bgr(frame, bbox, mask)
        if colour is not None:
            tid = int(track_id)
            self._samples.setdefault(tid, []).append(colour)
            crops = self._crops.setdefault(tid, [])
            if self.keep_crops and len(crops) < self.keep_crops:
                x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    crops.append(frame[y1:y2, x1:x2].copy())

    def _track_colours(self) -> tuple[list[int], np.ndarray]:
        ids, colours = [], []
        for tid, samples in self._samples.items():
            if len(samples) >= self.min_samples:
                ids.append(tid)
                colours.append(np.median(np.stack(samples), axis=0))
        return ids, (np.stack(colours) if colours else np.empty((0, 3)))

    def fit(self, kits: list[str] | None = None) -> "TeamClassifier":
        """Cluster tracks into two teams (k-means in Lab colour space).

        When ``kits`` (a team profile's declared kit-colour names, e.g.
        ``["black", "white"]``) is given, each cluster is *named* by nearest-Lab
        match to a declared kit rather than the camera-dependent HSV heuristic —
        so a black kit a camera renders as navy is still named "black". Clustering
        is identical either way; only the cluster→name label changes. Falls back
        to ``name_bgr_colour`` per cluster when no kit matches (or ``kits`` is
        None), preserving the prior behaviour.
        """
        ids, colours = self._track_colours()
        self._fitted = True
        if len(ids) == 0:
            return self

        if len(ids) == 1:
            mapping = assign_kits_to_clusters([colours[0]], kits)
            name = mapping.get(0) or name_bgr_colour(colours[0])
            self._track_team[ids[0]] = "team_a"
            self._team_names = {"team_a": name}
            self._centroids = {"team_a": np.asarray(colours[0], dtype=float)}
            return self

        lab = cv2.cvtColor(colours.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(lab, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()

        for tid, lbl in zip(ids, labels):
            self._track_team[tid] = "team_a" if lbl == 0 else "team_b"

        # Cluster mean BGR per team, then name. Prefer nearest-Lab match to the
        # profile's declared kits (one-to-one, so the two teams can't collapse to
        # the same name); fall back to the HSV heuristic for any cluster with no
        # kit match. Assign both clusters together so the matching is global.
        means: dict[str, np.ndarray] = {}
        for cluster, key in ((0, "team_a"), (1, "team_b")):
            members = colours[labels == cluster]
            if len(members):
                means[key] = members.mean(axis=0)
        self._centroids = {k: np.asarray(v, dtype=float) for k, v in means.items()}

        ordered = [k for k in ("team_a", "team_b") if k in means]
        mapping = assign_kits_to_clusters([means[k] for k in ordered], kits)
        for idx, key in enumerate(ordered):
            self._team_names[key] = mapping.get(idx) or name_bgr_colour(means[key])
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

    def centroids(self) -> dict[str, np.ndarray]:
        """Mean torso BGR per team cluster (for previews / diagnostics)."""
        return {k: v.copy() for k, v in self._centroids.items()}

    def build_team_preview(
        self, out_path, per_team: int = 8, tile=(64, 96)
    ) -> bool:
        """Write a montage of torso crops grouped by team, for eyeball review.

        One row per team: the assigned colour name, the centroid swatch, then up
        to ``per_team`` sampled torso crops. Lets a user confirm the cluster→name
        mapping at a glance (headless-friendly — open the PNG after a run) and is
        the surface an interactive labeller would reuse. Returns ``False`` when no
        crops were retained (``keep_crops=0``) so nothing was written.
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called before build_team_preview()")
        tw, th = tile
        rows: list[np.ndarray] = []
        by_team: dict[str, list[int]] = {}
        for tid, key in self._track_team.items():
            by_team.setdefault(key, []).append(tid)
        if not any(self._crops.get(t) for tids in by_team.values() for t in tids):
            return False

        pad = 6
        for key in ("team_a", "team_b"):
            tids = by_team.get(key)
            if not tids:
                continue
            name = self._team_names.get(key, key)
            swatch_bgr = self._centroids.get(key)
            # Header strip: name + centroid swatch.
            header = np.full((th, tw, 3), 30, np.uint8)
            if swatch_bgr is not None:
                header[:] = tuple(int(c) for c in swatch_bgr)
            cv2.putText(header, name, (4, th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(header, name, (4, th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles = [header]
            # Representative crops from the tracks with the most samples.
            for tid in sorted(tids, key=lambda t: -len(self._crops.get(t, [])))[:per_team]:
                crop_list = self._crops.get(tid, [])
                if not crop_list:
                    continue
                crop = max(crop_list, key=lambda c: c.shape[0] * c.shape[1])
                tiles.append(cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA))
            row = np.full((th, (tw + pad) * len(tiles) - pad, 3), 20, np.uint8)
            x = 0
            for t in tiles:
                row[:, x:x + tw] = t
                x += tw + pad
            rows.append(row)

        if not rows:
            return False
        width = max(r.shape[1] for r in rows)
        canvas = np.full((sum(r.shape[0] for r in rows) + pad * (len(rows) - 1),
                          width, 3), 20, np.uint8)
        y = 0
        for r in rows:
            canvas[y:y + r.shape[0], 0:r.shape[1]] = r
            y += r.shape[0] + pad
        return bool(cv2.imwrite(str(out_path), canvas))
