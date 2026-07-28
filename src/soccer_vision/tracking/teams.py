"""Team assignment by jersey colour.

v1 team identity: cluster tracked players into two teams by the dominant colour
of their torso region, then map each cluster to a human colour name (blue, white,
red, ...) so events can be filtered by e.g. ``--team blue``.

This needs no extra models — it runs on the RF-DETR player boxes and ByteTrack
IDs already produced by the pipeline. Stable per-player identity is a separate
step; see ``cli/identify.py`` (jersey OCR) and ``cli/enroll.py`` (re-id).
"""

from __future__ import annotations

import cv2
import numpy as np

# Torso window inside a player bounding box (fractions of box height/width).
# Avoids head/shorts/legs and grabs the shirt.
_TORSO_TOP = 0.20
_TORSO_BOTTOM = 0.55
_TORSO_SIDE = 0.20

# Turf is identified by hue, not brightness. A low sun puts grass anywhere from
# bright yellow-green to near-black shade (local turf L* ranged 50-101 across a
# single U14G clip), so gating on value would classify shadowed grass as "not
# turf" — and, worse, a black kit is genuinely dark and must never be mistaken
# for shade. Hue plus a modest saturation floor is the part that holds up.
_TURF_HUE_LO, _TURF_HUE_HI = 30, 95
_TURF_SAT_MIN = 50
_MIN_TORSO_PIXELS = 8
_MIN_TURF_PIXELS = 20
# Below this many successful illuminant reads across the whole match, shadow
# correction is switched off rather than anchored to a noisy reference.
_MIN_ILLUM_SAMPLES = 20
# Fraction of tracks that must have at least one turf reading before the
# grass-relative split is trusted for the match as a whole.
_MIN_ILLUM_COVERAGE = 0.6


def turf_pixels(patch_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of grass-looking pixels in a BGR patch."""
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    return (hue >= _TURF_HUE_LO) & (hue <= _TURF_HUE_HI) & (sat >= _TURF_SAT_MIN)


def _torso_window(frame: np.ndarray, bbox):
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
    return tx1, ty1, tx2, ty2


def sample_jersey_bgr(
    frame: np.ndarray, bbox, mask: np.ndarray | None = None
) -> np.ndarray | None:
    """Median BGR colour of a player's torso region.

    ``bbox`` is (x1, y1, x2, y2) in pixel coordinates. Returns ``None`` when the
    box is too small or falls outside the frame.

    ``mask`` is an optional full-frame boolean segmentation mask for this player,
    for a segmentation detector that supplies one (RF-DETR does not). When given,
    the median is taken over *player
    pixels only*. On overhead footage a player is small and the torso rectangle
    is mostly turf, so a plain rectangular median drags every jersey toward green
    and the two team clusters collapse into one colour — job 37877642 named both
    teams "blue".

    With no mask (the RF-DETR path) the grass is excluded by hue instead, which
    recovers most of that benefit without a segmentation model. It is not the
    whole fix: see :func:`estimate_local_illuminant` for the shadow problem,
    which dominates on low-sun footage.
    """
    win = _torso_window(frame, bbox)
    if win is None:
        return None
    tx1, ty1, tx2, ty2 = win
    patch = frame[ty1:ty2, tx1:tx2]

    if mask is not None:
        m = mask[ty1:ty2, tx1:tx2]
        # Need a few real player pixels to be meaningful; otherwise fall back
        # to the rectangular median rather than returning noise.
        if int(m.sum()) >= 4:
            return np.median(patch[m], axis=0)
    else:
        keep = ~turf_pixels(patch)
        if int(keep.sum()) >= _MIN_TORSO_PIXELS:
            return np.median(patch[keep], axis=0)
    return np.median(patch.reshape(-1, 3), axis=0)


def estimate_local_illuminant(frame: np.ndarray, bbox) -> np.ndarray | None:
    """Median BGR of the turf immediately around a player, or ``None``.

    This is the lighting that player is standing in. It matters because kit
    colour is judged by *lightness*, and lightness is exactly what a low sun
    destroys: on the U14G clip a white kit in shade reads blue-grey and lands in
    the same Lab neighbourhood as a black kit in sun, so absolute torso colour
    put 174 of 188 tracks in one cluster. Grass around the player shares that
    player's illumination, so it is a usable reference — normalising by it turns
    "is this shirt bright?" into "is this shirt brighter than the grass it is
    standing on?", which is stable under shadow and separates the kits at 0.

    Returns ``None`` when too little grass is visible (indoor, snow, a synthetic
    test frame), and callers must fall back to uncorrected colour.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 8:
        return None
    # A ring one box-width to each side and 30% of box height above/below —
    # wide enough to contain grass even when players are close together.
    rx1, rx2 = max(0, int(x1 - bw)), min(w, int(x2 + bw))
    ry1, ry2 = max(0, int(y1 - 0.3 * bh)), min(h, int(y2 + 0.3 * bh))
    ring = frame[ry1:ry2, rx1:rx2]
    if ring.size == 0:
        return None
    grass = turf_pixels(ring)
    if int(grass.sum()) < _MIN_TURF_PIXELS:
        return None
    return np.median(ring[grass], axis=0)


def correct_illumination(
    colour_bgr: np.ndarray, local_illum: np.ndarray, reference_illum: np.ndarray
) -> np.ndarray:
    """Rescale a torso colour as if it were lit like ``reference_illum``.

    Per-channel von Kries adaptation: ``colour * (reference / local)``. Dividing
    channel-wise rather than scaling luminance also removes the *colour cast* —
    a low sun is warm and its shadows are blue, which is why an unlit white kit
    reads blue-grey in the first place.
    """
    local = np.maximum(np.asarray(local_illum, dtype=float), 1.0)
    gain = np.asarray(reference_illum, dtype=float) / local
    return np.clip(np.asarray(colour_bgr, dtype=float) * gain, 0, 255)


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


# Representative sunlit grass, used only to ask whether a declared kit is
# lighter or darker than the pitch — never as a measurement.
_TURF_REF_BGR = (50, 150, 50)


def kit_lighter_than_turf(name: str) -> bool | None:
    """Is this kit colour lighter than grass? ``None`` if the kit is unknown."""
    ref = kit_reference_bgr(name)
    if ref is None:
        return None
    return _bgr_to_lab(ref)[0] > _bgr_to_lab(_TURF_REF_BGR)[0]


def lightness_split_kits(kits: list[str] | None) -> tuple[str, str] | None:
    """``(darker_kit, lighter_kit)`` when two declared kits straddle the turf.

    Returns ``None`` unless exactly one declared kit is darker than grass and
    exactly one is lighter — the case where the *sign* of a player's lightness
    relative to the grass they stand on identifies their team outright. Black vs
    white and blue vs white qualify; red vs blue does not (both are darker than
    grass), and there hue is the separator, so the caller falls back to
    clustering.
    """
    known = [k for k in (kits or []) if kit_reference_bgr(k) is not None]
    if len(known) != 2:
        return None
    lighter = [k for k in known if kit_lighter_than_turf(k)]
    darker = [k for k in known if not kit_lighter_than_turf(k)]
    if len(lighter) != 1 or len(darker) != 1:
        return None
    return darker[0], lighter[0]


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
        self._illum: dict[int, list[np.ndarray | None]] = {}
        self._crops: dict[int, list[np.ndarray]] = {}
        self._track_team: dict[int, str] = {}
        self._team_names: dict[str, str] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._fitted = False
        self._reference_illum: np.ndarray | None = None
        self._split_method = "colour clustering"
        self._forced_names: dict[str, str] = {}

    def add_sample(self, track_id: int, frame: np.ndarray, bbox,
                   mask: np.ndarray | None = None) -> None:
        """Record one torso-colour observation for a track.

        ``mask`` (optional, from a segmentation detector) restricts the colour
        sample to player pixels — see :func:`sample_jersey_bgr`.

        The local illuminant is recorded alongside each colour so :meth:`fit` can
        normalise for shadow; it cannot be applied here because the reference it
        normalises *to* is only known once every sample is in.
        """
        colour = sample_jersey_bgr(frame, bbox, mask)
        if colour is not None:
            tid = int(track_id)
            self._samples.setdefault(tid, []).append(colour)
            self._illum.setdefault(tid, []).append(
                estimate_local_illuminant(frame, bbox)
            )
            crops = self._crops.setdefault(tid, [])
            if self.keep_crops and len(crops) < self.keep_crops:
                x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    crops.append(frame[y1:y2, x1:x2].copy())

    def _fit_reference_illuminant(self) -> np.ndarray | None:
        """Median turf colour across every sample — the lighting we normalise to.

        Using the match's own median rather than a fixed constant keeps corrected
        colours near their raw values, so the kit-naming references in
        ``_KIT_REF_BGR`` stay meaningful.
        """
        seen = [i for lst in self._illum.values() for i in lst if i is not None]
        if len(seen) < _MIN_ILLUM_SAMPLES:
            return None
        return np.median(np.stack(seen), axis=0)

    def _track_relative_lightness(self) -> dict[int, float]:
        """Per track: median (torso L*) - (local turf L*), where turf was found.

        Positive means the player is lighter than the grass they are standing on.
        Unlike absolute lightness this survives shadow, which is what makes it
        usable on low-sun footage.
        """
        out: dict[int, float] = {}
        for tid, samples in self._samples.items():
            if len(samples) < self.min_samples:
                continue
            illums = self._illum.get(tid) or []
            vals = [
                float(_bgr_to_lab(c)[0] - _bgr_to_lab(il)[0])
                for c, il in zip(samples, illums)
                if il is not None
            ]
            if vals:
                out[tid] = float(np.median(vals))
        return out

    @staticmethod
    def _place_uncovered(labels: np.ndarray, colours: np.ndarray) -> np.ndarray:
        """Assign tracks marked -1 to the nearer of the two group centroids.

        These are the tracks with no turf reading, so the grass-relative sign is
        unavailable and their illumination-corrected colour is the best remaining
        evidence.
        """
        known = labels >= 0
        if not known.any() or known.all():
            return labels
        lab = cv2.cvtColor(colours.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
        out = labels.copy()
        centroids = {}
        for k in (0, 1):
            members = lab[known & (labels == k)]
            if len(members):
                centroids[k] = members.mean(axis=0)
        if not centroids:
            return labels
        for i in np.where(~known)[0]:
            out[i] = min(centroids, key=lambda k: np.linalg.norm(lab[i] - centroids[k]))
        return out

    def _track_colours(self) -> tuple[list[int], np.ndarray]:
        ref = self._reference_illum
        ids, colours = [], []
        for tid, samples in self._samples.items():
            if len(samples) < self.min_samples:
                continue
            illums = self._illum.get(tid) or [None] * len(samples)
            if ref is not None:
                samples = [
                    correct_illumination(c, il, ref) if il is not None else c
                    for c, il in zip(samples, illums)
                ]
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
        self._reference_illum = self._fit_reference_illuminant()
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

        # Prefer the grass-relative split when the declared kits straddle the
        # turf. Clustering cannot find this boundary: the two kits abut rather
        # than separate — on the U14G clip the relative-lightness histogram is
        # unimodal with a long sunlit-white tail, so both k-means and Otsu cut at
        # +53 and isolate 12 bright shirts instead of the 73/115 the eye sees.
        # The boundary is at zero for a physical reason, not a statistical one:
        # a dark kit reflects less than the grass beside it, a light kit more.
        labels = None
        split = lightness_split_kits(kits)
        if split is not None:
            rel = self._track_relative_lightness()
            covered = [t for t in ids if t in rel]
            # Not every track ever stands on visible grass — one in a crowd of
            # players, or at the frame edge, can go a whole lane without a clean
            # turf ring. Requiring all of them would silently drop this path on
            # any real match, so require most and place the rest by colour.
            if len(covered) >= _MIN_ILLUM_COVERAGE * len(ids):
                dark_kit, light_kit = split
                labels = np.array([
                    (0 if rel[t] <= 0 else 1) if t in rel else -1 for t in ids
                ])
                labels = self._place_uncovered(labels, colours)
                self._split_method = (
                    f"turf-relative lightness ({len(covered)}/{len(ids)} tracks;"
                    f" rest by colour)"
                )
                self._forced_names = {"team_a": dark_kit, "team_b": light_kit}

        if labels is None or len(np.unique(labels)) < 2:
            lab = cv2.cvtColor(colours.reshape(-1, 1, 3).astype(np.uint8),
                               cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, _ = cv2.kmeans(lab, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            labels = labels.ravel()
            self._split_method = "colour clustering"
            self._forced_names = {}

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
        if self._forced_names:
            # The grass-relative split already knows which side is which kit;
            # re-deriving the names from cluster colour would only reintroduce
            # the shadow error it was chosen to avoid.
            for key in ordered:
                self._team_names[key] = self._forced_names[key]
            return self
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

    def split_method(self) -> str:
        """How the two teams were separated — for the run log.

        Either ``"turf-relative lightness"`` (the declared kits straddle the
        grass, so the sign of torso-minus-turf lightness decides) or
        ``"colour clustering"`` (the general fallback).
        """
        return self._split_method

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
