"""Match a player's *appearance* to their name, without reading the jersey.

For a team we see week after week, identity is better carried as an **appearance
gallery** than re-derived from OCR every match: enrol a handful of labelled crops
per player once, and from then on a track is named by nearest-neighbour lookup in
re-ID embedding space (:mod:`.reid` produces the embeddings). That works on the
backs, blurs and angles where jersey OCR abstains — on Veo footage OCR reads a
number on only a third of crops, and its confident-but-wrong reads cluster on a
few digits.

The gallery is plain arrays: ``emb`` holds every enrolled exemplar (L2-normalised
row per crop) and ``label`` says which player each row belongs to. Keeping the
exemplars rather than one centroid per player matters because a player looks
genuinely different from the front and the back, and across a kit change — one
mean vector would sit between those modes and match neither.

Matching mirrors :mod:`.vote`'s posture: a track is named only if the best player
clears an absolute similarity floor *and* leads the runner-up by a margin,
otherwise the result is ``None`` (unknown) and the caller can fall back to OCR.
Deliberately pure (numpy only, no torch, no video) so the matching maths is
unit-testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Reserved gallery label for exemplars that are *not* one of our players —
#: spectators, the neighbouring pitch, officials. A gallery holding only our
#: squad has no way to answer "none of the above": handed a stranger,
#: :func:`match_track` returns whichever of ours is nearest and the margin test
#: sees an ordinary win. Enrolling negatives under this label turns that
#: rejection into an ordinary nearest-neighbour outcome, and ``match_track``
#: reports it as an abstention with ``rejected=True`` rather than as a name.
NEGATIVE_LABEL = "not ours"

#: ``excluded`` value written to jerseys.json for a lane the negative class
#: claimed. Unlike ``"kit"`` and ``"short"``, which gate *before* any model runs,
#: this one is a model output — the lane was embedded and scored, and lost to the
#: negatives. Kept distinct so "we never looked" and "we looked and said no" are
#: never confused when auditing why a lane went unnamed.
EXCLUDED_NEGATIVE = "not_ours"


@dataclass(frozen=True)
class GalleryMatch:
    """Result of matching one track against the gallery.

    ``name`` is ``None`` when no player cleared the thresholds. ``similarity`` is
    the winner's mean cosine score (0..1 for same-hemisphere embeddings) and
    ``margin`` its lead over the runner-up; ``n_crops`` is how many crops of the
    track were scored.

    ``rejected`` distinguishes the two ways of arriving at ``name=None``: the
    gallery positively matched :data:`NEGATIVE_LABEL` (this is not one of our
    players), versus nothing cleared the thresholds (no claim either way). Only
    the first is evidence.
    """

    name: str | None
    similarity: float
    margin: float
    n_crops: int
    rejected: bool = False


#: Format of a ``source`` entry: ``"<run-or-video>@<frame>"``. Kept as one string
#: rather than parallel arrays so it survives ``merge_galleries`` and an ``.npz``
#: round trip without a schema, and stays readable when someone prints it.
def source_key(key: str, frame: int) -> str:
    """Provenance stamp for one exemplar — which footage, which frame."""
    from pathlib import Path as _Path

    return f"{_Path(str(key)).name}@{int(frame)}"


def parse_source(source: str) -> tuple[str, int] | None:
    """``"run@1234"`` → ``("run", 1234)``, or ``None`` if unstamped.

    A frame of ``-1`` means the footage is known but the frame isn't — the OCR
    bootstrap route embeds per track and never learns which frames survived. The
    held-out gate still ran, so such an exemplar is not a leak; it just can't be
    re-checked from the file alone, and the audit says so rather than passing it.
    """
    text = str(source or "")
    if "@" not in text:
        return None
    key, _, frame = text.rpartition("@")
    return (key, int(frame)) if frame.lstrip("-").isdigit() else None


def build_gallery(
    embeddings: np.ndarray,
    names: list[str],
    *,
    max_per_player: int = 64,
    rng: np.random.Generator | None = None,
    source: list[str] | None = None,
) -> dict:
    """Build a gallery from labelled crop embeddings.

    ``embeddings`` is ``(N, D)`` with ``names[i]`` naming row ``i``. Rows are
    L2-normalised so all downstream scoring is cosine similarity. Each player is
    capped at ``max_per_player`` exemplars (uniformly subsampled) to keep the
    gallery small and stop a player who happened to get a long track from
    dominating the nearest-neighbour search.

    ``source[i]`` records which footage and frame row ``i`` came from (see
    :func:`source_key`). It is optional only for backwards compatibility with
    galleries built before :mod:`soccer_vision.heldout` existed — without it a
    gallery cannot be audited against the held-out registry, and "we think it's
    clean" is exactly the claim that registry exists to stop anyone making.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(names) != len(embeddings):
        raise ValueError("embeddings must be (N, D) with one name per row")
    if source is not None and len(source) != len(embeddings):
        raise ValueError("source must have one entry per row when given")

    rng = rng or np.random.default_rng(0)
    roster = sorted(set(names))
    name_idx = {n: i for i, n in enumerate(roster)}
    names_arr = np.asarray(names)

    keep: list[np.ndarray] = []
    for name in roster:
        rows = np.flatnonzero(names_arr == name)
        if len(rows) > max_per_player:
            rows = np.sort(rng.choice(rows, max_per_player, replace=False))
        keep.append(rows)
    sel = np.concatenate(keep) if keep else np.zeros(0, dtype=int)

    emb = _l2_normalise(embeddings[sel])
    label = np.asarray([name_idx[n] for n in names_arr[sel]], dtype=np.int32)
    gallery = {"names": roster, "emb": emb, "label": label}
    if source is not None:
        gallery["source"] = [str(source[i]) for i in sel]
    return gallery


def merge_galleries(base: dict, extra: dict, *, max_per_player: int = 64) -> dict:
    """Union two galleries, re-capping per player.

    This is how a season's gallery grows: enrol from one match, then top up from
    the next without re-embedding anything already banked. Provenance is carried
    through when *both* sides have it; when either doesn't, the merged gallery
    drops it rather than stamping rows it can't vouch for — a half-stamped
    gallery would audit clean on the half that was recorded and say nothing about
    the rest, which is worse than admitting it is unauditable.
    """
    emb = np.concatenate([base["emb"], extra["emb"]])
    names = [base["names"][i] for i in base["label"]] + [
        extra["names"][i] for i in extra["label"]
    ]
    source = None
    if base.get("source") and extra.get("source"):
        source = list(base["source"]) + list(extra["source"])
    return build_gallery(emb, names, max_per_player=max_per_player, source=source)


def save_gallery(gallery: dict, path: str | Path) -> None:
    """Write a gallery to ``.npz`` (names, exemplar embeddings, labels, provenance)."""
    arrays = {
        "names": np.asarray(gallery["names"], dtype=object),
        "emb": gallery["emb"],
        "label": gallery["label"],
    }
    if gallery.get("source"):
        arrays["source"] = np.asarray(list(gallery["source"]), dtype=object)
    np.savez_compressed(path, **arrays)


def load_gallery(path: str | Path) -> dict:
    """Read a gallery written by :func:`save_gallery`.

    ``source`` comes back empty for galleries written before provenance was
    stamped; :mod:`scripts.audit_heldout` reports those as unauditable rather
    than as clean.
    """
    with np.load(path, allow_pickle=True) as z:
        gallery = {
            "names": [str(n) for n in z["names"]],
            "emb": np.asarray(z["emb"], dtype=np.float32),
            "label": np.asarray(z["label"], dtype=np.int32),
        }
        gallery["source"] = ([str(s) for s in z["source"]]
                             if "source" in z.files else [])
    return gallery


def track_scores(
    embeddings: np.ndarray,
    gallery: dict,
    *,
    top_k: int = 3,
) -> np.ndarray:
    """Score one track against every gallery identity — one number per identity.

    Each crop is scored against a player as the mean of its ``top_k`` best
    similarities to that player's exemplars, then averaged over the track. This
    is the whole of :func:`match_track`'s evidence; that function only adds the
    thresholds. It is split out because deciding *between* identities is not the
    only question worth asking of these numbers — :mod:`.assign` needs the full
    vector to resolve several lanes against each other at once, and throwing
    away everything but the winner and the runner-up is what makes a per-lane
    decision unable to see that two lanes are claiming the same child.

    An identity with no exemplars scores ``-1.0`` so it can never win.
    """
    embeddings = _l2_normalise(np.atleast_2d(np.asarray(embeddings, dtype=np.float32)))
    n_players = len(gallery["names"])
    sims = embeddings @ gallery["emb"].T  # (n_crops, n_exemplars)

    scores = np.empty(n_players, dtype=np.float32)
    for p in range(n_players):
        cols = sims[:, gallery["label"] == p]
        if cols.size == 0:
            scores[p] = -1.0
            continue
        k = min(top_k, cols.shape[1])
        # np.sort is ascending, so the k best per crop are the last k columns.
        scores[p] = float(np.sort(cols, axis=1)[:, -k:].mean())
    return scores


def match_track(
    embeddings: np.ndarray,
    gallery: dict,
    *,
    min_similarity: float = 0.5,
    min_margin: float = 0.05,
    top_k: int = 3,
) -> GalleryMatch:
    """Name the track whose crops embed to ``embeddings``, or abstain.

    Each crop is scored against each player as the mean of its ``top_k`` best
    similarities to that player's exemplars — a single lucky nearest neighbour
    shouldn't decide a name, but neither should averaging over exemplars from a
    pose the crop doesn't share. Crop scores are then averaged over the track,
    which is what makes this robust where per-frame OCR is not: one blurred crop
    barely moves a twenty-crop mean.

    Returns ``name=None`` unless the winner scores at least ``min_similarity``
    *and* leads the runner-up by ``min_margin`` — an unenrolled player (the other
    team, a referee) resembles everyone equally, and abstaining sends the track
    to the OCR fallback instead of mislabelling a clip.

    When :data:`NEGATIVE_LABEL` is enrolled and wins on those same terms, the
    result is ``name=None, rejected=True``: the gallery is claiming this is not
    one of our players, which is a stronger statement than failing a threshold.
    A negative win that *doesn't* clear the thresholds is an ordinary
    abstention — the class gets no special authority, only its own entry in the
    ranking.
    """
    embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
    n_crops = len(embeddings)
    n_players = len(gallery["names"])
    if n_crops == 0 or n_players == 0:
        return GalleryMatch(None, 0.0, 0.0, n_crops)

    scores = track_scores(embeddings, gallery, top_k=top_k)

    order = np.argsort(-scores)
    best, runner = scores[order[0]], (scores[order[1]] if n_players > 1 else -1.0)
    margin = float(best - max(runner, 0.0))

    if best < min_similarity or margin < min_margin:
        return GalleryMatch(None, float(best), margin, n_crops)
    winner = gallery["names"][order[0]]
    if winner == NEGATIVE_LABEL:
        return GalleryMatch(None, float(best), margin, n_crops, rejected=True)
    return GalleryMatch(winner, float(best), margin, n_crops)


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    """Rows to unit length so a dot product is a cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)
