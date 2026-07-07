"""Collapse noisy per-frame jersey reads into one number per track.

Per-frame OCR flickers (motion blur, players facing away, angled digits), but a
player's jersey number is constant across a track. So we read the number on
every legible frame, then take a **confidence-weighted majority vote** per track
id, returning an explicit ``None`` (unknown) when support is weak or the vote is
split — better to say "don't know" than to mislabel a clip.

This module is deliberately pure (no model, no video) so the voting logic can be
unit-tested on synthetic observation lists.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class JerseyVote:
    """Result of voting one track's per-frame reads.

    ``jersey`` is ``None`` when no number cleared the thresholds. ``confidence``
    is the winner's share of total vote weight (0..1). ``n_obs`` counts the
    legible reads that went into the vote; ``legible_frac`` is that over the
    number of frames sampled for the track.
    """

    jersey: int | None
    confidence: float
    n_obs: int
    legible_frac: float


def vote_jersey(
    observations: list[tuple[int, float]],
    *,
    n_sampled: int | None = None,
    min_votes: int = 3,
    min_share: float = 0.5,
    min_margin: float = 0.15,
) -> JerseyVote:
    """Confidence-weighted majority vote over ``(number, confidence)`` reads.

    A number wins only if it clears every guard, otherwise ``jersey`` is
    ``None``:

    - ``min_votes`` — at least this many legible reads (a couple of frames of
      the same false read shouldn't name a player).
    - ``min_share`` — the winner holds at least this fraction of total weight
      (a genuine majority, not a plurality of noise).
    - ``min_margin`` — the winner leads the runner-up by at least this much
      weight share (rejects 6-vs-8 style ties the recognizer can't separate).

    ``n_sampled`` is the number of frames sampled for the track (>= number of
    legible reads); when given it sets ``legible_frac``. ``confidence`` is the
    winner's weight share.
    """
    n_obs = len(observations)
    legible_frac = (n_obs / n_sampled) if n_sampled else 0.0

    if n_obs < min_votes:
        return JerseyVote(None, 0.0, n_obs, legible_frac)

    weights: dict[int, float] = defaultdict(float)
    for number, conf in observations:
        if number is None:
            continue
        weights[number] += max(0.0, float(conf))

    total = sum(weights.values())
    if total <= 0:
        return JerseyVote(None, 0.0, n_obs, legible_frac)

    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    best_num, best_w = ranked[0]
    runner_w = ranked[1][1] if len(ranked) > 1 else 0.0

    share = best_w / total
    margin = (best_w - runner_w) / total
    if share < min_share or margin < min_margin:
        return JerseyVote(None, share, n_obs, legible_frac)

    return JerseyVote(int(best_num), share, n_obs, legible_frac)
