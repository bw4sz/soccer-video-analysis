"""Let the jersey reader veto a re-id name it can prove wrong.

The two identity pathways fail in opposite ways, which is what makes combining
them worth doing:

- **Re-id** names nearly every track, and on teammates in an identical kit it is
  right about 42% of the time (see *Enroll* in ``CLAUDE.md``). When it is wrong
  it is confidently wrong, and the mistake reaches the reel as another player's
  clip — the failure this module exists to catch.
- **OCR** abstains on most crops (a number is legible on roughly a third of them
  on Veo footage), but several *high-confidence* reads agreeing on one number
  across a lane is about the strongest evidence available that the shirt in that
  lane carries that number.

So they are combined **asymmetrically**: re-id keeps naming, and OCR is only ever
allowed to *veto*. It never renames a track, because on the reads that matter it
mostly abstains — promoting it to arbiter would trade a 42%-correct namer for one
that says nothing at all. A veto drops the track back to unknown rather than
relabelling it: the evidence says "this lane is not Morgan", which is not the
same as saying who it is (the reader may be seeing an opponent, or a number that
belongs to nobody on the roster).

The veto bar is deliberately far above the bar for *naming* a track from OCR
(``vote_jersey``'s 3 reads / 0.5 share / 0.15 margin). Reads are first floored at
``min_read_conf`` — PARSeq's low-confidence output on this footage hallucinates
digits, notably ``1`` — and the surviving reads must then be numerous and near
unanimous. A wrong veto costs one dropped clip; a missed veto puts the wrong
child in a parent's highlight reel.

Pure (no model, no video) so the decision can be unit-tested on synthetic reads.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from soccer_vision.identify.vote import vote_jersey

AGREE = "agree"
CONFLICT = "conflict"
NO_EVIDENCE = "no_evidence"


@dataclass(frozen=True)
class CrossCheck:
    """What the jersey reads say about a name that came from somewhere else.

    ``verdict`` is one of ``"agree"`` (the reads back the expected number),
    ``"conflict"`` (they back a *different* number, strongly enough to act on) or
    ``"no_evidence"`` (nothing cleared the bar, or there was no number to check
    against). ``jersey`` is the number the surviving reads stand behind, with
    ``confidence`` its share of their weight and ``n_obs`` how many there were.
    """

    verdict: str
    jersey: int | None
    confidence: float
    n_obs: int

    @property
    def conflicts(self) -> bool:
        return self.verdict == CONFLICT


def crosscheck_jersey(
    expected: int | None,
    reads: Sequence[tuple[int, float]],
    *,
    min_read_conf: float = 0.7,
    min_reads: int = 4,
    min_share: float = 0.75,
    min_margin: float = 0.5,
    exclude: Iterable[int] = (),
) -> CrossCheck:
    """Decide whether ``reads`` corroborate or contradict jersey ``expected``.

    ``reads`` are the ``(number, confidence)`` observations behind a track's
    :class:`~soccer_vision.identify.vote.JerseyVote`. Only reads at or above
    ``min_read_conf`` are counted, and numbers in ``exclude`` are discarded
    outright (hallucination classes). The survivors are re-voted under the
    stricter ``min_reads`` / ``min_share`` / ``min_margin`` guards; a number that
    clears them and differs from ``expected`` is a conflict.

    ``expected`` of ``None`` — a re-id name with no roster number behind it —
    always yields ``no_evidence``: there is nothing for the reads to contradict.
    """
    excluded = {int(n) for n in exclude}
    strong = [
        (int(n), float(c))
        for n, c in reads
        if n is not None and float(c) >= min_read_conf and int(n) not in excluded
    ]

    vote = vote_jersey(
        strong,
        n_sampled=len(strong),
        min_votes=min_reads,
        min_share=min_share,
        min_margin=min_margin,
    )
    if vote.jersey is None:
        return CrossCheck(NO_EVIDENCE, None, vote.confidence, len(strong))
    if expected is None:
        return CrossCheck(NO_EVIDENCE, vote.jersey, vote.confidence, len(strong))

    verdict = AGREE if vote.jersey == int(expected) else CONFLICT
    return CrossCheck(verdict, vote.jersey, vote.confidence, len(strong))
