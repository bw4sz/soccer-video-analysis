"""One player, one lane at a time — naming as an assignment, not twelve contests.

:func:`~soccer_vision.identify.gallery.match_track` scores a lane against every
identity and keeps the winner if it leads the runner-up by a margin. Every lane
is decided *alone*, so nothing in the pipeline knows that there is one Morgan on
the pitch. Measured on ``runs/saints-u14g-full``, "Morgan Lobey" was 275 lanes and
on 18.4% of the frames she was named at, **two to four of them were alive at
once** — every one of those frames has at least one other child wearing her name,
and the halo strobes between them.

The independence also breaks the margin, which is the reason this module exists
rather than being a tidy-up. Four players whose exemplars are spread wide score
around 0.64 against *every* lane, so on any given lane they finish first, second,
third and fourth separated by hundredths: on lane 27840, Morgan 0.699, Eveleigh
0.692, Morrighan 0.689, Leire 0.683, Catherine 0.680. The winner's lead is 0.007
against a gate of 0.05, so the lane goes unnamed — and the same five bunch again
on the next lane, and the next. **A margin between two arbitrary identities
measures how crowded the gallery is, not how confident the match is.** Morgan
converts 0.7% of the lanes she ranks first on into names; Quinn, whose exemplars
are tight enough to keep her out of those contests, converts 50%.

Here the margin is measured against **feasible** alternatives instead. If three
of those four rivals are already committed to lanes that overlap this one in
time, they are not alternatives at all, and Morgan's real runner-up is whoever
remains. The margin stops asking "who else is in the gallery" and starts asking
"who else could this actually be, given everything else on screen".

**Greedy, deliberately.** Candidates are taken best-first, each accepted if the
lane is still free and the identity is not already committed to an overlapping
lane. This is not the optimal assignment, and an optimal solver is easy to reach
for — but these scores separate at the third decimal, so exact optimisation over
them buys precision the numbers do not carry, while costing the property that
makes a greedy pass auditable: every decision can be explained by the ones before
it, which is what :func:`explain` prints.

**Two things are exempt.** :data:`~.gallery.NEGATIVE_LABEL` is never exclusive —
a crowd is full of people who are all simultaneously not ours — though it still
competes for the margin, so a lane that looks more like a spectator than like any
player is still rejected. And lanes that never overlap are never in competition:
a player is one person at an instant, not across a match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gallery import NEGATIVE_LABEL


@dataclass(frozen=True)
class Lane:
    """A lane's identity in time — everything the constraint needs to know."""

    tid: str
    start_s: float
    end_s: float


@dataclass
class Assignment:
    """What one lane ended up with, and what it was decided against."""

    tid: str
    name: str | None = None
    score: float = 0.0
    #: Lead over the best identity still *available* to this lane, which is what
    #: the threshold is applied to. Compare ``open_margin`` to see what the
    #: constraint was worth on this lane.
    margin: float = 0.0
    #: Lead over the runner-up ignoring the constraint — the old per-lane margin.
    open_margin: float = 0.0
    rejected: bool = False
    #: Identities committed to an overlapping lane that would otherwise have set
    #: this lane's margin, with the lane holding each. This is the audit trail for
    #: a surprising name: it says which other claim made this one look confident.
    blocked_by: list[tuple[str, str]] = field(default_factory=list)


def _overlaps(a: Lane, b: Lane, tol: float) -> bool:
    """True when two lanes are alive at the same time (beyond ``tol`` seconds).

    ``tol`` forgives the frame or two of overhang left by a tracker handing a
    player from one id to the next. It should stay small: the *containment* case
    — a duplicate detection box living and dying entirely inside a long lane —
    is a real conflict and the one this pass is most useful against, since
    ``merge_duplicate_lanes`` only ever pairs a death with a birth and cannot see
    it.
    """
    return min(a.end_s, b.end_s) - max(a.start_s, b.start_s) > tol


def assign_identities(
    lanes: list[Lane],
    scores: np.ndarray,
    names: list[str],
    *,
    min_similarity: float = 0.5,
    min_margin: float = 0.05,
    overlap_tolerance_s: float = 0.0,
) -> dict[str, Assignment]:
    """Name lanes under the constraint that one identity holds one lane at a time.

    ``scores`` is ``(len(lanes), len(names))`` as returned by
    :func:`~.gallery.track_scores`, one row per lane in ``lanes`` order.

    Returns an :class:`Assignment` per lane id, including the lanes that came
    away with nothing, so a lane that went unnamed can always be explained.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (len(lanes), len(names)):
        raise ValueError(
            f"scores must be (n_lanes, n_names) = ({len(lanes)}, {len(names)}), "
            f"got {scores.shape}"
        )

    neg = names.index(NEGATIVE_LABEL) if NEGATIVE_LABEL in names else None
    out = {ln.tid: Assignment(tid=ln.tid) for ln in lanes}

    # Every (lane, identity) worth considering, best first. Below the similarity
    # floor nothing can be named, so those pairs cannot affect an outcome.
    cand = [
        (float(scores[i, p]), i, p)
        for i in range(len(lanes))
        for p in range(len(names))
        if scores[i, p] >= min_similarity
    ]
    cand.sort(key=lambda c: -c[0])

    taken: dict[int, list[Lane]] = {}   # identity -> lanes already committed to it
    done: set[int] = set()             # lane indices already assigned
    # Lanes whose best feasible candidate has been evaluated. A lane that fails
    # the margin is *not* finished — an identity it was torn between can still be
    # committed elsewhere, which resolves the ambiguity and lets a later, lower
    # candidate through. But its diagnostics should record that first, best
    # attempt rather than the weaker ones tried afterwards.
    looked: set[int] = set()

    for score, i, p in cand:
        if i in done:
            continue
        lane = lanes[i]

        # Which identities could this lane still take? An identity is unavailable
        # only if it is already committed to a lane overlapping this one.
        available = np.ones(len(names), dtype=bool)
        holder: dict[int, str] = {}
        for q, held in taken.items():
            if q == neg:
                continue
            clash = next((h for h in held if _overlaps(lane, h, overlap_tolerance_s)), None)
            if clash is not None:
                available[q] = False
                holder[q] = clash.tid

        if not available[p]:
            continue  # a stronger claim holds this identity over an overlap

        rivals = scores[i].copy()
        rivals[p] = -np.inf
        open_runner = float(rivals.max())
        rivals[~available] = -np.inf
        feasible_runner = float(rivals.max())

        # Worth reporting only where it changed the answer: a committed identity
        # scoring below the best still-available one would not have been the
        # runner-up anyway, so removing it moved nothing.
        blocked = [(names[q], t) for q, t in holder.items()
                   if scores[i, q] > feasible_runner]

        margin = score - max(feasible_runner, 0.0)
        a = out[lane.tid]
        if margin < min_margin:
            if i not in looked:
                looked.add(i)
                a.score, a.margin = score, margin
                a.open_margin = score - max(open_runner, 0.0)
                a.blocked_by = sorted(blocked, key=lambda b: b[0])
            continue

        looked.add(i)
        done.add(i)
        a.score = score
        a.margin = margin
        a.open_margin = score - max(open_runner, 0.0)
        a.blocked_by = sorted(blocked, key=lambda b: b[0])
        if p == neg:
            a.rejected = True       # not exclusive: many lanes are not ours
            continue
        a.name = names[p]
        taken.setdefault(p, []).append(lane)

    return out


def concurrency(assignments: dict[str, Assignment], lanes: list[Lane]) -> dict[str, int]:
    """Worst number of lanes carrying one name at the same instant, per name.

    The check that this pass is doing its job: any value above 1 is a name on two
    children at once, so it is a *lower bound* on error and should be 1 for every
    name once the constraint holds.
    """
    by_tid = {ln.tid: ln for ln in lanes}
    worst: dict[str, int] = {}
    named: dict[str, list[Lane]] = {}
    for a in assignments.values():
        if a.name:
            named.setdefault(a.name, []).append(by_tid[a.tid])
    for name, group in named.items():
        edges = sorted([(g.start_s, 1) for g in group] + [(g.end_s, -1) for g in group])
        live = peak = 0
        for _, d in edges:
            live += d
            peak = max(peak, live)
        worst[name] = peak
    return worst


def explain(assignments: dict[str, Assignment], tid: str) -> str:
    """One lane's outcome in a sentence, including what outranked it and where."""
    a = assignments.get(tid)
    if a is None:
        return f"lane {tid}: not scored (gated out before the gallery ran)"
    if a.rejected:
        return f"lane {tid}: rejected as '{NEGATIVE_LABEL}' at {a.score:.3f}"
    blocked = ("; outscored by " +
               ", ".join(f"{n} (held by lane {t})" for n, t in a.blocked_by)
               ) if a.blocked_by else ""
    if a.name is None:
        return (f"lane {tid}: unnamed — best feasible lead {a.margin:.3f} "
                f"(unconstrained {a.open_margin:.3f}){blocked}")
    return (f"lane {tid}: {a.name} at {a.score:.3f}, lead {a.margin:.3f} over the "
            f"best available rival (unconstrained {a.open_margin:.3f}){blocked}")
