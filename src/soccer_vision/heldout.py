"""Footage that nothing may learn from, declared before the annotation exists.

The accuracy figures this project has collected so far — 42% teammate
separation, 5/6 precision at margin 0.05, the gallery A/B tables in
``CLAUDE.md`` — were all measured on crops drawn from the same match, and often
the same minute, as the exemplars they were scored against. Re-id is
nearest-neighbour, so a crop enrolled 40 s from the query is very nearly a
duplicate of it; a number measured that way is a memory test wearing a
generalization test's clothes, and there is no way to tell the two apart after
the fact.

This module is the fix, and its whole point is that it runs *before* the
annotation: ``heldout.yaml`` declares spans of video that no gallery, training
set or threshold sweep may draw from, ``enroll`` refuses to bank a crop inside
one, and :mod:`scripts.audit_heldout` checks a finished gallery against the
registry. A convention in a commit message cannot do this; a file the code reads
can.

**Matching a span to the thing being filtered.** A span is declared against a
source video, but the crops being filtered come from a *run*, whose footage is
always called ``broadcast_proxy.mp4``. So a block lists both the source video
basename and the run directory names that derive from it, and either identifies
it. An unrecognised run matches nothing and is therefore unprotected — which is
the dangerous direction, so :func:`Registry.spans_for` says so out loud rather
than returning an empty list quietly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Where to look for the registry when no path is given, in order: an explicit
#: environment override, then the repo root walking up from the working
#: directory, then the repo root inferred from this file's location.
ENV_VAR = "SOCCER_VISION_HELDOUT"
REGISTRY_NAME = "heldout.yaml"


class HeldoutViolation(RuntimeError):
    """Raised when something tries to learn from protected footage."""


@dataclass(frozen=True)
class Span:
    """One protected frame range on one video.

    Frames, not seconds, because that is what every caller actually holds — a
    box, a crop, a track sample. Seconds are the human unit and live in the YAML.
    """

    block_id: str
    flavour: str
    video: str
    first_frame: int
    last_frame: int
    start_s: float
    end_s: float

    def contains(self, frame: int) -> bool:
        return self.first_frame <= int(frame) <= self.last_frame

    def __str__(self) -> str:
        return (f"{self.block_id} [{_clock(self.start_s)}-{_clock(self.end_s)}] "
                f"frames {self.first_frame}-{self.last_frame}")


@dataclass(frozen=True)
class Gold:
    """An annotated subset of a block — the evaluation set itself."""

    gold_id: str
    block_id: str
    start_s: float
    end_s: float
    project: str
    scope: str


@dataclass
class Block:
    id: str
    flavour: str
    video: str
    runs: list[str]
    fps: float
    start_s: float
    end_s: float
    why: str = ""
    gold: list[Gold] = field(default_factory=list)

    def matches(self, key: str) -> bool:
        """Whether ``key`` names this block's video or one of its runs.

        Both a bare directory name and a path are accepted, so callers can pass
        whatever they happen to hold without normalising first.
        """
        name = Path(str(key)).name
        stem = Path(str(key)).stem
        if name.lower() == self.video.lower():
            return True
        return any(name == r or stem == r for r in self.runs)

    def span(self) -> Span:
        return Span(
            block_id=self.id,
            flavour=self.flavour,
            video=self.video,
            first_frame=int(round(self.start_s * self.fps)),
            last_frame=int(round(self.end_s * self.fps)),
            start_s=self.start_s,
            end_s=self.end_s,
        )


class Registry:
    """The parsed ``heldout.yaml``, and the questions worth asking of it."""

    def __init__(self, blocks: list[Block], path: Path | None = None):
        self.blocks = blocks
        self.path = path

    # -- lookup ----------------------------------------------------------

    def block(self, block_id: str) -> Block | None:
        return next((b for b in self.blocks if b.id == block_id), None)

    def gold(self, gold_id: str) -> tuple[Block, Gold] | None:
        for b in self.blocks:
            for g in b.gold:
                if g.gold_id == gold_id:
                    return b, g
        return None

    def spans_for(self, key: str | Path) -> list[Span]:
        """Protected spans on the video or run named by ``key``.

        Empty means "nothing declared for this footage", which is not the same as
        "this footage is safe" — it usually means the run is missing from the
        registry's ``runs:`` list. Callers that are about to enrol should say so;
        :func:`describe_coverage` writes that sentence for them.
        """
        return [b.span() for b in self.blocks if b.matches(key)]

    def describe_coverage(self, key: str | Path) -> str:
        spans = self.spans_for(key)
        if spans:
            return (f"held out: {', '.join(str(s) for s in spans)}"
                    f"  (registry: {self.path})")
        return (f"held out: nothing declared for '{Path(str(key)).name}'. If this is "
                f"footage we evaluate on, add it to {self.path or REGISTRY_NAME} "
                f"BEFORE enrolling from it — an unregistered run is unprotected, "
                f"not safe.")

    # -- filtering -------------------------------------------------------

    def is_protected(self, key: str | Path, frame: int) -> Span | None:
        """The span covering ``frame``, or ``None``."""
        return next((s for s in self.spans_for(key) if s.contains(frame)), None)

    def partition(self, key: str | Path, frames) -> tuple[list[int], list[int]]:
        """Split frame numbers into ``(outside, inside)`` the protected spans."""
        spans = self.spans_for(key)
        outside, inside = [], []
        for f in frames:
            (inside if any(s.contains(f) for s in spans) else outside).append(f)
        return outside, inside


# -- modes ---------------------------------------------------------------

#: Drop anything inside a protected span. The default everywhere.
EXCLUDE = "exclude"
#: Keep *only* what is inside — for staging the gold annotation project itself.
#: Enrolment refuses this mode outright; it exists to build evaluation data.
ONLY = "only"
#: No filtering. Prints a warning, because this is how a leak gets in.
OFF = "off"
MODES = (EXCLUDE, ONLY, OFF)


def filter_frames(items, mode: str, *, key: str | Path, registry: Registry | None,
                  frame_of=lambda x: x[0]) -> tuple[list, list]:
    """Apply ``mode`` to an iterable of things carrying a frame number.

    Returns ``(kept, dropped)``. ``frame_of`` says where the frame number lives,
    so this works for ``(frame, bbox, name)`` triples, ``(frame, bbox)`` track
    samples and bare frame numbers alike.
    """
    items = list(items)
    if mode == OFF or registry is None:
        return items, []
    spans = registry.spans_for(key)
    if not spans:
        return items, []

    kept, dropped = [], []
    for item in items:
        hit = any(s.contains(frame_of(item)) for s in spans)
        (dropped if hit == (mode == EXCLUDE) else kept).append(item)
    return kept, dropped


# -- loading -------------------------------------------------------------


def find_registry(start: Path | None = None) -> Path | None:
    """Locate ``heldout.yaml``: env var, then upward from ``start``, then the repo."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()

    here = Path(start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / REGISTRY_NAME
        if candidate.is_file():
            return candidate

    # src/soccer_vision/heldout.py -> repo root
    fallback = Path(__file__).resolve().parents[2] / REGISTRY_NAME
    return fallback if fallback.is_file() else None


def load_registry(path: str | Path | None = None) -> Registry:
    """Read the registry, or an empty one if there is no file to read."""
    import yaml

    found = Path(path) if path else find_registry()
    if not found or not Path(found).is_file():
        return Registry([], path=None)

    doc = yaml.safe_load(Path(found).read_text()) or {}
    blocks = []
    for raw in doc.get("blocks") or []:
        blocks.append(Block(
            id=str(raw["id"]),
            flavour=str(raw.get("flavour", "")),
            video=str(raw.get("video", "")),
            runs=[str(r) for r in (raw.get("runs") or [])],
            fps=float(raw.get("fps") or 30.0),
            start_s=float(raw["start_s"]),
            end_s=float(raw["end_s"]),
            why=str(raw.get("why", "")).strip(),
            gold=[
                Gold(
                    gold_id=str(g["id"]),
                    block_id=str(raw["id"]),
                    start_s=float(g["start_s"]),
                    end_s=float(g["end_s"]),
                    project=str(g.get("project", "")),
                    scope=str(g.get("scope", "")).strip(),
                )
                for g in (raw.get("gold") or [])
            ],
        ))
    return Registry(blocks, path=Path(found))


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60:d}:{int(seconds) % 60:02d}"
