"""Event-source abstraction.

An *event source* turns a processed match into a list of event dicts. Sources are
agnostic to the downstream player/team association and clip selection, so any new
detector — the T-DEED team tackle model (see
``training/sn_spotting/train_teamspotting.py``), a SAM3-prompt source, or a
Claude-vision source — plugs in behind the same interface without touching the
association or clip code.

Event dicts follow the shape produced by ``events/set_piece.py``:
``{label, frame, timestamp_s, position_ms, confidence, ...}``. ``team`` and
``track_id`` are added later by ``events/associate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from soccer_vision.events.set_piece import detect_all_set_pieces


@dataclass
class DetectionContext:
    """Everything an event source may need, assembled once by the pipeline."""

    fps: float
    ball_positions: list[dict] = field(default_factory=list)
    frame_players: dict[int, dict] = field(default_factory=dict)
    proxy_path: str | None = None
    config: dict = field(default_factory=dict)


@runtime_checkable
class EventSource(Protocol):
    """A pluggable detector that emits event dicts for a match."""

    name: str

    def is_available(self) -> bool:
        """Whether this source can run (weights present, deps installed, ...)."""
        ...

    def detect(self, ctx: DetectionContext) -> list[dict]:
        """Return event dicts for the match described by ``ctx``."""
        ...


class SetPieceSource:
    """Ball-position heuristics for goal kicks, corners, and throw-ins.

    Wraps the existing ``detect_all_set_pieces`` — always available (classical CV).
    """

    name = "set_piece"

    def is_available(self) -> bool:
        return True

    def detect(self, ctx: DetectionContext) -> list[dict]:
        kwargs = ctx.config.get("set_piece", {})
        events = detect_all_set_pieces(ctx.ball_positions, **kwargs)
        for e in events:
            e.setdefault("source", self.name)
        return events


class TackleSource:
    """Team ball-action tackle detection (T-DEED / sn-teamspotting).

    Interface only: no trained checkpoint ships yet, so ``is_available`` is False
    and ``detect`` returns nothing. Train/export a model with
    ``training/sn_spotting/train_teamspotting.py`` (label ``PLAYER SUCCESSFUL
    TACKLE`` → ``tackle``, already team-aware) and load it here to activate the
    ``tackle`` label end-to-end — association and clip code need no changes.
    """

    name = "tackle"

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint

    def is_available(self) -> bool:
        return False

    def detect(self, ctx: DetectionContext) -> list[dict]:
        return []


class SoccerChatSource:
    """SoccerChat VLM as a sliding-window event spotter (opt-in, experimental).

    The primary SoccerChat integration is the clip verifier/captioner in
    :mod:`soccer_vision.verify.soccerchat` (see ``soccer-vision describe``). This
    source is the heavier alternative: it slides a 10-second window across the
    broadcast proxy, classifies each window into the 16 SoccerNet classes, and
    emits events for the non-empty ones. It is **off by default** (listed in
    :data:`_OPT_IN`) and only activates when named explicitly in
    ``config['sources']`` *and* the ms-swift runtime is installed — so it never
    fires on a plain ``soccer-vision process`` run.

    Trained on professional broadcast footage, so expect weak recall on youth /
    Veo video until fine-tuned; prefer the verifier path for now.
    """

    name = "soccerchat"

    def __init__(self, window_s: float = 10.0, stride_s: float = 10.0):
        self.window_s = window_s
        self.stride_s = stride_s

    def is_available(self) -> bool:
        from soccer_vision.verify.soccerchat import is_available

        return is_available()

    def detect(self, ctx: DetectionContext) -> list[dict]:
        if not ctx.proxy_path:
            return []
        import tempfile

        from soccer_vision.io.video import VideoReader, ffmpeg_extract_clip
        from soccer_vision.verify.soccerchat import SoccerChatModel

        cfg = ctx.config.get("soccerchat", {})
        window_s = cfg.get("window_s", self.window_s)
        stride_s = cfg.get("stride_s", self.stride_s)

        reader = VideoReader(ctx.proxy_path)
        duration_s = reader.duration_s
        reader.close()

        model = SoccerChatModel()
        events: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            t = 0.0
            while t < duration_s:
                clip = f"{tmp}/win_{int(t)}.mp4"
                ffmpeg_extract_clip(ctx.proxy_path, t, window_s, clip, reencode=True)
                cls = model.classify(clip)
                if cls["label"] and cls["confidence"] >= cfg.get("min_confidence", 0.6):
                    mid = t + window_s / 2
                    events.append({
                        "label": cls["label"],
                        "timestamp_s": round(mid, 2),
                        "position_ms": int(mid * 1000),
                        "frame": int(mid * ctx.fps),
                        "confidence": cls["confidence"],
                        "source": self.name,
                        "sc_class": cls["sc_class"],
                    })
                t += stride_s
        return events


# Registry of known sources. Extend as detectors land.
_REGISTRY: dict[str, type] = {
    "set_piece": SetPieceSource,
    "tackle": TackleSource,
    "soccerchat": SoccerChatSource,
}

# Sources that must be named explicitly in ``config['sources']`` — never part of
# the default active set even when their runtime is available.
_OPT_IN: frozenset[str] = frozenset({"soccerchat"})


def active_sources(config: dict | None = None) -> list[EventSource]:
    """Instantiate the available event sources selected by ``config``.

    ``config['sources']`` may name a subset (opt-in sources like ``soccerchat``
    only run when named here); the default set is every registered source except
    the opt-in ones. Sources whose ``is_available()`` is False are dropped.
    """
    config = config or {}
    names = config.get("sources") or [n for n in _REGISTRY if n not in _OPT_IN]
    sources: list[EventSource] = []
    for name in names:
        cls = _REGISTRY.get(name)
        if cls is None:
            continue
        src = cls()
        if src.is_available():
            sources.append(src)
    return sources


def run_sources(sources: list[EventSource], ctx: DetectionContext) -> list[dict]:
    """Run every source and return merged, time-sorted events."""
    events: list[dict] = []
    for src in sources:
        events.extend(src.detect(ctx))
    events.sort(key=lambda e: e.get("timestamp_s", 0.0))
    return events
