"""Action-detector abstraction.

An *action detector* turns a processed match into a list of event dicts. Detectors
are agnostic to the downstream player/team attribution and clip selection, so any
new engine plugs in behind the same interface without touching the attribution or
clip code. Each detector is a swappable *engine* named by a plain adjective — never
a model name:

- ``rules``   — ball-position heuristics for set pieces (always available).
- ``learned`` — the trained action model over player tracklets (needs a checkpoint).
- ``vlm``     — a video language model as a sliding-window spotter (opt-in).

Event dicts follow the shape produced by ``events/set_piece.py``:
``{label, frame, timestamp_s, position_ms, confidence, ...}``. ``team`` and
``track_id`` are added later by ``events/associate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from soccer_vision.events.set_piece import detect_all_set_pieces


@dataclass
class ActionContext:
    """Everything an action detector may need, assembled once by the pipeline."""

    fps: float
    ball_positions: list[dict] = field(default_factory=list)
    frame_players: dict[int, dict] = field(default_factory=dict)
    proxy_path: str | None = None
    config: dict = field(default_factory=dict)


@runtime_checkable
class ActionDetector(Protocol):
    """A pluggable engine that emits action/event dicts for a match."""

    name: str

    def is_available(self) -> bool:
        """Whether this engine can run (weights present, deps installed, ...)."""
        ...

    def detect(self, ctx: ActionContext) -> list[dict]:
        """Return event dicts for the match described by ``ctx``."""
        ...


class RulesActionDetector:
    """``rules`` engine — ball-position heuristics for goal kicks, corners, throw-ins.

    Wraps the existing ``detect_all_set_pieces`` — always available (classical CV),
    the default engine on a plain ``soccer-vision process`` run.
    """

    name = "rules"

    def is_available(self) -> bool:
        return True

    def detect(self, ctx: ActionContext) -> list[dict]:
        # Accept the new ``rules`` config block, falling back to the old
        # ``set_piece`` key for existing configs.
        kwargs = ctx.config.get("rules") or ctx.config.get("set_piece") or {}
        events = detect_all_set_pieces(ctx.ball_positions, **kwargs)
        for e in events:
            e.setdefault("source", self.name)
        return events


class LearnedActionDetector:
    """``learned`` engine — the trained action model over player tracklets.

    Interface only: no trained checkpoint ships yet, so ``is_available`` is False
    and ``detect`` returns nothing. This is the home for the learned action-spotting
    inference path — load a checkpoint, run our extracted tracklets (bbox + team +
    jersey) through it, and emit the predicted action per player. Attribution and
    clip code need no changes: team/jersey come from tracking, this engine predicts
    only the action class.
    """

    name = "learned"

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint

    def is_available(self) -> bool:
        return self.checkpoint is not None

    def detect(self, ctx: ActionContext) -> list[dict]:
        return []


class VLMActionDetector:
    """``vlm`` engine — a video language model as a sliding-window spotter (opt-in).

    The primary video-language-model integration is the clip verifier/captioner in
    :mod:`soccer_vision.verify.soccerchat` (see ``soccer-vision describe``). This
    engine is the heavier alternative: it slides a 10-second window across the
    broadcast proxy, classifies each window into the SoccerNet classes, and emits
    events for the non-empty ones. It is **off by default** (listed in
    :data:`_OPT_IN`) and only activates when named explicitly (``vlm`` in
    ``config['action_engines']``) *and* its runtime (transformers + peft) is
    installed — so it never fires on a plain ``soccer-vision process`` run.

    Trained on professional broadcast footage, so expect weak recall on youth /
    Veo video until fine-tuned; prefer the verifier path for now.
    """

    name = "vlm"

    def __init__(self, window_s: float = 10.0, stride_s: float = 10.0):
        self.window_s = window_s
        self.stride_s = stride_s

    def is_available(self) -> bool:
        from soccer_vision.verify.soccerchat import is_available

        return is_available()

    def detect(self, ctx: ActionContext) -> list[dict]:
        if not ctx.proxy_path:
            return []
        import tempfile

        from soccer_vision.io.video import VideoReader, ffmpeg_extract_clip
        from soccer_vision.verify.soccerchat import SoccerChatModel

        cfg = ctx.config.get("vlm") or ctx.config.get("soccerchat") or {}
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


# Registry of known engines, keyed by their plain-adjective name. Extend as
# engines land.
_REGISTRY: dict[str, type] = {
    "rules": RulesActionDetector,
    "learned": LearnedActionDetector,
    "vlm": VLMActionDetector,
}

# Back-compat: the old model-named registry keys still resolve to the new engines,
# so pre-rename configs keep working.
_LEGACY_KEYS: dict[str, str] = {
    "set_piece": "rules",
    "tackle": "learned",
    "soccerchat": "vlm",
}

# Engines that must be named explicitly — never part of the default active set even
# when their runtime is available.
_OPT_IN: frozenset[str] = frozenset({"vlm"})


def active_detectors(config: dict | None = None) -> list[ActionDetector]:
    """Instantiate the available action-detection engines selected by ``config``.

    ``config['action_engines']`` (or the legacy ``config['sources']``) may name a
    subset; opt-in engines like ``vlm`` only run when named here. The default set is
    every registered engine except the opt-in ones. Engines whose ``is_available()``
    is False are dropped.
    """
    config = config or {}
    names = (
        config.get("action_engines")
        or config.get("sources")
        or [n for n in _REGISTRY if n not in _OPT_IN]
    )
    detectors: list[ActionDetector] = []
    for name in names:
        name = _LEGACY_KEYS.get(name, name)
        cls = _REGISTRY.get(name)
        if cls is None:
            continue
        det = cls()
        if det.is_available():
            detectors.append(det)
    return detectors


def run_detectors(detectors: list[ActionDetector], ctx: ActionContext) -> list[dict]:
    """Run every detector and return merged, time-sorted events."""
    events: list[dict] = []
    for det in detectors:
        events.extend(det.detect(ctx))
    events.sort(key=lambda e: e.get("timestamp_s", 0.0))
    return events


# --- Back-compat aliases (pre-rename import paths) --------------------------------
DetectionContext = ActionContext
EventSource = ActionDetector
SetPieceSource = RulesActionDetector
TackleSource = LearnedActionDetector
SoccerChatSource = VLMActionDetector
active_sources = active_detectors
run_sources = run_detectors
