"""SoccerChat local vision-language verifier for event clips.

`SoccerChat <https://arxiv.org/html/2505.16630v1>`_ (Gautam et al., 2025) is
``Qwen2-VL-7B-Instruct`` fine-tuned (LoRA) on ~49k QA pairs derived from
SoccerNet-v2 broadcast footage. This module runs it *locally* on the 10-second
clips the pipeline already extracts and returns, per clip, a free-text
description plus a predicted event class — mirroring the role
:mod:`soccer_vision.verify.claude` plays with the Claude API, but with a
soccer-specialised model and no network round-trip.

Model card: ``SimulaMet/SoccerChat-qwen2-vl-7b`` (Apache-2.0). The checkpoint is
a plain PEFT LoRA over ``Qwen2-VL-7B-Instruct``, so we load it with
transformers + peft + qwen-vl-utils — a stable inference path independent of the
upstream ms-swift training stack (https://github.com/simula/SoccerChat).

**Caveat:** SoccerChat is trained on professional broadcast video (tight tele
lens, on-screen graphics, commentary), *not* overhead Veo / youth footage. Treat
its output as a hypothesis to be confirmed in Label Studio — that correction loop
is also what produces youth fine-tuning data. See ``SOCCERCHAT_INTEGRATION.md``.

The heavy imports (torch / transformers) are deferred into the methods that need
them so importing this module — and running the offline tests — never requires a
GPU or the ``soccerchat`` optional dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from soccer_vision.events.labels import EVENT_LABELS

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_ID = "SimulaMet/SoccerChat-qwen2-vl-7b"

# The 16 SoccerNet-v2 action classes SoccerChat was trained to emit, verbatim
# from the paper's taxonomy. Note there is **no "goal kick" class** — goal kicks
# fall under "Kick-off" / "Ball out of play" in SoccerNet, which matters for
# youth footage where goal kicks are frequent (see ``verify_clip``).
SOCCERCHAT_CLASSES: list[str] = [
    "Ball out of play",
    "Clearance",
    "Corner",
    "Direct free-kick",
    "Foul",
    "Goal",
    "Indirect free-kick",
    "Kick-off",
    "Offside",
    "Penalty",
    "Red card",
    "Shots off target",
    "Shots on target",
    "Substitution",
    "Throw-in",
    "Yellow card",
]

# Exact-match map from SoccerChat's class strings to canonical soccer-vision
# labels. Defined explicitly (rather than reusing spotting.SOCCERNET_LABEL_MAP via
# its normaliser) because SoccerChat's casing/hyphenation — "Kick-off",
# "Throw-in", "Red card" — does not match that map's keys and would silently fall
# through to a wrong slug.
SOCCERCHAT_TO_LABEL: dict[str, str] = {
    "Ball out of play": "ball_out",
    "Clearance": "clearance",
    "Corner": "corner_kick",
    "Direct free-kick": "free_kick",
    "Foul": "foul",
    "Goal": "goal",
    "Indirect free-kick": "free_kick",
    "Kick-off": "kickoff",
    "Offside": "offside",
    "Penalty": "penalty",
    "Red card": "red_card",
    "Shots off target": "shot",
    "Shots on target": "shot",
    "Substitution": "substitution",
    "Throw-in": "throw_in",
    "Yellow card": "yellow_card",
}

# SoccerChat has no goal-kick class; these are the classes it is expected to emit
# for a genuine youth goal kick, so a heuristic ``goal_kick`` matched to one of
# these is scored PLAUSIBLE rather than REJECTED.
_GOAL_KICK_PROXY_CLASSES = frozenset({"Kick-off", "Ball out of play"})

_CLASSIFY_PROMPT = (
    "You are analysing a 10-second soccer clip. Classify the single most salient "
    "event into exactly one of these categories:\n{classes}\n"
    "Reply with only the category name, exactly as written above."
)
_DESCRIBE_PROMPT = (
    "Describe what happens in this 10-second soccer clip in one or two sentences: "
    "the phase of play, where the ball is, and any set piece or shot."
)


def is_available() -> bool:
    """Whether the SoccerChat runtime (transformers + peft + qwen-vl-utils) imports.

    Does not require a GPU to return True, but a GPU is strongly recommended for
    the 7B model. Returns False when the ``soccerchat`` optional dependencies are
    not installed.
    """
    try:
        import peft  # noqa: F401
        import qwen_vl_utils  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _normalise_class(text: str) -> str | None:
    """Map raw model text onto one of :data:`SOCCERCHAT_CLASSES`.

    Returns the canonical class string, or ``None`` if nothing matches.
    """
    t = text.strip().strip(".").lower()
    # Exact (case-insensitive) match first.
    for cls in SOCCERCHAT_CLASSES:
        if t == cls.lower():
            return cls
    # Substring / containment fallback (model sometimes adds a justification).
    for cls in SOCCERCHAT_CLASSES:
        if cls.lower() in t:
            return cls
    return None


class SoccerChatModel:
    """Lazy wrapper around Qwen2-VL-7B + the SoccerChat LoRA adapter.

    The SoccerChat checkpoint is a standard PEFT LoRA
    (``adapter_config.json`` → ``peft_type: LORA``) over ``Qwen2-VL-7B-Instruct``,
    so it loads with transformers + peft rather than ms-swift — a stable path
    independent of the training stack. The base model (~16 GB) and adapter load on
    first inference, not at construction, so this object stays cheap to create and
    easy to mock in tests.
    """

    def __init__(
        self,
        *,
        adapter: str = ADAPTER_ID,
        model: str = MODEL_ID,
        max_frames: int = 24,
        max_pixels: int = 100352,
        temperature: float = 0.0,
        max_tokens: int = 512,
        device_map: str = "auto",
    ):
        self.adapter = adapter
        self.model = model
        self.max_frames = max_frames
        self.max_pixels = max_pixels
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device_map = device_map
        self._model = None
        self._processor = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from huggingface_hub import snapshot_download
            from peft import PeftModel
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - requires optional dep
            raise RuntimeError(
                "SoccerChat needs the 'soccerchat' extra: pip install "
                f"'soccer-vision[soccerchat]'. Original error: {exc}"
            ) from exc

        base = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model, torch_dtype=torch.bfloat16, device_map=self.device_map,
        )
        # Resolve a HuggingFace adapter id to a local snapshot for peft.
        adapter_path = self.adapter
        if "/" in self.adapter and not Path(self.adapter).exists():
            adapter_path = snapshot_download(self.adapter)
        self._model = PeftModel.from_pretrained(base, adapter_path)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model, max_pixels=self.max_pixels)

    def infer(self, clip_path: str | Path, question: str) -> str:
        """Run one clip + question through the model and return the text answer."""
        import torch
        from qwen_vl_utils import process_vision_info

        self._ensure_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(clip_path),
                        "nframes": self.max_frames,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # qwen-vl-utils returns (images, videos) or (images, videos, video_kwargs).
        vision = process_vision_info(messages)
        image_inputs, video_inputs = vision[0], vision[1]
        video_kwargs = vision[2] if len(vision) > 2 else {}
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self._model.device)

        gen_kwargs = {"max_new_tokens": self.max_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **gen_kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def classify(self, clip_path: str | Path) -> dict[str, Any]:
        """Classify a clip into a SoccerChat class + canonical label.

        Returns ``{"sc_class", "label", "confidence", "raw"}``. ``confidence`` is
        a coarse proxy (the model exposes no calibrated score through this API):
        0.9 for an exact class echo, 0.6 for a contained match, 0.0 if unmatched.
        """
        prompt = _CLASSIFY_PROMPT.format(classes="\n".join(f"- {c}" for c in SOCCERCHAT_CLASSES))
        raw = self.infer(clip_path, prompt)
        sc_class = _normalise_class(raw)
        if sc_class is None:
            return {"sc_class": None, "label": None, "confidence": 0.0, "raw": raw}
        confidence = 0.9 if raw.strip().strip(".").lower() == sc_class.lower() else 0.6
        return {
            "sc_class": sc_class,
            "label": SOCCERCHAT_TO_LABEL.get(sc_class),
            "confidence": confidence,
            "raw": raw,
        }

    def describe(self, clip_path: str | Path) -> str:
        """Return a one/two-sentence natural-language description of the clip."""
        return self.infer(clip_path, _DESCRIBE_PROMPT)


def verify_clip(
    model: SoccerChatModel,
    clip_path: str | Path,
    candidate_label: str,
    *,
    describe: bool = True,
) -> dict[str, Any]:
    """Verify one candidate event against its clip with SoccerChat.

    ``verdict`` is one of:

    * ``CONFIRMED``  — SoccerChat's class maps to the same canonical label.
    * ``PLAUSIBLE``  — the candidate is a ``goal_kick`` and SoccerChat emitted a
      Kick-off / Ball-out proxy (it has no goal-kick class), so it neither
      confirms nor contradicts.
    * ``REJECTED``   — SoccerChat predicts a different, incompatible label.
    * ``UNKNOWN``    — SoccerChat produced nothing mappable.
    """
    cls = model.classify(clip_path)
    sc_label = cls["label"]
    sc_class = cls["sc_class"]

    if sc_label is None:
        verdict = "UNKNOWN"
        reason = f"SoccerChat gave no mappable class (said: {cls['raw']!r})."
    elif sc_label == candidate_label:
        verdict = "CONFIRMED"
        reason = f"SoccerChat agrees: {sc_class}."
    elif candidate_label == "goal_kick" and sc_class in _GOAL_KICK_PROXY_CLASSES:
        verdict = "PLAUSIBLE"
        reason = (
            f"SoccerChat said {sc_class}; it has no goal-kick class, so this is "
            "consistent with a goal kick but unconfirmed."
        )
    else:
        verdict = "REJECTED"
        reason = f"SoccerChat predicts {sc_class} ({sc_label}), not {candidate_label}."

    result = {
        "label": candidate_label,
        "sc_class": sc_class,
        "sc_label": sc_label,
        "confidence": cls["confidence"],
        "verdict": verdict,
        "reason": reason,
        "raw": cls["raw"],
    }
    if describe:
        result["caption"] = model.describe(clip_path)
    return result


def verify_events(
    event_clips: list[tuple[dict, str | Path]],
    *,
    model: SoccerChatModel | None = None,
    describe: bool = True,
    profile: dict | None = None,  # accepted for API parity with verify.claude
) -> dict[str, Any]:
    """Verify a batch of ``(event, clip_path)`` pairs.

    Returns a dict shaped for both the ``describe`` CLI and the Label Studio task
    builder::

        {
          "results":  [ {frame, timestamp_s, clip, ...verify_clip fields...}, ... ],
          "verified": [ {frame, label, reason}, ... ],   # CONFIRMED only
          "rejected": [ {frame, label, reason}, ... ],   # REJECTED only
        }

    ``model`` is injected for testing; when ``None`` a real
    :class:`SoccerChatModel` is constructed (and weights load on first clip).
    """
    if model is None:
        model = SoccerChatModel()

    results: list[dict] = []
    verified: list[dict] = []
    rejected: list[dict] = []

    for event, clip_path in event_clips:
        if clip_path is None:
            continue
        r = verify_clip(model, clip_path, event["label"], describe=describe)
        r["frame"] = event.get("frame")
        r["timestamp_s"] = event.get("timestamp_s")
        r["clip"] = str(clip_path)
        results.append(r)

        summary = {"frame": event.get("frame"), "label": event["label"], "reason": r["reason"]}
        if r["verdict"] == "CONFIRMED":
            verified.append(summary)
        elif r["verdict"] == "REJECTED":
            rejected.append(summary)

    return {"results": results, "verified": verified, "rejected": rejected}


# Guard: keep the two taxonomies from drifting out of the canonical label set.
_UNKNOWN = set(SOCCERCHAT_TO_LABEL.values()) - set(EVENT_LABELS)
assert not _UNKNOWN, f"SoccerChat map targets non-canonical labels: {_UNKNOWN}"
del _UNKNOWN
