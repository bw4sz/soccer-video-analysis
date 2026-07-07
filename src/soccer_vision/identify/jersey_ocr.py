"""Dedicated jersey-number recognizer + the per-track crop/vote driver.

Reading a jersey number is a two-stage job (the SoccerNet recipe):

1. **Legibility** — most player crops from an elevated youth camera show a back
   at an angle with no readable number. A cheap gate drops crops too small or
   low-contrast to bother the recognizer with; a learned legibility classifier
   can replace it later.
2. **Recognition** — a scene-text model reads the digits off the surviving
   crops. The default backend is PARSeq (``baudm/parseq``), which the SoccerNet
   jersey-number challenge winners fine-tune; a jersey-fine-tuned checkpoint
   drops in via ``model_id`` without touching callers.

:func:`assign_jerseys` drives the recognizer across one track's sampled boxes
and hands the reads to :func:`soccer_vision.identify.vote.vote_jersey`. The pure
voting math lives in :mod:`.vote`; this module owns everything that touches a
model or pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_vision.identify.vote import JerseyVote, vote_jersey

# Default recognizer. A jersey-fine-tuned PARSeq checkpoint should replace this
# once weights are pinned (see plan open items); the generic scene-text model is
# a usable first cut and keeps the dependency to a torch.hub entrypoint.
DEFAULT_PARSEQ_HUB = ("baudm/parseq", "parseq")


@dataclass
class JerseyRead:
    """One crop's recognition result. ``number`` is ``None`` if illegible/unparseable."""

    number: int | None
    confidence: float


def crop_number_region(frame: np.ndarray, bbox) -> np.ndarray | None:
    """Crop the upper-torso region of a player box where the number sits.

    Numbers ride high on the back/front, so we take the top ~55% of the box and
    trim the outer 10% each side to shed arms/background. Returns ``None`` for
    boxes too small to hold a legible number.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    if w < 12 or h < 24:
        return None
    cx1 = int(round(x1 + 0.10 * w))
    cx2 = int(round(x2 - 0.10 * w))
    cy1 = int(round(y1))
    cy2 = int(round(y1 + 0.55 * h))
    H, W = frame.shape[:2]
    cx1, cx2 = max(0, cx1), min(W, cx2)
    cy1, cy2 = max(0, cy1), min(H, cy2)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    return frame[cy1:cy2, cx1:cx2]


def is_legible(crop: np.ndarray, *, min_side: int = 16, min_std: float = 18.0) -> bool:
    """Cheap gate: reject crops too small or too flat to carry a number.

    ``min_std`` is the grayscale standard deviation — a uniform patch (bare
    jersey, turf) has almost none. A learned legibility classifier can supersede
    this without changing the interface.
    """
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if h < min_side or w < min_side:
        return False
    gray = crop.mean(axis=2) if crop.ndim == 3 else crop
    return float(gray.std()) >= min_std


def parse_number(text: str, char_confs: list[float] | None = None) -> JerseyRead:
    """Parse a recognizer's raw string to a jersey number in ``[0, 99]``.

    Keeps digit characters only; a 1–2 digit run becomes the number and its
    confidence is the min over those digits' confidences (a number is only as
    trustworthy as its least certain digit). Anything else is illegible.
    """
    digits = [(c, char_confs[i] if char_confs else 1.0)
              for i, c in enumerate(text) if c.isdigit()]
    if not digits or len(digits) > 2:
        return JerseyRead(None, 0.0)
    number = int("".join(c for c, _ in digits))
    conf = min(cf for _, cf in digits)
    return JerseyRead(number, float(conf))


class JerseyNumberRecognizer:
    """Wraps a scene-text model, exposing ``predict(crop) -> JerseyRead``.

    Mirrors the load pattern of
    :class:`soccer_vision.detection.rfdetr.RFDETRSoccerDetector`: construct via
    :meth:`from_pretrained`, which lazily pulls model weights so importing this
    module stays cheap and dependency-free.
    """

    def __init__(self, model, device: str = "cpu", *, img_size=(32, 128)):
        self.model = model
        self.device = device
        self.img_size = img_size

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        device: str | None = None,
    ) -> JerseyNumberRecognizer:
        """Load the recognizer. ``model_id`` overrides the default checkpoint.

        Raises ``ImportError`` with install guidance when the ``identify`` extra
        isn't present.
        """
        try:
            import torch
        except ImportError as e:  # pragma: no cover - env guard
            raise ImportError(
                "jersey recognition needs the 'identify' extra: "
                "pip install 'soccer-vision[identify]'"
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        repo, entry = DEFAULT_PARSEQ_HUB
        if model_id:
            repo = model_id
        model = torch.hub.load(repo, entry, pretrained=True, trust_repo=True)
        model = model.to(device).eval()
        return cls(model, device=device)

    def predict(self, crop: np.ndarray) -> JerseyRead:
        """Recognize the number on a single BGR crop."""
        import torch
        from PIL import Image

        rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
        img = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
        img = img.resize((self.img_size[1], self.img_size[0]))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # PARSeq normalises to [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = logits.softmax(-1)
            labels, confs = self.model.tokenizer.decode(probs)

        text = labels[0] if labels else ""
        char_confs = confs[0].tolist() if confs is not None and len(confs) else None
        return parse_number(text, char_confs)


def assign_jerseys(
    track_boxes: dict[int, list],
    recognizer: JerseyNumberRecognizer,
    reader,
    *,
    max_samples_per_track: int = 40,
    vote_kwargs: dict | None = None,
    progress: bool = False,
) -> dict[int, JerseyVote]:
    """Read + vote a jersey number for every track.

    ``track_boxes`` is ``{track_id: [(frame, bbox), ...]}`` from
    :func:`soccer_vision.clips.halo.load_track_boxes`. ``reader`` is a
    :class:`soccer_vision.io.video.VideoReader`. For each track we sample up to
    ``max_samples_per_track`` boxes evenly across its span, crop the number
    region, gate for legibility, recognize, then vote. Returns
    ``{track_id: JerseyVote}`` for every track (including unknowns).
    """
    vote_kwargs = vote_kwargs or {}
    out: dict[int, JerseyVote] = {}

    for tid, samples in track_boxes.items():
        if not samples:
            out[tid] = vote_jersey([], n_sampled=0, **vote_kwargs)
            continue
        step = max(1, len(samples) // max_samples_per_track)
        chosen = samples[::step]

        observations: list[tuple[int, float]] = []
        for frame_no, bbox in chosen:
            frame = reader.read_frame(int(frame_no))
            if frame is None:
                continue
            crop = crop_number_region(frame, bbox)
            if not is_legible(crop):
                continue
            read = recognizer.predict(crop)
            if read.number is not None:
                observations.append((read.number, read.confidence))

        out[tid] = vote_jersey(observations, n_sampled=len(chosen), **vote_kwargs)
        if progress:
            v = out[tid]
            tag = f"#{v.jersey}" if v.jersey is not None else "unknown"
            print(f"  track {tid}: {tag} "
                  f"(conf {v.confidence:.2f}, {v.n_obs}/{len(chosen)} legible)")

    return out
