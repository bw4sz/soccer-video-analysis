"""Appearance embeddings for player crops (the sportsreid re-ID backbone).

Turns a player crop into a 512-d vector where the same person lands close to
themselves across frames, poses and camera cuts. :mod:`.gallery` does the naming;
this module owns everything that touches a model or pixels.

Weights come from **sportsreid** (github.com/shallowlearn/sportsreid, MIT), which
fine-tunes Torchreid backbones on SoccerNet-ReID and placed 2nd in the SoccerNet
2022 re-identification challenge. We use its ``OSNet_x1_0`` entry (83.4 mAP /
78.0 rank-1) — 2.2M parameters, the best accuracy-per-parameter in its zoo and
small enough to embed a whole match's tracks on CPU. The published checkpoint is
~1 GB because it carries a 161k-identity classifier head and optimiser state; we
strip it to the backbone (~9 MB) on first load and cache that, since the
classifier is exactly the part we throw away — SoccerNet identities are not our
players. Our team-specific identities live in the gallery, not in the weights.

The architecture is vendored in :mod:`._osnet` (see that module for why).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# sportsreid model-zoo entry: OSNet_x1_0, 83.4 mAP / 78.0 rank-1 on SoccerNet-ReID.
SPORTSREID_OSNET_GDRIVE_ID = "1To0Ww6_HxU2ITAlb4kQEgYExV-orwit8"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "soccer_vision" / "reid"
SLIM_CHECKPOINT = "sportsreid_osnet_x1_0.backbone.pt"

# The zoo checkpoint is osnet_x1_0 with an extra 512-d fc head; num_classes only
# sizes the classifier we discard, so any value that loads is fine.
_FC_DIMS = [512]
_NUM_CLASSES = 161443


def crop_player(
    frame: np.ndarray, bbox, *, pad: float = 0.05, max_aspect: float = 1.5
) -> np.ndarray | None:
    """Crop a whole player box (BGR), padded slightly, or ``None`` if unusable.

    Unlike :func:`~.jersey_ocr.crop_number_region` this keeps the *full* body:
    re-ID reads kit, build, hair and gait cues spread over the whole figure, not
    just the number panel.

    Boxes wider than ``max_aspect`` times their height are rejected: a standing
    player is taller than wide, so these are the tracker's occasional blowouts
    (on the Saints U11 match, ~3% of boxes, some spanning the full 1920px frame).
    One of those in a gallery is a patch of turf and crowd enrolled as a person.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16 or w > max_aspect * h:
        return None
    H, W = frame.shape[:2]
    cx1 = max(0, int(round(x1 - pad * w)))
    cy1 = max(0, int(round(y1 - pad * h)))
    cx2 = min(W, int(round(x2 + pad * w)))
    cy2 = min(H, int(round(y2 + pad * h)))
    if cx2 - cx1 < 8 or cy2 - cy1 < 16:
        return None
    return frame[cy1:cy2, cx1:cx2]


class ReIDEmbedder:
    """Wraps the re-ID backbone, exposing ``embed(crops) -> (N, 512)``.

    Mirrors the load pattern of
    :class:`~.jersey_ocr.JerseyNumberRecognizer`: construct via
    :meth:`from_pretrained`, which lazily fetches weights so importing this
    module stays cheap.
    """

    def __init__(self, model, device: str = "cpu", *, img_size=(256, 128)):
        self.model = model
        self.device = device
        self.img_size = img_size

    @classmethod
    def from_pretrained(
        cls,
        weights: str | Path | None = None,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> ReIDEmbedder:
        """Load the backbone, downloading the sportsreid checkpoint if needed.

        ``weights`` overrides the default checkpoint (e.g. one fine-tuned on your
        own annotated crops). Raises ``ImportError`` with install guidance when
        the ``identify`` extra isn't present.
        """
        try:
            import torch
        except ImportError as e:  # pragma: no cover - env guard
            raise ImportError(
                "player re-id needs the 'identify' extra: "
                "pip install 'soccer-vision[identify]'"
            ) from e

        from soccer_vision.identify._osnet import osnet_x1_0

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        path = Path(weights) if weights else _ensure_backbone(Path(cache_dir or DEFAULT_CACHE_DIR))
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        state = {k.removeprefix("module."): v for k, v in state.items()}

        model = osnet_x1_0(num_classes=_NUM_CLASSES, fc_dims=_FC_DIMS)
        missing, _ = model.load_state_dict(state, strict=False)
        backbone_missing = [k for k in missing if not k.startswith("classifier.")]
        if backbone_missing:
            raise ValueError(
                f"{path} is not a sportsreid OSNet_x1_0 checkpoint "
                f"({len(backbone_missing)} backbone weights missing, "
                f"e.g. {backbone_missing[:3]})"
            )
        # eval() also switches OSNet.forward to return features, not logits.
        return cls(model.to(device).eval(), device=device)

    def embed(self, crops: list[np.ndarray], *, batch_size: int = 64) -> np.ndarray:
        """Embed BGR crops to an ``(N, 512)`` float32 array of unit vectors."""
        import torch

        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        out = []
        for i in range(0, len(crops), batch_size):
            batch = np.stack([self._preprocess(c) for c in crops[i:i + batch_size]])
            tensor = torch.from_numpy(batch).to(self.device)
            with torch.no_grad():
                out.append(self.model(tensor).cpu().numpy())
        feats = np.concatenate(out).astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-12)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """BGR crop → CHW float32, ImageNet-normalised at the model's input size."""
        import cv2

        h, w = self.img_size
        img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return ((rgb - mean) / std).transpose(2, 0, 1)


def embed_tracks(
    track_boxes: dict[int, list],
    embedder: ReIDEmbedder,
    reader,
    *,
    max_samples_per_track: int = 20,
    track_ids: set[int] | None = None,
    progress: bool = False,
) -> dict[int, np.ndarray]:
    """Embed up to ``max_samples_per_track`` crops for each track.

    ``track_boxes`` is ``{track_id: [(frame, bbox), ...]}`` from
    :func:`soccer_vision.clips.halo.load_track_boxes`; ``reader`` is a
    :class:`soccer_vision.io.video.VideoReader`. Samples are spread evenly across
    a track's span so a gallery entry covers the poses and lighting the player
    actually appeared in. Returns ``{track_id: (M, 512)}``, skipping tracks whose
    boxes were all too small to crop.
    """
    out: dict[int, np.ndarray] = {}
    items = [(t, s) for t, s in track_boxes.items() if track_ids is None or t in track_ids]

    for tid, samples in items:
        if not samples:
            continue
        step = max(1, len(samples) // max_samples_per_track)
        crops = []
        for frame_no, bbox in samples[::step][:max_samples_per_track]:
            frame = reader.read_frame(int(frame_no))
            if frame is None:
                continue
            crop = crop_player(frame, bbox)
            if crop is not None:
                crops.append(crop)
        if not crops:
            continue
        out[tid] = embedder.embed(crops)
        if progress:
            print(f"  track {tid}: {len(crops)} crops embedded")

    return out


def _ensure_backbone(cache_dir: Path) -> Path:
    """Return the slim backbone checkpoint, fetching + stripping it on first use."""
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    slim = cache_dir / SLIM_CHECKPOINT
    if slim.exists():
        return slim

    full = cache_dir / "sportsreid_osnet_x1_0.pth"
    if not full.exists():
        import gdown

        print(f"Downloading sportsreid OSNet_x1_0 weights → {full} (~1 GB, once)")
        gdown.download(id=SPORTSREID_OSNET_GDRIVE_ID, output=str(full), quiet=False)

    state = torch.load(full, map_location="cpu", weights_only=False)["state_dict"]
    backbone = {
        k.removeprefix("module."): v
        for k, v in state.items()
        if not k.removeprefix("module.").startswith("classifier.")
    }
    torch.save(backbone, slim)
    print(f"Cached backbone-only checkpoint → {slim}")
    return slim
