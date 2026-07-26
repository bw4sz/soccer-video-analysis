"""SAM3 player segmentation + tracking adapter.

SAM3's video model does *promptable concept segmentation*: give it the text
prompt ``"soccer player"`` and it detects, segments, and **tracks** every
matching instance across the video, maintaining a persistent object id per
player via its masklet memory. That replaces BOTH halves of the old pipeline —
the player detector and the separate ByteTrack pass — with one model, and the
persistent ids largely remove the per-frame identity flicker that ByteTrack
had to be filtered around.

Validated on Saints U11 (job 37864846): the prompt found 20-22 players/frame
with 21 of 22 track ids stable across >=50% of a 48-frame clip, against 5-6
players/frame for RF-DETR on the same footage.

Streaming mode (``start()`` then ``track()`` per frame) feeds frames one at a
time. Note the session's masklet memory bank grows ~0.12 GB per processed frame
and is never released, so ``SAM3PlayerTracker`` rotates the session in chunks
and re-stitches identities across the seam — see the class docstring.

Note: SAM3 finds *people*, not *on-field players* — coaches, subs and
spectators beyond the touchline are still returned, so the caller must keep
applying the field mask / spectator filter.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

PLAYER_CLASS_ID = 1
DEFAULT_MODEL = "facebook/sam3"
DEFAULT_PROMPT = "soccer player"


def is_available() -> bool:
    """Check if SAM3 dependencies (transformers with SAM3 + CUDA) are present."""
    try:
        import torch
        from transformers import Sam3VideoModel  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


def _masks_to_detections(
    object_ids,
    obj_id_to_mask,
    obj_id_to_score,
    height: int,
    width: int,
    min_score: float,
) -> sv.Detections:
    """Convert SAM3's per-object masks into sv.Detections with persistent ids."""
    import cv2
    import torch

    boxes, masks, scores, ids = [], [], [], []
    for oid in object_ids:
        oid = int(oid)
        m = obj_id_to_mask[oid]
        if isinstance(m, torch.Tensor):
            m = m.detach().float().cpu().numpy()
        m = np.asarray(m).squeeze()
        if m.ndim > 2:
            m = m.reshape(m.shape[-2], m.shape[-1])
        if m.shape != (height, width):
            m = cv2.resize(m.astype(np.float32), (width, height),
                           interpolation=cv2.INTER_NEAREST)
        # masks may arrive as logits (>1) or probabilities
        binary = m > (0.0 if m.max() > 1.0 else 0.5)
        if not binary.any():
            continue

        score = obj_id_to_score.get(oid, 1.0) if obj_id_to_score else 1.0
        if isinstance(score, torch.Tensor):
            score = float(score.detach().float().cpu().item())
        score = float(score)
        if score < min_score:
            continue

        ys, xs = np.where(binary)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        masks.append(binary)
        scores.append(score)
        ids.append(oid)

    if not boxes:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        mask=np.array(masks, dtype=bool),
        confidence=np.array(scores, dtype=float),
        class_id=np.full(len(boxes), PLAYER_CLASS_ID, dtype=int),
        tracker_id=np.array(ids, dtype=int),
    )


class SAM3PlayerTracker:
    """Text-prompted player detection + tracking via SAM3's video model.

    Uses **chunked sessions**. SAM3's session keeps a per-frame masklet memory
    bank that grows ~0.12 GB per processed frame and is never released (measured
    in job 37875082: 2.7 GB at frame 10 -> 20.6 GB at frame 150, while the object
    count stayed flat at ~45). A full match would need ~1.1 TB, so the session is
    rotated every ``chunk_frames`` frames and identities are re-stitched across
    the seam by mask IoU. Callers see one continuous set of ids.

    Usage (streaming, one frame at a time)::

        tracker = SAM3PlayerTracker(device="cuda")
        tracker.start()
        for frame in frames:                 # BGR ndarray
            dets = tracker.track(frame)      # sv.Detections with stable tracker_id
    """

    def __init__(
        self,
        device: str = "cuda",
        prompt: str = DEFAULT_PROMPT,
        model_id: str = DEFAULT_MODEL,
        min_score: float = 0.3,
        dtype=None,
        chunk_frames: int = 60,
        stitch_iou: float = 0.3,
    ):
        try:
            import torch
            from transformers import Sam3VideoModel, Sam3VideoProcessor
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "SAM3 needs transformers>=5.12 with SAM3 support. "
                "Install with: pip install 'transformers>=5.12'"
            ) from exc

        self.device = device
        self.prompt = prompt
        self.min_score = min_score
        self.chunk_frames = chunk_frames
        self.stitch_iou = stitch_iou
        self._dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)

        self.processor = Sam3VideoProcessor.from_pretrained(model_id)
        self.model = (
            Sam3VideoModel.from_pretrained(model_id, dtype=self._dtype)
            .to(device)
            .eval()
        )
        self.session = None
        # session-local obj id -> stable global id
        self._id_map: dict[int, int] = {}
        self._next_global_id = 0
        # masks from the most recent frame, keyed by global id (for re-stitching)
        self._prev_masks: dict[int, "np.ndarray"] = {}
        self._frame_in_chunk = 0
        self._restitch = False
        self.n_rotations = 0

    @classmethod
    def sharing(cls, other: "SAM3PlayerTracker", prompt: str, **kwargs) -> "SAM3PlayerTracker":
        """A second tracker with a different prompt, reusing loaded weights.

        SAM3 is one model driven by text, so tracking a second concept (e.g.
        ``"soccer ball"`` alongside ``"soccer player"``) needs only its own
        session — not a second ~3GB copy of the weights. Sessions stay separate
        because each keeps its own masklet memory and object ids.
        """
        new = cls.__new__(cls)
        new.__dict__.update({
            k: v for k, v in other.__dict__.items()
            if k in ("device", "processor", "model", "_dtype", "min_score",
                     "chunk_frames", "stitch_iou")
        })
        new.prompt = prompt
        for k, v in kwargs.items():
            setattr(new, k, v)
        new.session = None
        new._id_map = {}
        new._next_global_id = 0
        new._prev_masks = {}
        new._frame_in_chunk = 0
        new._restitch = False
        new.n_rotations = 0
        return new

    def start(self) -> None:
        """Open a streaming session and register the concept prompt."""
        self.session = self.processor.init_video_session(
            inference_device=self.device, dtype=self._dtype
        )
        self.processor.add_text_prompt(self.session, self.prompt)
        self._frame_in_chunk = 0
        self._id_map = {}

    def _rotate(self) -> None:
        """Drop the session (freeing its per-frame memory) and start a fresh one."""
        import torch

        self.session = None
        self._id_map = {}
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.start()
        self._restitch = True
        self.n_rotations += 1

    @staticmethod
    def _iou(a, b) -> float:
        inter = np.logical_and(a, b).sum()
        if inter == 0:
            return 0.0
        return float(inter) / float(np.logical_or(a, b).sum())

    def _assign_global_ids(self, dets: sv.Detections) -> sv.Detections:
        """Map session-local ids to stable global ids, stitching across rotations."""
        if dets.tracker_id is None or len(dets) == 0:
            return dets

        new_ids = []
        for i, local in enumerate(dets.tracker_id):
            local = int(local)
            gid = self._id_map.get(local)
            if gid is None:
                # On the first frame of a new chunk, try to inherit the identity
                # of the best-overlapping track from the previous chunk.
                if self._restitch and dets.mask is not None:
                    best, best_iou = None, self.stitch_iou
                    for pgid, pmask in self._prev_masks.items():
                        if pgid in new_ids:
                            continue
                        v = self._iou(dets.mask[i], pmask)
                        if v > best_iou:
                            best, best_iou = pgid, v
                    gid = best
                if gid is None:
                    gid = self._next_global_id
                    self._next_global_id += 1
                self._id_map[local] = gid
            new_ids.append(gid)

        dets.tracker_id = np.array(new_ids, dtype=int)
        return dets

    def track(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Stream one BGR frame; return detections carrying stable global ids."""
        import torch

        if self.session is None:
            self.start()
        elif self._frame_in_chunk >= self.chunk_frames:
            self._rotate()

        h, w = frame_bgr.shape[:2]
        # ascontiguousarray: the ::-1 BGR->RGB view has a negative stride, which
        # torch.from_numpy rejects inside the image processor.
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self._dtype)

        with torch.inference_mode():
            out = self.model(inference_session=self.session, frame=pixel_values[0])

        dets = _masks_to_detections(
            out.object_ids,
            out.obj_id_to_mask,
            getattr(out, "obj_id_to_score", None),
            h,
            w,
            self.min_score,
        )
        dets = self._assign_global_ids(dets)
        self._restitch = False
        self._frame_in_chunk += 1

        # Remember this frame's masks so the next rotation can stitch onto them.
        if dets.mask is not None and len(dets):
            self._prev_masks = {
                int(t): m for t, m in zip(dets.tracker_id, dets.mask)
            }
        return dets

    def close(self) -> None:
        """Release the session (and its cached per-frame memory)."""
        self.session = None
        self._id_map = {}
        self._prev_masks = {}
