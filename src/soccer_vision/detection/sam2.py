"""SAM (Segment Anything Model) player detection adapter.

SAM is a foundation model requiring no soccer-specific fine-tuning. It segments
any object with visual boundaries. We use automatic mask generation on the full
frame, then filter to player-sized segments on the field.

Overhead cameras (Veo, XbotGo Falcon+, Trace, Pixellot) see high visual
contrast between players and turf, so SAM masks are clean. This is simpler
and more robust than fine-tuning RF-DETR on domain-shifted footage.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

PLAYER_CLASS_ID = 1

# Expected player bbox area range (pixels²) for 1080p footage, scaling with resolution
MIN_PLAYER_AREA_RATIO = 0.0002  # 0.02% of frame (small players on overhead)
MAX_PLAYER_AREA_RATIO = 0.12    # 12% of frame (very large, unlikely)


class SAMPlayerDetector:
    """SAM-based player detector, outputs sv.Detections compatible with ByteTrack."""

    def __init__(self, device: str = "cuda", model_type: str = "base"):
        """Initialize SAM model.

        Args:
            device: "cuda" or "cpu"
            model_type: "vit_h", "vit_l", "vit_b" (from Meta's official checkpoints)
        """
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError:
            raise ImportError(
                "segment-anything not installed. "
                "Install with: pip install 'segment-anything'"
            )

        # Map our naming to official SAM checkpoint names
        checkpoint_map = {
            "base": "vit_b",
            "large": "vit_l",
            "huge": "vit_h",
            "tiny": "vit_b",  # Use base for tiny (no nano model)
        }
        vit_type = checkpoint_map.get(model_type, "vit_b")

        self.device = device
        self.model_type = model_type

        # SAM auto-downloads from Meta on first use (requires ~100-2400 MB depending on model)
        sam = sam_model_registry[vit_type]().to(device)
        self.generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,  # Grid density for prompting
            pred_iou_thresh=0.70,  # Relax from 0.88 to catch more masks
            stability_score_thresh=0.85,  # Relax from 0.95 for robustness
            crop_n_layers=0,  # Single layer (no multi-scale crops)
            min_mask_region_area=100,  # Minimum 100px² to filter noise
        )

    def predict(self, frame: np.ndarray, conf_threshold: float | None = None) -> sv.Detections:
        """Segment players in frame and return as bounding boxes.

        Args:
            frame: BGR numpy array (H, W, 3)
            conf_threshold: Unused (SAM doesn't output confidence scores for segments)

        Returns:
            sv.Detections with player bounding boxes
        """
        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1]  # BGR -> RGB for SAM

        # Generate masks for all objects in the frame
        masks = self.generator.generate(rgb)

        bboxes = []
        confidences = []

        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            stability = mask_dict.get("stability_score", 1.0)

            # Get bounding box from mask
            points = np.where(mask)
            if len(points[0]) == 0:
                continue

            y_min, y_max = points[0].min(), points[0].max()
            x_min, x_max = points[1].min(), points[1].max()
            x1, y1, x2, y2 = x_min, y_min, x_max, y_max
            w_box = x2 - x1
            h_box = y2 - y1
            area = w_box * h_box
            area_ratio = area / (h * w)

            # Filter by size: must be player-sized
            if not (MIN_PLAYER_AREA_RATIO < area_ratio < MAX_PLAYER_AREA_RATIO):
                continue

            # Filter by aspect ratio: players are typically taller than wide
            # (0.3-0.7 for standing players; 0.2-1.0 for prone/stretched)
            aspect = w_box / (h_box + 1e-6)
            if not (0.25 < aspect < 1.0):
                continue

            bboxes.append([x1, y1, x2, y2])
            confidences.append(stability)

        if not bboxes:
            return sv.Detections.empty()

        bboxes = np.array(bboxes, dtype=float)
        confidences = np.array(confidences, dtype=float)

        # All detections are players (SAM is agnostic; filtering by size/shape)
        class_ids = np.full(len(bboxes), PLAYER_CLASS_ID, dtype=int)

        return sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=class_ids,
        )


def is_available() -> bool:
    """Check if SAM dependencies are installed."""
    try:
        import segment_anything  # noqa: F401
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
