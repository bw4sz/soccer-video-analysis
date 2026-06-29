"""RF-DETR detector wrapper producing supervision.Detections.

Uses the rfdetr package with SoccerNet fine-tuned weights from
julianzu9612/RFDETR-Soccernet on HuggingFace.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
import torch
from PIL import Image

RFDETR_CLASSES = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}

BALL_CLASS_IDS = {0}
PLAYER_CLASS_IDS = {1, 3}  # player + goalkeeper
ALL_PERSON_CLASS_IDS = {1, 2, 3}

HF_MODEL_ID = "julianzu9612/RFDETR-Soccernet"
HF_WEIGHTS_FILE = "weights/checkpoint_best_regular.pth"
NUM_CLASSES = 4


class RFDETRSoccerDetector:
    """RF-DETR fine-tuned on SoccerNet, outputting sv.Detections."""

    def __init__(self, model, device: str = "cpu", conf_threshold: float = 0.3):
        self.model = model
        self.device = device
        self.conf_threshold = conf_threshold

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = HF_MODEL_ID,
        device: str | None = None,
    ) -> RFDETRSoccerDetector:
        from huggingface_hub import hf_hub_download
        from rfdetr import from_checkpoint

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        weights_path = hf_hub_download(model_id, HF_WEIGHTS_FILE)
        model = from_checkpoint(weights_path)

        return cls(model, device=device)

    def predict(self, frame: np.ndarray, conf_threshold: float | None = None) -> sv.Detections:
        """Run detection on a BGR numpy frame, return sv.Detections."""
        conf = conf_threshold or self.conf_threshold
        rgb = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb)

        with torch.no_grad():
            detections = self.model.predict(pil_image, threshold=conf)

        if detections is None or len(detections) == 0:
            return sv.Detections.empty()

        return detections

    def predict_ball(self, frame: np.ndarray, conf_threshold: float = 0.2) -> sv.Detections:
        """Detect ball only."""
        dets = self.predict(frame, conf_threshold=conf_threshold)
        if len(dets) == 0:
            return dets
        mask = np.isin(dets.class_id, list(BALL_CLASS_IDS))
        return dets[mask]

    def predict_players(self, frame: np.ndarray, conf_threshold: float = 0.3) -> sv.Detections:
        """Detect players and goalkeepers."""
        dets = self.predict(frame, conf_threshold=conf_threshold)
        if len(dets) == 0:
            return dets
        mask = np.isin(dets.class_id, list(PLAYER_CLASS_IDS))
        return dets[mask]
