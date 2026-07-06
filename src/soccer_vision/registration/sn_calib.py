"""sn-calibration adapter: DeepLabv3 line segmentation → homography.

Uses the DeepLabv3-ResNet50 trained on SoccerNet calibration data to detect
semantic field lines, then fits a homography from the four boundary lines
(touchlines + goal lines) to field metres.

Falls back gracefully to Hough when:
- Weights are not present
- Fewer than 2 boundary lines are detected with enough pixels

SoccerNet calibration line class semantics (1-indexed, background=0):
  Classes 1-4 are the field boundary lines most useful for homography:
    1  Big rect. left bottom   (left goal line, bottom half)
    2  Big rect. left main     (left touchline)
    3  Big rect. left top      (left goal line, top half)
    4  Big rect. right bottom  (right goal line, bottom half)
    5  Big rect. right main    (right touchline)
    6  Big rect. right top     (right goal line, top half)
  Classes 7-26 are internal lines (penalty area, center circle, etc.)
  We use classes 1-6 to extract the four field edges.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

from soccer_vision.registration.hough import FIELD_H_M, FIELD_W_M

NUM_LINE_CLASSES = 27
_DEFAULT_WEIGHTS = Path("/blue/ewhite/b.weinstein/soccer-vision/weights/deeplabv3_sn.pth")

# Line classes that correspond to the four field boundary edges
_LEFT_GOAL_LINE_CLASSES = {1, 3}    # Big rect. left bottom / top
_RIGHT_GOAL_LINE_CLASSES = {4, 6}   # Big rect. right bottom / top
_TOP_TOUCHLINE_CLASS = {2}          # Big rect. left main (== top touchline in image)
_BOTTOM_TOUCHLINE_CLASS = {5}       # Big rect. right main (== bottom touchline in image)
_BOUNDARY_CLASSES = _LEFT_GOAL_LINE_CLASSES | _RIGHT_GOAL_LINE_CLASSES | \
                    _TOP_TOUCHLINE_CLASS | _BOTTOM_TOUCHLINE_CLASS

# Minimum pixels in a line mask to trust it
_MIN_LINE_PIXELS = 50


def _load_model(weights_path: Path):
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(weights=None, num_classes=NUM_LINE_CLASSES)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _preprocess(frame: np.ndarray, size: int = 960) -> "torch.Tensor":
    import torch
    rgb = frame[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(rgb, (new_w, new_h))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    # ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), scale


def _fit_line_to_mask(mask: np.ndarray) -> tuple[float, float] | None:
    """Fit a single line (rho, theta) to nonzero pixels in a binary mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < _MIN_LINE_PIXELS:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    # PCA: first principal component is the line direction
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    dx, dy = vt[0]
    # Convert to (rho, theta) form
    theta = np.arctan2(dy, dx) + np.pi / 2  # normal direction
    rho = mean[0] * np.cos(theta) + mean[1] * np.sin(theta)
    return float(rho), float(theta)


def _intersect_lines(r1, t1, r2, t2):
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    b = np.array([r1, r2])
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


class SNCalibration:
    """Neural field registration using pretrained DeepLabv3.

    Usage:
        calib = SNCalibration.from_weights()          # load once
        H, ok = calib.compute_homography(frame)       # per frame
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    @classmethod
    def from_weights(
        cls,
        weights_path: Path | str | None = None,
        device: str | None = None,
    ) -> SNCalibration:
        import torch
        weights_path = Path(weights_path or _DEFAULT_WEIGHTS)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"sn-calibration weights not found: {weights_path}\n"
                "Download with: python training/sn_calib/train_calibration.py --download-weights"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _load_model(weights_path)
        return cls(model, device=device)

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Run DeepLabv3 segmentation, return (H, W) class-index mask (original resolution)."""
        import torch
        tensor, scale = _preprocess(frame)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.model(tensor)["out"]
        pred = out.argmax(dim=1).squeeze(0).cpu().numpy()  # (h_scaled, w_scaled)
        # Resize back to original frame size
        h, w = frame.shape[:2]
        pred_full = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return pred_full

    def compute_homography(
        self,
        frame: np.ndarray,
        field_w: float = FIELD_W_M,
        field_h: float = FIELD_H_M,
    ) -> tuple[np.ndarray | None, bool]:
        """Compute homography from frame to field metres.

        Returns (H, success). On failure returns (None, False).
        """
        seg = self.segment(frame)

        # Collect boundary line masks grouped by edge
        left_mask  = np.isin(seg, list(_LEFT_GOAL_LINE_CLASSES)).astype(np.uint8)
        right_mask = np.isin(seg, list(_RIGHT_GOAL_LINE_CLASSES)).astype(np.uint8)
        top_mask   = np.isin(seg, list(_TOP_TOUCHLINE_CLASS)).astype(np.uint8)
        bot_mask   = np.isin(seg, list(_BOTTOM_TOUCHLINE_CLASS)).astype(np.uint8)

        lines = {}
        for name, mask in [("left", left_mask), ("right", right_mask),
                            ("top", top_mask), ("bot", bot_mask)]:
            result = _fit_line_to_mask(mask)
            if result is not None:
                lines[name] = result

        if len(lines) < 3:
            return None, False

        # Build image corners from line intersections (use available pairs)
        def corner(l1, l2):
            if l1 in lines and l2 in lines:
                return _intersect_lines(*lines[l1], *lines[l2])
            return None

        tl = corner("top", "left")
        tr = corner("top", "right")
        bl = corner("bot", "left")
        br = corner("bot", "right")

        corners_img = [c for c in [tl, tr, bl, br] if c is not None]
        if len(corners_img) < 4:
            return None, False

        img_h, img_w = frame.shape[:2]
        margin = 0.35
        for cx, cy in corners_img:
            if (cx < -img_w * margin or cx > img_w * (1 + margin) or
                    cy < -img_h * margin or cy > img_h * (1 + margin)):
                return None, False

        src = np.array([tl, tr, bl, br], dtype=np.float32)
        dst = np.array([
            [0, 0], [field_w, 0],
            [0, field_h], [field_w, field_h],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return None, False
        return H, True


def is_available(weights_path: Path | str | None = None) -> bool:
    """Return True if pretrained weights are present on disk."""
    path = Path(weights_path or _DEFAULT_WEIGHTS)
    return path.exists()


def compute_homography_neural(
    frame: np.ndarray,
    model: SNCalibration,
    field_w: float = FIELD_W_M,
    field_h: float = FIELD_H_M,
) -> tuple[np.ndarray | None, bool]:
    """Convenience wrapper for pipeline use."""
    return model.compute_homography(frame, field_w, field_h)
