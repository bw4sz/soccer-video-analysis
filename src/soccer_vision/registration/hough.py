"""Hough-line homography fallback for field registration."""

from __future__ import annotations

import cv2
import numpy as np

FIELD_W_M = 55.0  # touchline (7v7)
FIELD_H_M = 36.0  # goal line to goal line
BOX_DEPTH_M = 5.5  # goal area depth


def detect_field_lines(frame: np.ndarray) -> tuple[list, list]:
    """Detect dominant horizontal/vertical lines via white-line HSV mask + Hough."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None:
        return [], []

    h_lines, v_lines = [], []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta)
        if angle_deg < 20 or angle_deg > 160:
            v_lines.append((rho, theta))
        elif 70 < angle_deg < 110:
            h_lines.append((rho, theta))
    return h_lines, v_lines


def _line_intersection(rho1: float, theta1: float, rho2: float, theta2: float):
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)],
    ])
    b = np.array([rho1, rho2])
    try:
        x, y = np.linalg.solve(A, b)
        return float(x), float(y)
    except np.linalg.LinAlgError:
        return None


def compute_homography(
    frame: np.ndarray,
    field_w: float = FIELD_W_M,
    field_h: float = FIELD_H_M,
) -> tuple[np.ndarray | None, bool]:
    """Compute homography from image coords to field metres via Hough lines.

    Returns (H, success). H is None on failure.
    """
    h_lines, v_lines = detect_field_lines(frame)
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None, False

    h_sorted = sorted(h_lines, key=lambda line: line[0])
    v_sorted = sorted(v_lines, key=lambda line: line[0])

    corners_img = [
        _line_intersection(*h_sorted[0], *v_sorted[0]),
        _line_intersection(*h_sorted[0], *v_sorted[-1]),
        _line_intersection(*h_sorted[-1], *v_sorted[0]),
        _line_intersection(*h_sorted[-1], *v_sorted[-1]),
    ]

    if any(c is None for c in corners_img):
        return None, False

    img_h, img_w = frame.shape[:2]
    margin = 0.3
    for cx, cy in corners_img:
        if cx < -img_w * margin or cx > img_w * (1 + margin):
            return None, False
        if cy < -img_h * margin or cy > img_h * (1 + margin):
            return None, False

    src = np.array(corners_img, dtype=np.float32)
    dst = np.array([
        [0, 0], [field_w, 0],
        [0, field_h], [field_w, field_h],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, False
    return H, True


def pixel_to_field(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    """Project a pixel coordinate to field metres."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0, 0, 0]), float(result[0, 0, 1])
