"""
track.py

Stage 1 of the YOLO pipeline: detect goal-kick candidates using ball tracking
and field registration. Cheaper and more precise than uniform frame sampling.

Algorithm:
  1. Sample every N frames (2–5 fps by default).
  2. Detect field lines with Hough → compute homography (image → field coords).
  3. Detect the ball with YOLOv8 (sports-ball class).
  4. Project ball position into field coordinates.
  5. Flag frames where the ball is inside the goal-area region AND has been
     nearly stationary for ≥ STATIONARY_FRAMES consecutive samples.
  6. Write candidates.json and a contact sheet for Claude visual verification.

Usage:
    python track.py --video match.mp4
    python track.py --video match.mp4 --fps 3 --team blue --half first
    python track.py --video match.mp4 --fps 5 --stationary-frames 4

Options:
    --video             Path to input video (required)
    --fps               Sample rate in frames-per-second (default: 3)
    --team              Team colour whose goal kicks to find: blue / white / both
                        (default: both — Claude filters in the verify step)
    --half              Match half to restrict: first / second / both (default: both)
    --half-frame        Frame number where second half starts (auto-detected if omitted)
    --stationary-frames Min consecutive stationary samples before flagging (default: 3)
    --stationary-px     Max pixel movement between samples to count as stationary
                        (in image coordinates, default: 40)
    --goal-zone-pct     Ball must be within this fraction of the field from either
                        goal line to count (0.25 = penalty-box depth, default: 0.25)
    --model             YOLOv8 model variant: n / s / m  (default: n — fastest)
    --out-dir           Output directory (default: candidates/)
    --no-contact-sheet  Skip generating the contact sheet (saves time if unneeded)
    --device            PyTorch device: cpu / cuda / mps (default: cpu)

Outputs:
    candidates/candidates.json   Frame list + field coords + confidence
    candidates/sheet_001.jpg     Contact sheet of candidate thumbnails (for Claude)
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

# Suppress ultralytics verbose output until we actually run inference
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard youth 7v7 field dimensions in metres (FIFA / OYSA guidelines)
# Used as the canonical field coordinate system for the homography.
FIELD_W_M = 55.0   # width  (touchline)
FIELD_H_M = 36.0   # height (goal line to goal line)

# 6-yard box dimensions in metres (half-width from post to post = 3.66 + 5.5 on each side)
# For 7v7 small-sided: roughly 10 m wide, 5 m deep
BOX_DEPTH_M = 5.5   # metres from goal line

THUMB_W, THUMB_H = 320, 180
SHEET_COLS = 5

# YOLOv8 COCO class index for 'sports ball'
SPORTS_BALL_CLASS = 32


# ---------------------------------------------------------------------------
# Field registration (Hough lines → homography)
# ---------------------------------------------------------------------------

def detect_field_lines(frame: np.ndarray):
    """
    Return the dominant horizontal and vertical lines found in the frame using
    a white-line HSV mask + Probabilistic Hough transform.
    Returns (h_lines, v_lines) as lists of (rho, theta) pairs, or empty lists.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White mask: low saturation, high value
    mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    # Morphological cleanup
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
        if angle_deg < 20 or angle_deg > 160:    # near-vertical
            v_lines.append((rho, theta))
        elif 70 < angle_deg < 110:               # near-horizontal
            h_lines.append((rho, theta))
    return h_lines, v_lines


def line_intersection(rho1, theta1, rho2, theta2):
    """Compute (x, y) intersection of two Hough lines."""
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


def compute_homography(frame: np.ndarray, field_w=FIELD_W_M, field_h=FIELD_H_M):
    """
    Attempt to compute a homography mapping image coords → field coords (metres).
    Returns (H, success_bool).

    Strategy: find four corners of the field (intersections of the outermost
    horizontal and vertical lines). Falls back gracefully — if fewer than 4
    lines are found, returns None and the caller uses pixel coords directly.
    """
    h_lines, v_lines = detect_field_lines(frame)
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None, False

    # Sort: h_lines by rho (top to bottom), v_lines by rho (left to right)
    h_sorted = sorted(h_lines, key=lambda l: l[0])
    v_sorted = sorted(v_lines, key=lambda l: l[0])

    top_h = h_sorted[0]
    bot_h = h_sorted[-1]
    lft_v = v_sorted[0]
    rgt_v = v_sorted[-1]

    corners_img = [
        line_intersection(*top_h, *lft_v),  # top-left
        line_intersection(*top_h, *rgt_v),  # top-right
        line_intersection(*bot_h, *lft_v),  # bottom-left
        line_intersection(*bot_h, *rgt_v),  # bottom-right
    ]

    if any(c is None for c in corners_img):
        return None, False

    # Sanity: corners should be roughly inside the frame
    H_img, W_img = frame.shape[:2]
    margin = 0.3  # allow 30% outside frame (camera may clip the field)
    for (cx, cy) in corners_img:
        if cx < -W_img * margin or cx > W_img * (1 + margin):
            return None, False
        if cy < -H_img * margin or cy > H_img * (1 + margin):
            return None, False

    src = np.array(corners_img, dtype=np.float32)
    dst = np.array([
        [0,       0],
        [field_w, 0],
        [0,       field_h],
        [field_w, field_h],
    ], dtype=np.float32)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, False
    return H, True


def pixel_to_field(px, py, H):
    """Project a pixel coordinate to field metres using homography H."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0, 0, 0]), float(result[0, 0, 1])


# ---------------------------------------------------------------------------
# Ball detection
# ---------------------------------------------------------------------------

def load_yolo(model_size="n", device="cpu"):
    """Load YOLOv8 model. Returns model or None on failure."""
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO(f"yolov8{model_size}.pt")
        model.to(device)
        return model
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")
    except Exception as e:
        sys.exit(f"Failed to load YOLO model: {e}")


def detect_ball(frame: np.ndarray, model, conf_thresh=0.25, imgsz=1280):
    """
    Return (cx, cy, conf) of the highest-confidence sports-ball detection,
    or None if none found.

    imgsz must be large for wide overhead (Veo) footage: at the YOLO default of
    640 the ball shrinks below detectability and nothing is found. 1280+ is
    required to see a ball in a full-pitch panorama.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model(frame, verbose=False, classes=[SPORTS_BALL_CLASS],
                        conf=conf_thresh, imgsz=imgsz)
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        # Pick highest confidence detection
        confs = boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        xyxy = boxes.xyxy[idx].cpu().numpy()
        cx = float((xyxy[0] + xyxy[2]) / 2)
        cy = float((xyxy[1] + xyxy[3]) / 2)
        return cx, cy, float(confs[idx])
    return None


# ---------------------------------------------------------------------------
# Goal-kick logic
# ---------------------------------------------------------------------------

def in_goal_zone(fx, fy, field_w=FIELD_W_M, field_h=FIELD_H_M,
                 box_depth=BOX_DEPTH_M):
    """
    Return which goal zone the field point (fx, fy) is in:
      'left'  — within box_depth of the left goal line
      'right' — within box_depth of the right goal line
      None    — not in a goal zone
    """
    if 0 <= fy <= field_h:
        if 0 <= fx <= box_depth:
            return "left"
        if field_w - box_depth <= fx <= field_w:
            return "right"
    return None


def is_stationary(positions, thresh_px):
    """
    Given a list of (cx, cy) pixel positions from recent samples,
    return True if all are within thresh_px of each other.
    """
    if len(positions) < 2:
        return False
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return math.sqrt(dx * dx + dy * dy) < thresh_px


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def build_contact_sheet(frames_data, video_path, out_dir: Path, fps):
    """
    Read the raw frames from video for each candidate, build a contact sheet.
    frames_data: list of dicts from candidates list
    """
    cap = cv2.VideoCapture(video_path)
    thumbs = []
    blank = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)

    for d in frames_data:
        fn = d["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            thumbs.append(blank.copy())
            continue
        thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
        ts = d["timestamp_s"]
        m, s = divmod(ts, 60)
        hms = f"{int(m):02d}:{s:04.1f}"
        zone = d.get("goal_zone", "?")
        label1 = f"F{fn}"
        label2 = f"{hms} [{zone}]"
        cv2.putText(thumb, label1, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(thumb, label2, (4, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (0, 255, 255), 1, cv2.LINE_AA)
        # Draw ball position estimate
        if d.get("ball_px") and d.get("ball_py"):
            bx = int(d["ball_px"] * THUMB_W / frame.shape[1])
            by = int(d["ball_py"] * THUMB_H / frame.shape[0])
            cv2.circle(thumb, (bx, by), 8, (0, 0, 255), 2)
        thumbs.append(thumb)

    cap.release()

    # Tile into sheets
    rows_per_sheet = 6
    per_sheet = SHEET_COLS * rows_per_sheet
    n_sheets = max(1, math.ceil(len(thumbs) / per_sheet))
    sheet_paths = []

    for s_idx in range(n_sheets):
        chunk = thumbs[s_idx * per_sheet:(s_idx + 1) * per_sheet]
        # Pad to a full sheet so every row has content (hstack on an empty
        # row slice would raise — see detect_actions.py for the same fix).
        while len(chunk) < per_sheet:
            chunk.append(blank.copy())
        rows = [np.hstack(chunk[r * SHEET_COLS:(r + 1) * SHEET_COLS])
                for r in range(rows_per_sheet)]
        sheet = np.vstack(rows)
        path = out_dir / f"sheet_{s_idx + 1:03d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sheet_paths.append(str(path))
        print(f"  Saved {path}")

    return sheet_paths


# ---------------------------------------------------------------------------
# Homography cache loader (for KpSFR pre-computed registrations)
# ---------------------------------------------------------------------------

def load_homography_cache(cache_path: str):
    """
    Load homographies.json produced by register.py.
    Returns a sorted list of (frame_no, H_or_None) pairs.
    """
    with open(cache_path) as f:
        data = json.load(f)
    result = []
    for entry in data["homographies"]:
        H = None
        if entry["H"] is not None:
            H = np.array(entry["H"], dtype=np.float64)
        result.append((entry["frame"], H))
    result.sort(key=lambda x: x[0])
    print(f"Loaded {len(result)} pre-computed homographies from {cache_path}")
    valid = sum(1 for _, H in result if H is not None)
    print(f"  {valid}/{len(result)} frames registered successfully")
    return result


def nearest_H(homography_cache, frame_no):
    """Return the nearest valid H from the cache for a given frame number."""
    best_H = None
    best_dist = float("inf")
    for fn, H in homography_cache:
        if H is None:
            continue
        dist = abs(fn - frame_no)
        if dist < best_dist:
            best_dist = dist
            best_H = H
    return best_H


# ---------------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------------

def run_tracking(args):
    video_path = args.video
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_min = total_frames / native_fps / 60
    print(f"Video: {total_frames} frames @ {native_fps:.2f} fps ({duration_min:.1f} min)")

    # Sample interval in frames
    sample_interval = max(1, int(round(native_fps / args.fps)))
    n_samples = total_frames // sample_interval
    print(f"Sampling @ {args.fps} fps → every {sample_interval} frames "
          f"({n_samples} samples)")

    # Optional half restriction
    half_start = 0
    half_end = total_frames
    if args.half != "both":
        if args.half_frame:
            mid = args.half_frame
        else:
            mid = total_frames // 2
        if args.half == "first":
            half_end = mid
        else:
            half_start = mid
    print(f"Analysing frames {half_start}–{half_end}")

    # Load YOLO
    print(f"Loading YOLOv8{args.model} on {args.device}…")
    model = load_yolo(args.model, args.device)
    print("Model ready.")

    # Homography source: pre-computed KpSFR cache or live Hough lines
    homography_cache = None
    if args.homography_cache:
        homography_cache = load_homography_cache(args.homography_cache)
        print("Field registration: KpSFR cache (register.py)")
    else:
        print("Field registration: live Hough lines (fallback)")

    # Live Hough cache (recomputed every 5 min when not using KpSFR cache)
    H_cache = None
    H_frame_interval = int(native_fps * 60 * 5)  # re-register every 5 min
    last_H_frame = -H_frame_interval

    # Sliding window of recent ball positions (pixel coords)
    recent_positions = []   # list of (cx, cy)
    recent_frames = []      # corresponding frame numbers

    candidates = []

    frame_nums = list(range(half_start, half_end, sample_interval))
    total = len(frame_nums)
    print(f"\nStarting tracking loop ({total} samples)…")

    for i, fn in enumerate(frame_nums):
        if i % 50 == 0:
            pct = i / total * 100
            ts = fn / native_fps
            m, s = divmod(ts, 60)
            print(f"  [{i:5d}/{total}] {pct:4.0f}% — {int(m):02d}:{s:04.1f}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue

        # Get homography for this frame
        if homography_cache is not None:
            # Use nearest pre-computed KpSFR registration
            H_cache = nearest_H(homography_cache, fn)
        elif fn - last_H_frame >= H_frame_interval:
            # Recompute Hough-line homography
            H_new, ok = compute_homography(frame)
            if ok:
                H_cache = H_new
            last_H_frame = fn

        # Detect ball
        ball = detect_ball(frame, model, imgsz=args.imgsz)
        if ball is None:
            recent_positions.clear()
            recent_frames.clear()
            continue

        bx, by, bconf = ball

        # Keep sliding window
        recent_positions.append((bx, by))
        recent_frames.append(fn)
        # Trim window to args.stationary_frames
        if len(recent_positions) > args.stationary_frames + 2:
            recent_positions.pop(0)
            recent_frames.pop(0)

        # Check stationarity
        if len(recent_positions) < args.stationary_frames:
            continue
        window_pos = recent_positions[-args.stationary_frames:]
        if not is_stationary(window_pos, args.stationary_px):
            continue

        # Project to field coordinates
        goal_zone = None
        fx, fy = None, None
        if H_cache is not None:
            fx, fy = pixel_to_field(bx, by, H_cache)
            goal_zone = in_goal_zone(fx, fy)
        else:
            # Fallback: use image-space heuristic
            # Ball in left/right 20% of frame width
            fw = frame.shape[1]
            if bx < fw * 0.20:
                goal_zone = "left"
            elif bx > fw * 0.80:
                goal_zone = "right"

        if goal_zone is None:
            continue

        # Deduplicate: skip if already have a candidate within 5 seconds
        ts = fn / native_fps
        if candidates and abs(ts - candidates[-1]["timestamp_s"]) < 5.0:
            continue

        m, s = divmod(ts, 60)
        hms = f"{int(m):02d}:{s:04.1f}"
        entry = {
            "frame": fn,
            "timestamp_s": round(ts, 2),
            "timestamp_hms": hms,
            "ball_conf": round(bconf, 3),
            "ball_px": round(bx, 1),
            "ball_py": round(by, 1),
            "goal_zone": goal_zone,
        }
        if fx is not None:
            entry["field_x_m"] = round(fx, 2)
            entry["field_y_m"] = round(fy, 2)

        candidates.append(entry)
        print(f"  *** Candidate: F{fn} {hms} — ball in {goal_zone} zone "
              f"(conf={bconf:.2f})")

    cap.release()

    print(f"\nFound {len(candidates)} candidate(s).")

    # Write candidates.json
    meta = {
        "video": video_path,
        "fps": native_fps,
        "sample_fps": args.fps,
        "sample_interval": sample_interval,
        "total_candidates": len(candidates),
        "candidates": candidates,
    }
    json_path = out_dir / "candidates.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"candidates.json → {json_path}")

    # Build contact sheet
    if not args.no_contact_sheet and candidates:
        print("\nBuilding contact sheet…")
        sheet_paths = build_contact_sheet(candidates, video_path, out_dir,
                                          native_fps)
        meta["sheets"] = sheet_paths
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\nDone. Next: show the sheet(s) to Claude for visual verification,")
    print(f"then run:")
    print(f"  python extract_clips.py --video {video_path} "
          f"--candidates {json_path} --pre 0 --post 20 --concat")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect goal-kick candidates via YOLO ball tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video", required=True, help="Input video file")
    p.add_argument("--fps", type=float, default=3,
                   help="Sample rate (frames per second, default: 3)")
    p.add_argument("--team", choices=["blue", "white", "both"], default="both",
                   help="Team whose goal kicks to flag (default: both)")
    p.add_argument("--half", choices=["first", "second", "both"], default="both",
                   help="Which half to analyse (default: both)")
    p.add_argument("--half-frame", type=int, default=None,
                   help="Frame where second half starts (auto = total/2)")
    p.add_argument("--stationary-frames", type=int, default=3,
                   help="Consecutive stationary samples needed (default: 3)")
    p.add_argument("--stationary-px", type=float, default=40,
                   help="Max pixel movement for 'stationary' (default: 40)")
    p.add_argument("--goal-zone-pct", type=float, default=0.25,
                   help="Fraction of field width for goal zone (default: 0.25)")
    p.add_argument("--model", choices=["n", "s", "m"], default="n",
                   help="YOLOv8 model size: n=nano s=small m=medium (default: n)")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="YOLO inference resolution. 640 (YOLO default) is too "
                        "small for wide overhead footage — the ball vanishes. "
                        "Use 1280+ (default: 1280)")
    p.add_argument("--out-dir", default="candidates",
                   help="Output directory (default: candidates/)")
    p.add_argument("--no-contact-sheet", action="store_true",
                   help="Skip contact sheet generation")
    p.add_argument("--device", default="cpu",
                   help="PyTorch device: cpu / cuda / mps (default: cpu)")
    p.add_argument("--homography-cache",
                   help="Path to homographies.json from register.py (KpSFR). "
                        "When provided, replaces the built-in Hough-line registration.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tracking(args)
