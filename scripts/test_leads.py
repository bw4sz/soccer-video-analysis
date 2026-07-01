"""Test all ready leads on sample fixtures and save annotated results.

Runs RF-DETR detection, ByteTrack tracking, and Hough registration on
the 2-minute sample clips. Saves annotated frames as contact sheets
for visual review.
"""

from pathlib import Path
import json
import time

import cv2
import numpy as np
import supervision as sv

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
RESULTS = Path(__file__).resolve().parent.parent / "tests" / "results"
RESULTS.mkdir(exist_ok=True)

SAMPLE_A = FIXTURES / "sample_match_a.mp4"
SAMPLE_B = FIXTURES / "sample_match_b.mp4"
# Fall back to 10s clips if 2-min not available
if not SAMPLE_A.exists():
    SAMPLE_A = FIXTURES / "clip_10s_a.mp4"
if not SAMPLE_B.exists():
    SAMPLE_B = FIXTURES / "clip_10s_b.mp4"


def test_rfdetr_detection(video_path: Path, out_dir: Path):
    """Run RF-DETR on sampled frames, save annotated results."""
    from soccer_vision.detection.rfdetr import RFDETRSoccerDetector, RFDETR_CLASSES

    print(f"\n{'='*60}")
    print(f"RF-DETR Detection: {video_path.name}")
    print(f"{'='*60}")

    t0 = time.time()
    detector = RFDETRSoccerDetector.from_pretrained()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps * 2))  # sample every 2 seconds

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    annotated_frames = []
    detection_log = []

    for fn in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            break

        t1 = time.time()
        detections = detector.predict(frame, conf_threshold=0.25)
        dt = time.time() - t1

        ts = fn / fps
        n_ball = int((detections.class_id == 0).sum()) if len(detections) > 0 else 0
        n_player = int(np.isin(detections.class_id, [1, 2]).sum()) if len(detections) > 0 else 0
        n_ref = int((detections.class_id == 3).sum()) if len(detections) > 0 else 0

        detection_log.append({
            "frame": fn, "timestamp_s": round(ts, 2),
            "balls": n_ball, "players": n_player, "referees": n_ref,
            "total": len(detections), "inference_ms": round(dt * 1000, 1),
        })

        if len(detections) > 0:
            labels = [
                f"{RFDETR_CLASSES.get(cid, '?')} {conf:.2f}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)
        else:
            annotated = frame.copy()

        # Add frame info overlay
        cv2.putText(annotated, f"F{fn} t={ts:.1f}s  det={len(detections)}  {dt*1000:.0f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        annotated_frames.append(annotated)

        if len(annotated_frames) % 10 == 0:
            print(f"  Frame {fn}/{total}: {len(detections)} detections ({dt*1000:.0f}ms)")

    cap.release()

    # Save annotated frames as contact sheet
    sheet_path = _build_sheet(annotated_frames, out_dir / f"rfdetr_{video_path.stem}.jpg")
    print(f"  Contact sheet: {sheet_path}")

    # Save detection log
    log_path = out_dir / f"rfdetr_{video_path.stem}_log.json"
    with open(log_path, "w") as f:
        json.dump(detection_log, f, indent=2)
    print(f"  Log: {log_path}")

    # Summary stats
    avg_ms = np.mean([d["inference_ms"] for d in detection_log])
    avg_det = np.mean([d["total"] for d in detection_log])
    ball_rate = np.mean([1 if d["balls"] > 0 else 0 for d in detection_log])
    print(f"\n  Summary:")
    print(f"    Frames sampled: {len(detection_log)}")
    print(f"    Avg inference: {avg_ms:.0f}ms")
    print(f"    Avg detections/frame: {avg_det:.1f}")
    print(f"    Ball detection rate: {ball_rate*100:.0f}%")

    return detector, detection_log


def test_bytetrack(video_path: Path, detector, out_dir: Path):
    """Run ByteTrack on detection results, save tracked frames."""
    print(f"\n{'='*60}")
    print(f"ByteTrack Tracking: {video_path.name}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps * 2))

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        frame_rate=int(fps),
    )

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    annotated_frames = []
    unique_ids = set()

    for fn in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.predict(frame, conf_threshold=0.25)
        tracked = tracker.update_with_detections(detections)

        if tracked.tracker_id is not None:
            for tid in tracked.tracker_id:
                unique_ids.add(int(tid))
            labels = [f"ID:{tid}" for tid in tracked.tracker_id]
            annotated = box_annotator.annotate(frame.copy(), tracked)
            annotated = label_annotator.annotate(annotated, tracked, labels)
        else:
            annotated = frame.copy()

        ts = fn / fps
        cv2.putText(annotated, f"F{fn} t={ts:.1f}s  tracks={len(tracked)}  unique={len(unique_ids)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        annotated_frames.append(annotated)

    cap.release()

    sheet_path = _build_sheet(annotated_frames, out_dir / f"bytetrack_{video_path.stem}.jpg")
    print(f"  Contact sheet: {sheet_path}")
    print(f"  Unique track IDs: {len(unique_ids)}")
    print(f"  Frames tracked: {len(annotated_frames)}")


def test_hough_registration(video_path: Path, out_dir: Path):
    """Run Hough-line field registration on sampled frames."""
    from soccer_vision.registration.hough import compute_homography, pixel_to_field

    print(f"\n{'='*60}")
    print(f"Hough Registration: {video_path.name}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps * 5))  # every 5 seconds

    annotated_frames = []
    n_success = 0
    n_total = 0

    for fn in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            break

        n_total += 1
        H, ok = compute_homography(frame)

        annotated = frame.copy()
        ts = fn / fps

        if ok:
            n_success += 1
            # Draw field coordinate grid projected back
            status = "OK"
            color = (0, 255, 0)

            # Test: project center of frame to field coords
            cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
            fx, fy = pixel_to_field(cx, cy, H)
            cv2.putText(annotated, f"Center -> ({fx:.1f}m, {fy:.1f}m)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            status = "FAIL"
            color = (0, 0, 255)

        cv2.putText(annotated, f"F{fn} t={ts:.1f}s  Hough: {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        annotated_frames.append(annotated)

    cap.release()

    sheet_path = _build_sheet(annotated_frames, out_dir / f"hough_{video_path.stem}.jpg")
    print(f"  Contact sheet: {sheet_path}")
    print(f"  Registration success: {n_success}/{n_total} ({n_success/max(1,n_total)*100:.0f}%)")


def _build_sheet(frames: list, out_path: Path, thumb_w=480, thumb_h=270, cols=4) -> Path:
    """Build a contact sheet from annotated frames."""
    import math

    thumbs = [cv2.resize(f, (thumb_w, thumb_h)) for f in frames]
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)

    rows_needed = math.ceil(len(thumbs) / cols)
    while len(thumbs) < rows_needed * cols:
        thumbs.append(blank.copy())

    rows = [np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(rows_needed)]
    sheet = np.vstack(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_path


if __name__ == "__main__":
    video = SAMPLE_A
    print(f"Using test video: {video} ({video.stat().st_size / 1e6:.1f} MB)")

    # 1. RF-DETR detection
    detector, det_log = test_rfdetr_detection(video, RESULTS)

    # 2. ByteTrack tracking
    test_bytetrack(video, detector, RESULTS)

    # 3. Hough registration
    test_hough_registration(video, RESULTS)

    print(f"\n{'='*60}")
    print(f"All results saved to: {RESULTS}")
    print(f"{'='*60}")
