#!/usr/bin/env python3
"""Quick test: compare RF-DETR (conf=0.3 vs 0.15) vs SAM on Saints U11 match.

Run diagnostics on frame 17376 to see detection counts.
"""

from pathlib import Path
import json
import numpy as np
import supervision as sv
from soccer_vision.io.video import VideoReader
from soccer_vision.detection.rfdetr import RFDETRSoccerDetector, ALL_PERSON_CLASS_IDS
from soccer_vision.detection.field_filter import filter_spectators
from soccer_vision.verify.sheets import build_contact_sheet

try:
    from soccer_vision.detection.sam2 import SAMPlayerDetector, is_available as sam2_available
    SAM2_AVAILABLE = sam2_available()
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM not available, will skip comparison")


def test_frame(
    video_path: str,
    frame_num: int = 17376,
    out_dir: str = "diagnostics_comparison",
) -> None:
    """Compare detector outputs on a single frame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    reader = VideoReader(video_path)
    frame = reader.read_frame(frame_num)
    reader.close()

    h, w = frame.shape[:2]
    print(f"\nFrame {frame_num} ({h}×{w})")
    print("=" * 60)

    # RF-DETR @ 0.3 (original)
    print("\n[1/3] RF-DETR @ conf=0.3 (original)...")
    detector_rfdetr = RFDETRSoccerDetector.from_pretrained(device="cuda")
    dets_rf03 = detector_rfdetr.predict(frame, conf_threshold=0.3)
    person_mask = np.isin(dets_rf03.class_id, list(ALL_PERSON_CLASS_IDS))
    players_rf03 = dets_rf03[person_mask]
    print(f"  Players detected: {len(players_rf03)}")

    # RF-DETR @ 0.15 (lowered)
    print("\n[2/3] RF-DETR @ conf=0.15 (lowered)...")
    dets_rf15 = detector_rfdetr.predict(frame, conf_threshold=0.15)
    person_mask = np.isin(dets_rf15.class_id, list(ALL_PERSON_CLASS_IDS))
    players_rf15 = dets_rf15[person_mask]
    print(f"  Players detected: {len(players_rf15)}")

    # SAM
    if SAM2_AVAILABLE:
        print("\n[3/3] SAM player detector...")
        try:
            detector_sam = SAMPlayerDetector(device="cuda", model_type="base")
            players_sam2 = detector_sam.predict(frame)
            print(f"  Players detected: {len(players_sam2)}")
        except Exception as e:
            print(f"  Error: {e}")
            players_sam2 = sv.Detections.empty()
    else:
        print("\n[3/3] SAM skipped (not installed)")
        players_sam2 = sv.Detections.empty()

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  RF-DETR @ 0.3:  {len(players_rf03)} players")
    print(f"  RF-DETR @ 0.15: {len(players_rf15)} players (+{len(players_rf15) - len(players_rf03)})")
    if SAM2_AVAILABLE:
        print(f"  SAM2:           {len(players_sam2)} players")
    print("\nDiagnostic images saved to:", out_dir)

    # Save diagnostic images with bboxes drawn
    import cv2

    def draw_boxes(img, detections, label):
        img = img.copy()
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for (x1, y1, x2, y2) in detections.xyxy:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return img

    frame_rf03 = draw_boxes(frame, players_rf03, f"RF-DETR conf=0.3 ({len(players_rf03)} players)")
    frame_rf15 = draw_boxes(frame, players_rf15, f"RF-DETR conf=0.15 ({len(players_rf15)} players)")
    frame_sam2 = draw_boxes(frame, players_sam2, f"SAM2 ({len(players_sam2)} players)")

    cv2.imwrite(str(out_dir / "frame_rf03.png"), frame_rf03)
    cv2.imwrite(str(out_dir / "frame_rf15.png"), frame_rf15)
    cv2.imwrite(str(out_dir / "frame_sam2.png"), frame_sam2)

    print(f"  ✓ frame_rf03.png: {len(players_rf03)} bboxes")
    print(f"  ✓ frame_rf15.png: {len(players_rf15)} bboxes")
    if SAM2_AVAILABLE:
        if SAM2_AVAILABLE:
        print(f"  ✓ frame_sam2.png: {len(players_sam2)} bboxes")

    return {
        "frame": frame_num,
        "rf03_count": len(players_rf03),
        "rf15_count": len(players_rf15),
        "sam2_count": len(players_sam2),
    }


if __name__ == "__main__":
    import cv2

    video = Path("/orange/ewhite/b.weinstein/soccer-video-analysis/data/SaintsU11_OVF_Jul192026.MP4")
    if video.exists():
        result = test_frame(str(video), frame_num=17376)
        print("\n" + json.dumps(result, indent=2))
    else:
        print(f"Video not found: {video}")
