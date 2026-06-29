"""Full pipeline CLI: soccer-vision process."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import yaml


def run_pipeline(args):
    """Execute the full 9-step pipeline on a match video."""
    from soccer_vision.broadcast.virtual_cam import BroadcastConfig, generate_broadcast_proxy
    from soccer_vision.clips.extract import extract_event_clips
    import numpy as np
    import supervision as sv

    from soccer_vision.detection.ball import detect_ball_position
    from soccer_vision.detection.field_filter import filter_spectators
    from soccer_vision.detection.rfdetr import ALL_PERSON_CLASS_IDS, RFDETRSoccerDetector
    from soccer_vision.events.phases import classify_phase
    from soccer_vision.events.set_piece import detect_all_set_pieces
    from soccer_vision.io.osl import add_event, new_osl_document, write_osl
    from soccer_vision.io.project import RunDir
    from soccer_vision.io.video import VideoReader
    from soccer_vision.metrics.distance import distance_per_player
    from soccer_vision.registration.hough import compute_homography, pixel_to_field
    from soccer_vision.store.db import MatchDB
    from soccer_vision.tracking.bytetrack import create_tracker, track_detections
    from soccer_vision.verify.sheets import build_contact_sheet

    video_path = Path(args.video)
    match_id = args.match_id or str(uuid.uuid4())[:8]
    run_dir = RunDir(Path(args.out_dir), match_id)

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    broadcast_config = BroadcastConfig()
    if "broadcast" in config:
        broadcast_config = BroadcastConfig.from_yaml(args.config)

    print("=== soccer-vision process ===")
    print(f"Video:    {video_path}")
    print(f"Match ID: {match_id}")
    print(f"Output:   {run_dir.root}")

    # Step 1: Load video
    print("\n[Step 1] Loading video...")
    reader = VideoReader(video_path)
    native_fps = reader.fps
    total_frames = reader.total_frames
    print(f"  {total_frames} frames @ {native_fps:.2f} fps ({reader.duration_s / 60:.1f} min)")
    reader.close()

    # Step 2: Virtual broadcast proxy
    print("\n[Step 2] Generating broadcast proxy...")
    device = args.device
    detector = RFDETRSoccerDetector.from_pretrained(device=device)
    generate_broadcast_proxy(
        video_path,
        run_dir.broadcast_proxy,
        config=broadcast_config,
        detector=detector,
        metadata_path=run_dir.crop_metadata,
    )

    # Step 3: Ball detection on proxy
    print("\n[Step 3] Ball detection...")
    proxy_reader = VideoReader(run_dir.broadcast_proxy)
    proxy_fps = proxy_reader.fps
    detect_interval = max(1, int(round(proxy_fps / 5)))  # 5 fps detection

    ball_positions = []
    H_cache = None
    h_recompute_interval = int(proxy_fps * 60 * 5)  # every 5 min
    last_h_frame = -h_recompute_interval

    # Step 4: Player tracking
    print("\n[Step 4] Player tracking...")
    tracker = create_tracker(frame_rate=int(proxy_fps))
    player_tracks: dict[int, list[tuple[float, float]]] = {}

    for fn, frame in proxy_reader.sample_frames(detect_interval):
        # Detect all objects
        detections = detector.predict(frame)

        # Filter spectators: keep ball detections, filter people by field position
        ball_mask = ~np.isin(detections.class_id, list(ALL_PERSON_CLASS_IDS))
        person_mask = np.isin(detections.class_id, list(ALL_PERSON_CLASS_IDS))
        ball_dets = detections[ball_mask]
        person_dets = filter_spectators(
            detections[person_mask], H_cache, frame.shape,
        )
        detections = sv.Detections.merge([ball_dets, person_dets])

        tracked = track_detections(tracker, detections)

        # Ball
        ball = detect_ball_position(frame, detector)
        if ball is not None:
            bx, by, bconf = ball

            # Step 5: Field registration
            if fn - last_h_frame >= h_recompute_interval:
                H_new, ok = compute_homography(frame)
                if ok:
                    H_cache = H_new
                last_h_frame = fn

            fx, fy = None, None
            if H_cache is not None:
                fx, fy = pixel_to_field(bx, by, H_cache)

            ball_positions.append({
                "frame": fn,
                "timestamp_s": round(fn / proxy_fps, 2),
                "pixel_x": bx,
                "pixel_y": by,
                "field_x": fx,
                "field_y": fy,
                "confidence": bconf,
            })

        # Player field positions for metrics
        if tracked.tracker_id is not None and H_cache is not None:
            for i, tid in enumerate(tracked.tracker_id):
                x1, y1, x2, y2 = tracked.xyxy[i]
                foot_x = (x1 + x2) / 2
                foot_y = y2  # bottom of bbox
                fx, fy = pixel_to_field(foot_x, foot_y, H_cache)
                player_tracks.setdefault(int(tid), []).append((fx, fy))

        if fn % 500 == 0:
            print(f"  Processing frame {fn}/{proxy_reader.total_frames}")

    proxy_reader.close()

    # Step 6: Event detection
    print("\n[Step 6] Event detection (set-piece heuristics)...")
    events = detect_all_set_pieces(ball_positions)
    events = classify_phase(events)
    print(f"  Found {len(events)} set-piece events")

    # Step 7: Metrics
    print("\n[Step 7] Computing metrics...")
    distances = distance_per_player(player_tracks)
    stats = {
        "match_id": match_id,
        "total_events": len(events),
        "event_counts": {},
        "distance_per_player": {str(k): round(v, 1) for k, v in distances.items()},
    }
    for e in events:
        label = e["label"]
        stats["event_counts"][label] = stats["event_counts"].get(label, 0) + 1

    stats_path = run_dir.stats
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved: {stats_path}")

    # Step 8: Database + OSL export
    print("\n[Step 8] Database logging + OSL export...")
    osl_doc = new_osl_document(
        match_id,
        video_path=str(video_path),
        fps=proxy_fps,
        field_dimensions={"width": 55.0, "height": 36.0},
    )
    for e in events:
        add_event(osl_doc, label=e["label"], position_ms=e["position_ms"],
                  frame=e.get("frame"), confidence=e.get("confidence"))

    write_osl(osl_doc, run_dir.annotations)
    print(f"  OSL JSON: {run_dir.annotations}")

    db = MatchDB(Path(args.out_dir) / "soccer_vision.db")
    db.add_match(
        match_id,
        raw_path=str(video_path),
        proxy_path=str(run_dir.broadcast_proxy),
        osl_path=str(run_dir.annotations),
        stats_path=str(stats_path),
    )
    for e in events:
        db.add_event(match_id, e["label"], e["position_ms"],
                     frame=e.get("frame"), confidence=e.get("confidence"))

    # Step 9: Clip extraction
    print("\n[Step 9] Extracting clips...")
    if events:
        clip_paths = extract_event_clips(
            run_dir.broadcast_proxy,
            events,
            run_dir.clips_dir,
            pre_s=config.get("clips", {}).get("pre_s", 5.0),
            post_s=config.get("clips", {}).get("post_s", 15.0),
        )
        for clip_path, event in zip(clip_paths, events):
            db.add_clip(match_id, str(clip_path), pre_s=5.0, post_s=15.0)
        print(f"  {len(clip_paths)} clips extracted")

    # Contact sheets
    print("\n[Verify] Building contact sheets...")
    frame_data = [{"frame": e["frame"], "timestamp_s": e["timestamp_s"],
                    "label": e["label"]} for e in events if "frame" in e]
    if frame_data:
        sheets = build_contact_sheet(run_dir.broadcast_proxy, frame_data, run_dir.sheets_dir)
        print(f"  {len(sheets)} contact sheet(s) saved")

    print("\n=== Done ===")
    print(f"Run directory: {run_dir.root}")
    print(f"Events: {len(events)}")
    print(f"Next: soccer-vision extract --run {run_dir.root}")


def run_broadcast_only(args):
    """Generate broadcast proxy only."""
    from soccer_vision.broadcast.virtual_cam import BroadcastConfig, generate_broadcast_proxy

    config = BroadcastConfig()
    if args.config:
        config = BroadcastConfig.from_yaml(args.config)

    out_dir = Path(args.out) if args.out else Path("runs/broadcast")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "broadcast_proxy.mp4"

    print("Generating broadcast proxy...")
    generate_broadcast_proxy(
        args.video,
        out_path,
        config=config,
        metadata_path=out_dir / "crop_metadata.json",
    )
    print(f"Done: {out_path}")
