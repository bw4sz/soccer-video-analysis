"""Full pipeline CLI: soccer-vision process."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import yaml


def run_pipeline(args):
    """Execute the full 9-step pipeline on a match video."""
    import numpy as np
    import supervision as sv

    from soccer_vision.broadcast.virtual_cam import BroadcastConfig, generate_broadcast_proxy
    from soccer_vision.clips.extract import extract_event_clips
    from soccer_vision.detection.ball import detect_ball_position
    from soccer_vision.detection.field_filter import filter_spectators
    from soccer_vision.detection.rfdetr import ALL_PERSON_CLASS_IDS, RFDETRSoccerDetector
    from soccer_vision.events.associate import associate_events, stamp_event_positions
    from soccer_vision.events.phases import classify_phase
    from soccer_vision.events.sources import ActionContext, active_detectors, run_detectors
    from soccer_vision.io.osl import add_event, new_osl_document, write_osl
    from soccer_vision.io.project import RunDir
    from soccer_vision.io.video import VideoReader
    from soccer_vision.metrics.distance import distance_per_player
    from soccer_vision.registration.hough import compute_homography, pixel_to_field
    from soccer_vision.store.db import MatchDB
    from soccer_vision.tracking.bytetrack import create_tracker, track_detections
    from soccer_vision.tracking.teams import TeamClassifier
    from soccer_vision.verify.sheets import build_contact_sheet

    video_path = Path(args.video)
    match_id = args.match_id or str(uuid.uuid4())[:8]
    run_dir = RunDir(Path(args.out_dir), match_id)

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    # --action-engine overrides which action-detection engines run.
    if getattr(args, "action_engine", None):
        config["action_engines"] = args.action_engine

    # Declared kit colours from the team profile name the two clusters
    # authoritatively (nearest-Lab match), instead of the camera-dependent HSV
    # heuristic that mislabels navy-rendered black kits (GitHub issue #11).
    kits: list[str] = []
    if getattr(args, "profile", None):
        from soccer_vision.profiles.loader import get_kits, load_profile
        kits = get_kits(load_profile(args.profile))
        if kits:
            print(f"  Team kits from profile: {', '.join(kits)}")

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

    # Step 2: Virtual broadcast proxy (opt-in — most footage doesn't need it)
    device = args.device
    detector_type = config.get("detector", {}).get("type", "rfdetr")

    # RF-DETR is the fallback ball detector. It is very unreliable on overhead
    # footage — on this match it "found" a ball in 71% of frames but with a p95
    # frame-to-frame jump of 1260px on a 1920px-wide frame (job 37883252), i.e.
    # mostly false positives in the trees and crowd. SAM3 prompted with
    # "soccer ball" cuts that to 208px, so the sam3 path below overrides it.
    ball_detector = RFDETRSoccerDetector.from_pretrained(device=device)

    # Player detection: SAM3 (text-prompted detect+track) or RF-DETR
    sam3_tracker = None
    sam3_ball = None
    if detector_type == "sam3":
        # SAM3 replaces BOTH the detector and ByteTrack: its video model returns
        # a persistent object id per player from a text prompt alone.
        from soccer_vision.tracking.sam3 import SAM3PlayerTracker
        prompt = config.get("detector", {}).get("prompt", "soccer player")
        print(f"  Detector: SAM3 text-prompt {prompt!r} (detect+track)")
        sam3_tracker = SAM3PlayerTracker(device=device, prompt=prompt)
        sam3_tracker.start()
        player_detector = None

        # Same model, second prompt: SAM3 tracks the ball far more reliably
        # than RF-DETR here (p95 jump 208px vs 1260px). Set
        # detector.ball_prompt: null to keep RF-DETR for the ball.
        ball_prompt = config.get("detector", {}).get("ball_prompt", "soccer ball")
        if ball_prompt:
            sam3_ball = SAM3PlayerTracker.sharing(sam3_tracker, prompt=ball_prompt)
            sam3_ball.start()
            print(f"  Ball: SAM3 text-prompt {ball_prompt!r} (shared weights)")
    elif detector_type == "sam":
        print("  Detector: SAM for players + RF-DETR for ball")
        from soccer_vision.detection.sam2 import SAMPlayerDetector
        player_detector = SAMPlayerDetector(device=device, model_type="base")
    else:
        # RF-DETR player confidence threshold (config: detector.conf_threshold,
        # default 0.3). Overhead cameras may need lower (e.g. 0.15) to recover
        # small players.
        conf_threshold = config.get("detector", {}).get("conf_threshold", 0.3)
        player_detector = ball_detector  # Use RF-DETR for both
        player_detector.conf_threshold = conf_threshold
        print(f"  Detector: RF-DETR (conf_threshold: {conf_threshold})")
    if getattr(args, "broadcast", False):
        print("\n[Step 2] Generating broadcast proxy...")
        generate_broadcast_proxy(
            video_path,
            run_dir.broadcast_proxy,
            config=broadcast_config,
            detector=ball_detector,
            metadata_path=run_dir.crop_metadata,
        )
    else:
        print("\n[Step 2] Skipping broadcast crop (pass --broadcast to enable) — "
              "using the source video as-is.")
        if run_dir.broadcast_proxy.exists() or run_dir.broadcast_proxy.is_symlink():
            run_dir.broadcast_proxy.unlink()
        run_dir.broadcast_proxy.symlink_to(video_path.resolve())

    # Step 3: Ball detection on proxy
    print("\n[Step 3] Ball detection...")
    proxy_reader = VideoReader(run_dir.broadcast_proxy)
    proxy_fps = proxy_reader.fps
    detect_interval = max(1, int(round(proxy_fps / 5)))  # 5 fps detection

    ball_positions = []
    ball_samples: list[dict] = []
    H_cache = None
    h_recompute_interval = int(proxy_fps * 60 * 5)  # every 5 min
    last_h_frame = -h_recompute_interval

    # Step 4: Player tracking
    print("\n[Step 4] Player tracking...")
    tracker = create_tracker(frame_rate=int(proxy_fps))
    player_tracks: dict[int, list[tuple[float, float]]] = {}
    # Per-frame player positions for event→player association, and jersey-colour
    # samples for team assignment.
    frame_players: dict[int, dict] = {}
    team_clf = TeamClassifier()

    for fn, frame in proxy_reader.sample_frames(detect_interval):
        if sam3_tracker is not None:
            # SAM3 detects AND tracks in one pass — the returned detections
            # already carry persistent tracker_ids, so ByteTrack is skipped.
            # Spectators are filtered after tracking (the prompt finds people
            # anywhere, including coaches/subs beyond the touchline).
            tracked = sam3_tracker.track(frame)
            tracked = filter_spectators(tracked, H_cache, frame.shape)
        else:
            # Detect players and ball
            person_dets = player_detector.predict(frame)

            # For RF-DETR, separate ball from people; for SAM, we only get people
            if detector_type == "sam":
                # SAM returns only player detections
                ball_dets = sv.Detections.empty()
            else:
                # RF-DETR returns mixed detections; separate by class_id
                person_mask = np.isin(person_dets.class_id, list(ALL_PERSON_CLASS_IDS))
                ball_dets = person_dets[~person_mask]
                person_dets = person_dets[person_mask]

            # Filter spectators: keep only field players
            person_dets = filter_spectators(
                person_dets, H_cache, frame.shape,
            )
            detections = sv.Detections.merge([ball_dets, person_dets])

            tracked = track_detections(tracker, detections)

        # Ball
        if sam3_ball is not None:
            bdets = sam3_ball.track(frame)
            if len(bdets):
                k = int(np.argmax(bdets.confidence))
                bx1, by1, bx2, by2 = bdets.xyxy[k]
                ball = ((bx1 + bx2) / 2, (by1 + by2) / 2, float(bdets.confidence[k]))
            else:
                ball = None
        else:
            ball = detect_ball_position(frame, ball_detector)
        # Record every sampled frame (visible or not) for the persisted ball
        # track. Kept separate from `ball_positions`, which the action engines
        # consume and which only carries frames where the ball was found.
        ball_samples.append({
            "frame": fn,
            "timestamp_s": round(fn / proxy_fps, 2),
            "visible": ball is not None,
            "pixel_x": float(ball[0]) if ball is not None else None,
            "pixel_y": float(ball[1]) if ball is not None else None,
            "confidence": float(ball[2]) if ball is not None else 0.0,
        })
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

        # Player positions (for metrics + event association) and jersey colour
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                x1, y1, x2, y2 = tracked.xyxy[i]
                foot_x = (x1 + x2) / 2
                foot_y = y2  # bottom of bbox

                fx = fy = None
                if H_cache is not None:
                    fx, fy = pixel_to_field(foot_x, foot_y, H_cache)
                    player_tracks.setdefault(tid, []).append((fx, fy))

                frame_players.setdefault(fn, {})[tid] = {
                    "pixel_x": float(foot_x),
                    "pixel_y": float(foot_y),
                    "field_x": fx,
                    "field_y": fy,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
                # SAM3 supplies a per-player mask; sampling jersey colour inside
                # it excludes turf, which otherwise collapses both teams to one
                # colour on overhead footage.
                pmask = tracked.mask[i] if tracked.mask is not None else None
                team_clf.add_sample(tid, frame, (x1, y1, x2, y2), pmask)

        if fn % 500 == 0:
            print(f"  Processing frame {fn}/{proxy_reader.total_frames}")

    proxy_reader.close()

    # Persist per-frame track boxes (transposed to per-track lists) so
    # `extract --halo` can draw a player spotlight across each clip window.
    tracks_by_id: dict[int, list[dict]] = {}
    for fn in sorted(frame_players):
        for tid, p in frame_players[fn].items():
            if p.get("bbox") is not None:
                tracks_by_id.setdefault(tid, []).append({"frame": fn, "bbox": p["bbox"]})
    # Persist the ball trajectory (previously computed then discarded).
    n_vis = sum(1 for s_ in ball_samples if s_["visible"])
    with open(run_dir.ball_track, "w") as f:
        json.dump(
            {
                "video": run_dir.broadcast_proxy.name,
                "fps": proxy_fps,
                "sample_fps": proxy_fps / detect_interval,
                "width": int(proxy_reader.width),
                "height": int(proxy_reader.height),
                "total_frames": int(proxy_reader.total_frames),
                "samples": ball_samples,
            },
            f,
        )
    if ball_samples:
        print(f"  Ball track: {n_vis}/{len(ball_samples)} samples visible "
              f"({100 * n_vis / len(ball_samples):.1f}%) -> {run_dir.ball_track}")

    with open(run_dir.tracks, "w") as f:
        json.dump(
            {"video": run_dir.broadcast_proxy.name, "fps": proxy_fps,
             "sample_interval": detect_interval,
             "tracks": {str(k): v for k, v in tracks_by_id.items()}},
            f,
        )

    # Assign players to teams by jersey colour, then run all available action
    # detectors (the rules engine today; the learned engine once a checkpoint ships).
    team_clf.fit(kits=kits)
    team_names = team_clf.team_names()
    if team_names:
        src = "profile kits" if kits else "colour heuristic"
        print(f"  Teams ({src}): {', '.join(sorted(team_names.values()))}")
    preview_path = run_dir.root / "teams_preview.png"
    if team_clf.build_team_preview(preview_path):
        print(f"  Team preview: {preview_path}")

    # Step 6: Action detection (pluggable engines, attribution-agnostic)
    print("\n[Step 6] Action detection...")
    ctx = ActionContext(
        fps=proxy_fps,
        ball_positions=ball_positions,
        frame_players=frame_players,
        proxy_path=str(run_dir.broadcast_proxy),
        config=config,
    )
    detectors = active_detectors(config)
    print(f"  Active action engines: {', '.join(d.name for d in detectors) or 'none'}")
    events = run_detectors(detectors, ctx)
    events = classify_phase(events)
    # Associate each event with the nearest player track and their team. Events
    # arrive with no coordinates, so anchor them on the ball first — without a
    # point to search from, association silently tags nothing.
    events = stamp_event_positions(events, ball_positions)
    events = associate_events(events, frame_players, team_clf)
    print(f"  Found {len(events)} events")

    # Step 7: Metrics
    print("\n[Step 7] Computing metrics...")
    distances = distance_per_player(player_tracks)
    stats = {
        "match_id": match_id,
        "total_events": len(events),
        "event_counts": {},
        "event_counts_by_team": {},
        "teams": team_names,
        "distance_per_player": {str(k): round(v, 1) for k, v in distances.items()},
    }
    for e in events:
        label = e["label"]
        stats["event_counts"][label] = stats["event_counts"].get(label, 0) + 1
        team = e.get("team") or "unknown"
        by_team = stats["event_counts_by_team"].setdefault(label, {})
        by_team[team] = by_team.get(team, 0) + 1

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
        extra = {k: e[k] for k in ("team", "track_id", "goal_zone") if e.get(k) is not None}
        add_event(osl_doc, label=e["label"], position_ms=e["position_ms"],
                  frame=e.get("frame"), confidence=e.get("confidence"),
                  source=e.get("source", "heuristic"), extra=extra or None)

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
    event_ids = []
    for e in events:
        eid = db.add_event(match_id, e["label"], e["position_ms"],
                           frame=e.get("frame"), confidence=e.get("confidence"),
                           team=e.get("team"), track_id=e.get("track_id"),
                           source=e.get("source"))
        event_ids.append(eid)

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
        for clip_path, event, eid in zip(clip_paths, events, event_ids):
            db.add_clip(match_id, str(clip_path), event_id=eid,
                        track_id=event.get("track_id"), team=event.get("team"),
                        pre_s=5.0, post_s=15.0)
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
