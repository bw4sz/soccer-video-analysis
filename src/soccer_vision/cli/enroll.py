"""CLI: soccer-vision enroll — bank a team's appearances into a re-ID gallery.

Run once per team, then top up (``--append``) after each match: the gallery is
the thing you carry between matches, so `identify --method reid` can name players
without reading a single jersey number. See
:mod:`soccer_vision.identify.gallery` for why appearance beats OCR here.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def run_enroll(args):
    import numpy as np

    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.identify.enroll import names_from_jerseys
    from soccer_vision.identify.reid import ReIDEmbedder, embed_tracks
    from soccer_vision.io.video import VideoReader
    from soccer_vision.profiles.loader import load_profile

    profile = load_profile(args.profile) if args.profile else None
    exclude = {int(n) for n in (args.exclude_jersey or [])}

    print("=== soccer-vision enroll ===")

    # Labelling frames needs boxes on a couple of dozen frames — not a tracked,
    # team-clustered match. `--video` detects on those frames alone, so a squad
    # can be enrolled straight off raw footage with no `process` run at all.
    if args.video and args.dump_frames:
        print(f"Video:  {args.video}")
        _dump_frames_from_video(Path(args.video), Path(args.dump_frames), args)
        return

    # An annotation export ships beside the frames it labelled, so enrolling it
    # reads those JPEGs and needs neither a processed run nor the source video.
    if args.from_label_studio:
        _enroll_from_label_studio(args, profile)
        return

    if args.video:
        print("--video goes with --dump-frames or --from-label-studio; "
              "other modes read a processed run.")
        return

    if not args.run:
        print("enroll needs --run <run-dir>, --video with --dump-frames, "
              "or --from-label-studio <export.json>.")
        return

    run_dir = Path(args.run)
    tracks_path = run_dir / "tracks.json"
    proxy_path = run_dir / "broadcast_proxy.mp4"
    out_path = Path(args.out) if args.out else run_dir / "gallery.npz"

    if not proxy_path.exists():
        print(f"No broadcast_proxy.mp4 in {run_dir} — run `soccer-vision process` first.")
        return

    print(f"Run:    {run_dir}")

    if args.dump_frames:
        _dump_frames(run_dir, tracks_path, proxy_path, Path(args.dump_frames), args)
        return

    if args.dump_tracklets:
        _dump_tracklets(run_dir, tracks_path, proxy_path,
                        Path(args.dump_tracklets), args, profile)
        return

    if args.from_tracklets:
        _enroll_from_tracklets(run_dir, tracks_path, proxy_path, out_path, args, profile)
        return

    jerseys_path = run_dir / "jerseys.json"
    if not jerseys_path.exists():
        print(f"No jerseys.json in {run_dir} — run `soccer-vision identify` "
              "first, or enrol annotations with --from-label-studio.")
        return

    print("Loading re-id backbone...")
    embedder = ReIDEmbedder.from_pretrained(weights=args.weights, device=args.device)

    doc = json.loads(jerseys_path.read_text())
    track_names = names_from_jerseys(
        doc, profile,
        min_confidence=args.min_confidence,
        min_obs=args.min_obs,
        exclude=exclude,
    )
    print(f"Source: jerseys.json — {len(track_names)} tracks above threshold "
          f"(conf>={args.min_confidence}, n_obs>={args.min_obs})")
    if not track_names:
        print("Nothing to enrol. Lower --min-confidence or annotate frames.")
        return

    reader = VideoReader(proxy_path)
    try:
        per_track = embed_tracks(
            load_track_boxes(tracks_path), embedder, reader,
            max_samples_per_track=args.max_samples,
            track_ids=set(track_names),
        )
    finally:
        reader.close()

    embeddings = (np.concatenate(list(per_track.values())) if per_track
                  else np.zeros((0, 512), dtype=np.float32))
    names = [track_names[t] for t, e in per_track.items() for _ in range(len(e))]

    _save_gallery(embeddings, names, out_path, args,
                  next_hint=f"--run {run_dir} --method reid --gallery {out_path}")


def _save_gallery(embeddings, names: list[str], out_path: Path, args, *, next_hint: str):
    """Build, optionally merge, and write the gallery — then say what's in it."""
    from soccer_vision.identify.gallery import (
        build_gallery,
        load_gallery,
        merge_galleries,
        save_gallery,
    )

    if len(embeddings) == 0:
        print("No crops could be embedded — nothing enrolled.")
        return

    gallery = build_gallery(embeddings, names, max_per_player=args.max_per_player)
    if args.append and out_path.exists():
        existing = load_gallery(out_path)
        gallery = merge_galleries(existing, gallery, max_per_player=args.max_per_player)
        print(f"Appended to existing gallery ({len(existing['names'])} players in it)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gallery(gallery, out_path)

    counts = Counter(gallery["names"][i] for i in gallery["label"])
    print(f"\nGallery: {len(gallery['names'])} players, {len(gallery['emb'])} exemplars")
    for name, n in counts.most_common():
        print(f"  {name:<24} {n} exemplars")
    print(f"Saved: {out_path}")
    print(f"Next: soccer-vision identify {next_hint}")


def _dump_tracklets(run_dir: Path, tracks_path: Path, proxy_path: Path,
                    out_dir: Path, args, profile):
    """Render windows of play with every lane ringed, plus the Label Studio project.

    One decision per lane instead of one per box, and the annotator sees motion
    and pitch position — the cues people actually use to tell youth players
    apart, and ones no still frame carries.
    """
    from soccer_vision.annotate.tracklets import (
        build_tasks,
        choose_windows,
        clip_url,
        labeling_config,
        render_window_clip,
        write_manifest,
    )
    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.profiles.loader import get_roster

    doc = json.loads(tracks_path.read_text())
    fps = float(doc.get("fps") or 30.0)
    teams = {int(k): v for k, v in (doc.get("teams") or {}).items()}
    track_samples = load_track_boxes(tracks_path)

    windows = choose_windows(
        track_samples, fps=fps, window_s=args.window,
        n_windows=args.n_windows, min_track_frames=args.min_track_frames,
        max_lanes=args.max_lanes, teams=teams, team=args.team,
        start_s=getattr(args, "at", None),
        all_lanes=getattr(args, "all_lanes", False),
    )
    if not windows:
        print(f"No lane lasted {args.min_track_frames} frames"
              f"{f' on team {args.team}' if args.team else ''}. "
              "Lower --min-track-frames, or drop --team.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    serve_root = Path(args.serve_root) if args.serve_root else out_dir.parent

    total_lanes = sum(len(w["lanes"]) for w in windows)
    crops = sum(lane["n_frames"] for w in windows for lane in w["lanes"])
    n_stretches = len({w["start_frame"] for w in windows})
    print(f"{len(windows)} tasks over {n_stretches} x {args.window:.0f}s of play, "
          f"{total_lanes} lanes ringed "
          f"({crops} crops behind them, {crops / max(1, total_lanes):.0f} per lane)")
    if getattr(args, "all_lanes", False):
        pages = max(w.get("n_pages", 1) for w in windows)
        print(f"  --all-lanes: every lane ringed, up to {pages} passes over the same "
              f"footage. Pages are longest-lane-first, so stopping early leaves a "
              f"gap you can measure rather than a biased sample.")

    urls = {}
    for w in windows:
        path = clips_dir / f"window_{w['window']:03d}_{int(w['start_s'])}s.mp4"
        print(f"  window {w['window']}/{len(windows)} at {w['start_s']:.0f}s "
              f"— {len(w['lanes'])} lanes", flush=True)
        render_window_clip(proxy_path, path, window=w,
                           track_samples=track_samples, fps=fps)
        urls[w["window"]] = clip_url(path, serve_root, args.serve_url)

    names = label_names(get_roster(profile) if profile else [])
    config = out_dir / "labeling_config.xml"
    config.write_text(labeling_config(names, args.max_lanes))
    tasks_path = out_dir / "label_studio_tasks.json"
    tasks_path.write_text(json.dumps(
        build_tasks(windows, urls, fps, max_lanes=args.max_lanes), indent=2))
    manifest = write_manifest(out_dir / "tracklets.json", run_dir=run_dir,
                              video=proxy_path, fps=fps, windows=windows,
                              max_lanes=args.max_lanes)

    size_mb = sum(p.stat().st_size for p in clips_dir.glob("*.mp4")) / 1e6
    print(f"\nWrote {len(windows)} clips ({size_mb:.0f} MB) → {clips_dir}")
    print(f"  config:   {config}")
    print(f"  tasks:    {tasks_path}")
    print(f"  manifest: {manifest}  (slot → track id; keep it, enrolment needs it)")
    if not names:
        print("  NOTE: no --profile roster, so the dropdowns offer only "
              f"'{'not ours'}'/'unsure' — pass --profile for one option per player.")
    print("\nNext: sync this folder to the machine running Label Studio, then")
    if args.serve_url:
        print(f"  cd {Path(serve_root).resolve()} && python -m http.server "
              f"{args.serve_url.rsplit(':', 1)[-1].rstrip('/') or 8000}")
        print("  label-studio start   # in another shell")
    else:
        print("  export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
        print(f"  export LOCAL_FILES_DOCUMENT_ROOT={Path(serve_root).resolve()}")
        print("  label-studio start")
    print("  # create project → paste config → import tasks")
    print(f"Then drop the export back here as {out_dir / 'annotations.json'} and:")
    print(f"  soccer-vision enroll --run {run_dir} "
          f"--from-tracklets {out_dir / 'annotations.json'} --out galleries/<team>.npz")


def _enroll_from_tracklets(run_dir: Path, tracks_path: Path, proxy_path: Path,
                           out_path: Path, args, profile):
    """Enrol every crop in each lane the annotator put a name to."""
    import numpy as np

    from soccer_vision.annotate.tracklets import boxes_from_tracklet_export
    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.identify.reid import ReIDEmbedder, crop_player
    from soccer_vision.io.video import VideoReader

    export_path = Path(args.from_tracklets)
    manifest_path = (Path(args.manifest) if args.manifest
                     else export_path.parent / "tracklets.json")
    if not export_path.exists():
        print(f"No such export: {export_path}")
        return
    if not manifest_path.exists():
        print(f"No tracklets.json beside the export ({manifest_path}). It carries the "
              "slot → track id map, which the export doesn't; point at it with "
              "--manifest.")
        return

    manifest = json.loads(manifest_path.read_text())
    track_samples = load_track_boxes(tracks_path)
    boxes, summary = boxes_from_tracklet_export(
        json.loads(export_path.read_text()), manifest, track_samples,
        max_samples_per_lane=args.max_samples,
    )
    boxes = [(f, b, roster_full_name(profile, n)) for f, b, n in boxes]

    print(f"Source: tracklets — {summary['lanes_named']} lanes named across "
          f"{summary['windows']} windows ({summary['lanes_skipped']} skipped as "
          f"not-ours/unsure), {len(boxes)} crops")
    if summary["matched_by_filename"]:
        print(f"  ({summary['matched_by_filename']} tasks matched on clip filename — "
              "the export lost its 'window' field, which Label Studio drops unless "
              "the tasks JSON itself was imported)")
    if summary["unmatched_tasks"]:
        print(f"  WARNING: {summary['unmatched_tasks']} task(s) matched no window in "
              f"{manifest_path} — is this the manifest that produced these clips?")
    for name, n in sorted(summary["per_player"].items(), key=lambda kv: -kv[1]):
        print(f"  {roster_full_name(profile, name):<24} {n} crops")
    if not boxes:
        print("Nothing named in the export — nothing to enrol.")
        return

    print("Loading re-id backbone...")
    embedder = ReIDEmbedder.from_pretrained(weights=args.weights, device=args.device)

    reader = VideoReader(proxy_path)
    crops, names = [], []
    try:
        by_frame: dict[int, list] = {}
        for frame_no, bbox, name in boxes:
            by_frame.setdefault(int(frame_no), []).append((bbox, name))
        for frame_no in sorted(by_frame):
            frame = reader.read_frame(frame_no)
            if frame is None:
                continue
            for bbox, name in by_frame[frame_no]:
                crop = crop_player(frame, bbox)
                if crop is not None:
                    crops.append(crop)
                    names.append(name)
    finally:
        reader.close()

    embeddings = embedder.embed(crops) if crops else np.zeros((0, 512), dtype=np.float32)
    _save_gallery(embeddings, names, out_path, args,
                  next_hint=f"--run {run_dir} --method reid --gallery {out_path}")


def _enroll_from_label_studio(args, profile):
    """Enrol the players named in a Label Studio export.

    The export is annotated against the frames ``--dump-frames`` wrote, and those
    JPEGs sit beside it — so the crops come off disk at the exact pixels the
    annotator drew, with no run directory and no video decode. A frame that
    travelled without its image is read from ``--video`` (or the run's proxy)
    instead, so an export moved on its own still enrols.

    Boxes left ``unknown`` are skipped rather than banked: they are the opponents
    and referees the annotator declined to name, and one enrolled under a shared
    ``unknown`` identity would match everybody.
    """
    import cv2

    from soccer_vision.identify.enroll import boxes_from_label_studio
    from soccer_vision.identify.reid import ReIDEmbedder, crop_player

    export_path = Path(args.from_label_studio)
    if not export_path.exists():
        print(f"No such export: {export_path}")
        return
    export = json.loads(export_path.read_text())

    boxes = boxes_from_label_studio(export)
    named, unresolved = named_boxes(boxes, profile)

    frames_with_boxes = {f for f, _, _ in named}
    print(f"Source: Label Studio — {len(named)} named boxes on "
          f"{len(frames_with_boxes)} frames "
          f"({len(boxes) - len(named)} left '{UNNAMED_LABEL}', skipped)")
    if unresolved:
        print("  Not on the roster, enrolled under the label as typed: "
              + ", ".join(f"{k} ({v})" for k, v in unresolved.most_common()))
        print("  A nickname belongs in the profile (`nickname: Mo`) so the gallery "
              "is keyed by the player's full name and --player still resolves.")
    if not named:
        print("Nothing named in the export — nothing to enrol.")
        return

    frames = _labelled_frames(export_path, args)
    print(f"Frames: {len(frames)} images beside the export"
          if frames else "Frames: none on disk beside the export")

    reader = None
    missing = sorted(frames_with_boxes - set(frames))
    if missing:
        video = (Path(args.video) if args.video
                 else Path(args.run) / "broadcast_proxy.mp4" if args.run else None)
        if video and video.exists():
            from soccer_vision.io.video import VideoReader
            print(f"  {len(missing)} labelled frames not on disk — reading from {video}")
            reader = VideoReader(video)
        else:
            print(f"  {len(missing)} labelled frames have no image and no --video "
                  "to read them from; their boxes are skipped.")

    out_path = Path(args.out) if args.out else export_path.parent / "gallery.npz"

    print("Loading re-id backbone...")
    embedder = ReIDEmbedder.from_pretrained(weights=args.weights, device=args.device)

    by_frame: dict[int, list] = {}
    for frame_no, bbox, name in named:
        by_frame.setdefault(int(frame_no), []).append((bbox, name))

    crops, names = [], []
    try:
        for frame_no in sorted(by_frame):
            path = frames.get(frame_no)
            image = cv2.imread(str(path)) if path else (
                reader.read_frame(frame_no) if reader else None)
            if image is None:
                continue
            for bbox, name in by_frame[frame_no]:
                crop = crop_player(image, bbox)
                if crop is not None:
                    crops.append(crop)
                    names.append(name)
    finally:
        if reader is not None:
            reader.close()

    import numpy as np

    embeddings = embedder.embed(crops) if crops else np.zeros((0, 512), dtype=np.float32)
    hint = f"--run runs/<match> --method reid --gallery {out_path}"
    _save_gallery(embeddings, names, out_path, args, next_hint=hint)


def named_boxes(boxes, profile):
    """Split parsed export boxes into enrollable ones and a tally of odd labels.

    Two things happen to a label here. Boxes left ``unknown`` are dropped — they
    are the opponents, referees and sideline figures the annotator declined to
    name, and banking them under one shared identity would produce a gallery
    entry that matches everybody. Everything else is resolved through the roster
    so the gallery is keyed by full names; a label that *doesn't* resolve is kept
    (it may be a deliberately hand-typed opponent) but counted, because the
    likelier cause is a nickname missing from the profile, which would silently
    split one player into two gallery entries.
    """
    named, unresolved = [], Counter()
    for frame_no, bbox, label in boxes:
        if str(label).strip().lower() == UNNAMED_LABEL:
            continue
        name = roster_full_name(profile, label)
        if profile and not _on_roster(profile, name):
            unresolved[label] += 1
        named.append((frame_no, bbox, name))
    return named, unresolved


FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _labelled_frames(export_path: Path, args) -> dict[int, Path]:
    """``{frame number: image path}`` for the frames the export was drawn on.

    Label Studio's export records its own server-side paths, which mean nothing
    on the machine enrolling it. The frame number does travel (in ``data.frame``,
    and in the filename ``--dump-frames`` chose), so images are matched by number
    against ``frames/`` beside the export — the layout the export folder already
    has when it comes back from a laptop.
    """
    roots = [Path(args.frames)] if getattr(args, "frames", None) else []
    roots += [export_path.parent / "frames", export_path.parent]

    for root in roots:
        if not root.is_dir():
            continue
        found: dict[int, Path] = {}
        for img in sorted(root.iterdir()):
            if img.suffix.lower() not in FRAME_SUFFIXES:
                continue
            digits = "".join(c for c in img.stem if c.isdigit())
            if digits:
                found.setdefault(int(digits), img)
        if found:
            return found
    return {}


def _on_roster(profile: dict | None, name: str) -> bool:
    """Whether ``name`` is a roster full name — i.e. a label that resolved."""
    from soccer_vision.profiles.loader import get_roster

    key = (name or "").strip().lower()
    return any((p.get("name") or "").strip().lower() == key for p in get_roster(profile or {}))


def _dump_frames_from_video(video: Path, out_dir: Path, args):
    """Export labelling frames straight from a video, detecting only on those frames.

    Going through ``process`` to get boxes on two dozen frames means tracking,
    team-clustering and ball-detecting every sampled frame of the match — on the
    U14G footage that was still unfinished after 5 GPU-hours (job 38176317).
    Detection alone is not the bottleneck: RF-DETR runs at ~0.1 s/frame on this
    same video, so the frames we actually export cost seconds.

    The tradeoff is no track ids and no kit colours, neither of which the
    labelling pass uses — the annotator supplies identity, which is the point.
    """
    import cv2

    from soccer_vision.annotate.label_studio import local_files_url
    from soccer_vision.cli.main import field_filter_kwargs
    from soccer_vision.detection.field_filter import filter_spectators
    from soccer_vision.io.video import VideoReader
    from soccer_vision.profiles.loader import get_roster, load_profile

    reader = VideoReader(video)
    total = reader.total_frames or 0
    if total <= 0:  # some containers don't report a count; fall back to duration
        total = int((reader.fps or 30) * 60 * 10)

    detector = _load_player_detector(args)

    # Detect on a wider pool than we keep, then keep the busiest frame per time
    # bin — same rule as the run-based path, which cannot be applied before
    # detection here because there are no tracks to count.
    pool = max(args.n_frames, args.n_frames * 3)
    step = max(1, total // (pool + 1))
    candidates = [step * (i + 1) for i in range(pool)]

    print(f"Detecting players on {len(candidates)} candidate frames "
          f"(keeping the busiest {args.n_frames})...")
    found: dict[int, list] = {}
    for frame_no in candidates:
        frame = reader.read_frame(frame_no)
        if frame is None:
            continue
        dets = detector(frame)
        dets = filter_spectators(dets, frame.shape, **field_filter_kwargs(args))
        h, w = frame.shape[:2]
        boxes = [b for b in _detection_boxes(dets) if _plausible_player_box(b, w, h)]
        boxes = labellable_boxes(boxes, h)
        if args.min_y_frac > 0:
            boxes = boxes_below(boxes, h, args.min_y_frac)
        if args.min_motion > 0:
            later = reader.read_frame(frame_no + int((reader.fps or 30) * 0.5))
            boxes = moving_boxes(boxes, frame, later, min_motion=args.min_motion)
        if len(boxes) >= args.min_players:
            found[frame_no] = boxes

    if not found:
        print(f"No frame had >= {args.min_players} players. Lower --min-players.")
        reader.close()
        return

    lo, hi = min(found), max(found)
    span = max(1, hi - lo + 1)
    bins: dict[int, int] = {}
    for f in sorted(found):
        b = min(args.n_frames - 1, (f - lo) * args.n_frames // span)
        if b not in bins or len(found[f]) > len(found[bins[b]]):
            bins[b] = f
    chosen = sorted(bins.values())

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    serve_root = Path(args.serve_root) if args.serve_root else out_dir.parent

    names = label_names(get_roster(load_profile(args.profile)) if args.profile else [])

    tasks, n_boxes = [], 0
    try:
        for frame_no in chosen:
            frame = reader.read_frame(frame_no)
            if frame is None:
                continue
            path = frames_dir / f"{frame_no:06d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            h, w = frame.shape[:2]
            results = [_rect_result(b, w, h, -1) for b in found[frame_no]]
            n_boxes += len(results)
            tasks.append({
                "data": {"image": local_files_url(path, serve_root),
                         "frame": frame_no,
                         "timestamp_s": round(frame_no / (reader.fps or 30), 2)},
                "predictions": [{"model_version": "soccer-vision-detector",
                                 "result": results}],
            })
    finally:
        reader.close()

    _write_frame_project(out_dir, tasks, names, n_boxes, serve_root)


def _load_player_detector(args):
    """Return ``frame -> sv.Detections`` for the player detector."""
    device = args.device or "cpu"
    from soccer_vision.detection.rfdetr import RFDETRSoccerDetector

    det = RFDETRSoccerDetector.from_pretrained(device=device)
    return det.predict_players


def _detection_boxes(dets) -> list:
    """Pixel boxes out of an ``sv.Detections``, empty list when there are none."""
    xyxy = getattr(dets, "xyxy", None)
    if xyxy is None or len(xyxy) == 0:
        return []
    return [tuple(float(v) for v in box) for box in xyxy]


def boxes_below(boxes: list, frame_h: int, min_y_frac: float) -> list:
    """Keep boxes whose feet are below ``min_y_frac`` of frame height.

    At a multi-pitch complex the wide view contains *another match*, so the
    extra detections are real soccer players and no appearance or motion cue
    separates them (measured: the far band has 22-26 mean frame-to-frame
    difference against 1.6-8 on the near pitch — the neighbours move more than
    our own game). Which pitch is ours is purely spatial, so it has to be told,
    not inferred. Feet, not centre: a player's feet locate them on the ground
    plane, which is what decides the pitch they're standing on.
    """
    floor = min_y_frac * frame_h
    return [b for b in boxes if b[3] >= floor]


def moving_boxes(boxes: list, frame, later_frame, *, min_motion: float = 6.0) -> list:
    """Boxes whose pixels changed between two frames — players, not spectators.

    At a tournament complex the far touchline is lined with seated families and
    the next pitch is in shot, and they outnumber the players badly: on the U14G
    Veo footage ~40 of ~45 detections per frame were crowd. Neither box size nor
    the centre-rectangle field filter separates them, because a seated row at
    mid-distance renders the same 50px as a player.

    Motion does. The Veo camera is fixed, so differencing two frames half a
    second apart leaves the running players lit up and the crowd near zero. This
    is a stand-in for the field-boundary mask the pipeline still lacks (see
    "Field registration" in CLAUDE.md); it needs no model and no calibration.
    """
    import cv2
    import numpy as np

    if later_frame is None:
        return boxes
    a = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    b = cv2.cvtColor(later_frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.abs(a - b)
    h, w = diff.shape
    out = []
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        if float(diff[y1:y2, x1:x2].mean()) >= min_motion:
            out.append(box)
    return out


def labellable_boxes(boxes: list, frame_h: int, min_frac: float = 0.025) -> list:
    """Boxes big enough for a human to put a name to.

    Ranking candidate frames on raw detection count picks the wrong frames at a
    tournament complex: the busiest views are the ones taking in a crowd and the
    next pitch over, where every figure is 12px tall and unnameable. Counting
    only boxes above a fraction of frame height favours play close to the camera,
    which is where identity is actually readable — and the small boxes are
    dropped from the export too, since they are clutter to click past.
    """
    floor = max(20.0, min_frac * frame_h)
    return [b for b in boxes if (b[3] - b[1]) >= floor]


def _write_frame_project(out_dir: Path, tasks: list, names: list[str], n_boxes: int,
                         serve_root: Path):
    """Write the config + tasks for a frame-labelling project and say what's next."""
    config = out_dir / "labeling_config.xml"
    config.write_text(_frame_labeling_config(names))
    tasks_path = out_dir / "label_studio_tasks.json"
    tasks_path.write_text(json.dumps(tasks, indent=2))

    print(f"\nWrote {len(tasks)} frames ({n_boxes} pre-drawn boxes) → {out_dir / 'frames'}")
    print(f"  config: {config}")
    print(f"  tasks:  {tasks_path}")
    if not names:
        print("  NOTE: no --profile roster, so the label list is only "
              f"'{UNNAMED_LABEL}' — pass --profile to get one label per player.")
    print("\nNext: sync this folder to the machine running Label Studio, then")
    print(f"  export LOCAL_FILES_DOCUMENT_ROOT={Path(serve_root).resolve()}")
    print("  label-studio start   # create project → paste config → import tasks")
    print(f"Then drop the export back in here ({out_dir}/annotations.json) and:")
    print(f"  soccer-vision enroll --from-label-studio {out_dir / 'annotations.json'} "
          "--out galleries/<team>.npz")


def _dump_frames(run_dir: Path, tracks_path: Path, proxy_path: Path, out_dir: Path, args):
    """Write whole frames + a Label Studio project for naming players on them.

    The alternative to labelling track folders, and the better one when a lane's
    crops are too small to recognise anyone from. Here you see the **full frame**
    at native resolution and can zoom into it, so identity comes from the same
    cues you'd use watching the match — where someone is, who they're beside,
    which way play is going.

    Boxes come pre-drawn from ``tracks.json`` and labelled ``unknown``, so the
    job is renaming boxes rather than drawing them. The detector's pixel
    coordinates survive the round trip (Label Studio stores percentages;
    :func:`soccer_vision.identify.enroll.boxes_from_label_studio` converts back),
    which is what lets the model crop at whatever size it wants later — the human
    labels a person, not a thumbnail.
    """
    import cv2

    from soccer_vision.annotate.label_studio import local_files_url
    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.io.video import VideoReader
    from soccer_vision.profiles.loader import get_roster, load_profile

    tracks = [(t, s) for t, s in load_track_boxes(tracks_path).items()
              if len(s) >= args.min_track_frames]
    reader = VideoReader(proxy_path)

    wanted = (args.team or "").strip().lower()
    teams = _resolve_track_teams(run_dir, tracks_path, tracks, reader, args) if wanted else {}

    by_frame: dict[int, list] = {}
    for tid, samples in tracks:
        if wanted and (teams.get(tid) or "").lower() != wanted:
            continue
        for frame_no, bbox in samples:
            by_frame.setdefault(int(frame_no), []).append((tid, bbox))

    # Diversity is the point: a gallery built from one passage of play sees one
    # patch of pitch in one light. Spread the frames evenly over the whole match,
    # and only keep frames showing enough of the squad to be worth annotating.
    candidates = sorted(f for f, boxes in by_frame.items() if len(boxes) >= args.min_players)
    if not candidates:
        print(f"No frames with >= {args.min_players} "
              f"{'“' + wanted + '” ' if wanted else ''}players. "
              f"Lower --min-players, or check --team.")
        reader.close()
        return
    # One frame per equal time bin, and within a bin the *busiest* frame. Even
    # spacing alone lands on warm-ups and stoppages, where three players stand in
    # one corner of the pitch and the frame teaches the gallery nothing.
    lo, hi = candidates[0], candidates[-1]
    span = max(1, hi - lo + 1)
    bins: dict[int, int] = {}
    for f in candidates:
        b = min(args.n_frames - 1, (f - lo) * args.n_frames // span)
        if b not in bins or len(by_frame[f]) > len(by_frame[bins[b]]):
            bins[b] = f
    chosen = sorted(bins.values())

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    serve_root = Path(args.serve_root) if args.serve_root else run_dir.parent

    roster = get_roster(load_profile(args.profile)) if args.profile else []
    names = label_names(roster)

    tasks, n_boxes = [], 0
    try:
        for frame_no in chosen:
            frame = reader.read_frame(frame_no)
            if frame is None:
                continue
            path = frames_dir / f"{frame_no:06d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            h, w = frame.shape[:2]
            results = [_rect_result(bbox, w, h, tid) for tid, bbox in by_frame[frame_no]
                       if _plausible_player_box(bbox, w, h)]
            n_boxes += len(results)
            tasks.append({
                "data": {"image": local_files_url(path, serve_root),
                         "frame": frame_no,
                         "timestamp_s": round(frame_no / (reader.fps or 30), 2)},
                "predictions": [{"model_version": "soccer-vision-tracker", "result": results}],
            })
    finally:
        reader.close()

    _write_frame_project(out_dir, tasks, names, n_boxes, serve_root)


UNNAMED_LABEL = "unknown"


def label_names(roster: list[dict]) -> list[str]:
    """Label-list names for annotators: first names, since that's how a squad is known.

    A coach picking a player off a list wants "Morrighan", not "Morrighan
    Wright" — the surname is noise in a dropdown you hit twenty times a frame.
    Two players sharing a first name get a surname initial ("Morgan L."), so the
    list stays unambiguous without spelling anyone out.

    A roster ``nickname`` wins outright ("Mo", "Evie", "Izzy"). Teams call each
    other by nicknames, and a label whose owner the annotator has to translate is
    a label they will eventually mis-click; :func:`roster_full_name` maps it back
    so the gallery is still keyed by the roster's full name.
    """
    firsts = [str(p.get("name", "")).split()[0] for p in roster if p.get("name")]
    out = []
    for player in roster:
        name = str(player.get("name", "")).strip()
        if not name:
            continue
        nickname = str(player.get("nickname") or "").strip()
        if nickname:
            out.append(nickname)
            continue
        parts = name.split()
        if firsts.count(parts[0]) > 1 and len(parts) > 1:
            out.append(f"{parts[0]} {parts[1][0]}.")
        else:
            out.append(parts[0])
    return out


def roster_full_name(profile: dict | None, label: str) -> str:
    """Map a label the annotator picked back to the roster's full name.

    The gallery is keyed by whatever this returns, so keeping it in step with the
    profile is what lets `--player "Morrighan Wright"` and `--number 21` reach the
    same person later. Full name, first name, "Morgan L." and the roster
    ``nickname`` all land on the same player. An unrecognised label (an opponent
    someone named by hand) passes through untouched rather than being dropped.
    """
    from soccer_vision.profiles.loader import get_roster

    label = (label or "").strip()
    if not profile or not label:
        return label
    wanted = label.rstrip(".").lower()
    for player in get_roster(profile):
        name = str(player.get("name", "")).strip()
        if not name:
            continue
        parts = name.split()
        aliases = {name.lower(), parts[0].lower()}
        if len(parts) > 1:
            aliases.add(f"{parts[0]} {parts[1][0]}".lower())
        nickname = str(player.get("nickname") or "").strip().lower()
        if nickname:
            aliases.add(nickname)
        if wanted in aliases:
            return name
    return label


def _rect_result(bbox, frame_w: int, frame_h: int, track_id: int) -> dict:
    """One pre-drawn Label Studio rectangle for a detected player.

    Label Studio stores rectangles as percentages of the image, so the pixel box
    is converted here and converted back by
    :func:`soccer_vision.identify.enroll.boxes_from_label_studio` at enrolment —
    the annotator names a person and the exact detector coordinates survive, to
    be cropped at whatever size the model wants.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return {
        "from_name": "player", "to_name": "image", "type": "rectanglelabels",
        "original_width": frame_w, "original_height": frame_h, "image_rotation": 0,
        "value": {"x": 100 * x1 / frame_w, "y": 100 * y1 / frame_h,
                  "width": 100 * (x2 - x1) / frame_w, "height": 100 * (y2 - y1) / frame_h,
                  "rotation": 0, "rectanglelabels": [UNNAMED_LABEL]},
        "meta": {"text": [f"track {track_id}"]},
    }


def _plausible_player_box(bbox, frame_w: int, frame_h: int) -> bool:
    """Whether a box could be a standing player, used to keep junk off the sheet.

    The tracker occasionally emits a lane whose box balloons across most of the
    frame. One of those in Label Studio covers every real player and has to be
    clicked past on every task, so they're dropped here rather than annotated.
    A player on this footage is upright and small: taller than wide, well under a
    third of frame height.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    if w < 3 or h < 6:
        return False
    if w > 0.15 * frame_w or h > 0.35 * frame_h:
        return False
    return h > w


def _frame_labeling_config(names: list[str]) -> str:
    """Label Studio config: one label per roster player, plus ``unknown``.

    Every pre-drawn box arrives as ``unknown``; naming one is a two-click change,
    and anything left ``unknown`` is skipped at enrolment rather than guessed at.
    """
    from xml.sax.saxutils import escape

    labels = "\n".join(f'    <Label value="{escape(n)}"/>' for n in names)
    return (
        '<View>\n'
        '  <Header value="Name each player on your squad. '
        'Delete boxes for opponents, referees and anyone off the pitch."/>\n'
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" '
        'rotateControl="false"/>\n'
        '  <RectangleLabels name="player" toName="image">\n'
        f'{labels}\n'
        f'    <Label value="{UNNAMED_LABEL}" background="#888888"/>\n'
        '  </RectangleLabels>\n'
        '</View>\n'
    )


def _resolve_track_teams(run_dir: Path, tracks_path: Path, tracks, reader, args) -> dict:
    """``{track_id: kit colour}`` for every lane, from the cheapest source available.

    Three sources, in order:

    1. the ``teams`` block ``process`` writes into ``tracks.json`` — free, and
       what a current run always has;
    2. ``track_teams.json``, this function's own cache from a previous call;
    3. a classification pass over the video, cached for next time.

    Runs processed before kit stamping landed have no ``teams`` block, so the
    third path exists to keep ``--team`` working on them without re-running the
    hours-long ``process``.
    """
    doc = json.loads(tracks_path.read_text())
    stamped = doc.get("teams")
    if stamped:
        print(f"Team colours: from the `teams` block of {tracks_path.name}")
        return {int(k): v for k, v in stamped.items()}

    cache_path = run_dir / "track_teams.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text()).get("teams", {})
        print(f"Team colours: cached in {cache_path.name} ({len(cached)} lanes)")
        return {int(k): v for k, v in cached.items()}

    from soccer_vision.profiles.loader import get_kits, load_profile
    from soccer_vision.tracking.teams import TeamClassifier

    kits = get_kits(load_profile(args.profile)) if args.profile else []

    # One read serves every lane alive in that frame, so sample frames globally
    # rather than per track: 300 reads classify the whole match, where three
    # frames per lane would be thousands. Seeks dominate the runtime here.
    by_frame: dict[int, list] = {}
    for tid, samples in tracks:
        for frame_no, bbox in samples:
            by_frame.setdefault(int(frame_no), []).append((tid, bbox))
    frames = sorted(by_frame)
    stride = max(1, len(frames) // max(1, args.team_frames))
    chosen = frames[::stride][:args.team_frames]
    print(f"Team colours: classifying from {len(chosen)} frames "
          f"(no `teams` block in {tracks_path.name}; caching to {cache_path.name})")

    # min_samples=2: a lane only shows up in a couple of the sampled frames, and
    # the default of 3 would leave most of the match unclassified.
    clf = TeamClassifier(min_samples=2, keep_crops=0)
    for frame_no in chosen:
        frame = reader.read_frame(frame_no)
        if frame is None:
            continue
        mask = _not_turf_mask(frame)
        for tid, bbox in by_frame[frame_no]:
            clf.add_sample(tid, frame, bbox, mask)

    clf.fit(kits=kits or None)
    teams = {tid: clf.predict(tid) for tid, _ in tracks}
    named = {k: v for k, v in teams.items() if v}
    cache_path.write_text(json.dumps(
        {"source": "enroll --team (turf-masked jersey colour)",
         "kits": kits, "teams": {str(k): v for k, v in named.items()}}, indent=1))
    counts = {v: sum(1 for x in named.values() if x == v) for v in sorted(set(named.values()))}
    print(f"  {len(named)}/{len(teams)} lanes classified: {counts}")
    return teams


def _not_turf_mask(frame):
    """Boolean mask of the pixels that aren't pitch, for jersey colour sampling.

    The detector returns boxes, not masks, so the sample would otherwise be a
    rectangular torso patch, which on an overhead
    camera is mostly grass — the median then drags every kit toward green and
    both clusters collapse to one colour (job 37877533 named both teams "blue").
    Dropping green pixels is the poor cousin of a real mask, but it removes the
    background that actually causes the collapse.
    """
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    turf = (h >= 30) & (h <= 90) & (s >= 60) & (v >= 40)
    return ~turf

