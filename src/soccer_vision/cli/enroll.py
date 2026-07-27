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
    from soccer_vision.identify.enroll import (
        boxes_from_label_studio,
        crops_from_directory,
        names_from_jerseys,
    )
    from soccer_vision.identify.gallery import (
        build_gallery,
        load_gallery,
        merge_galleries,
        save_gallery,
    )
    from soccer_vision.identify.reid import ReIDEmbedder, crop_player, embed_tracks
    from soccer_vision.io.video import VideoReader
    from soccer_vision.profiles.loader import load_profile

    run_dir = Path(args.run)
    tracks_path = run_dir / "tracks.json"
    proxy_path = run_dir / "broadcast_proxy.mp4"
    out_path = Path(args.out) if args.out else run_dir / "gallery.npz"

    if not proxy_path.exists():
        print(f"No broadcast_proxy.mp4 in {run_dir} — run `soccer-vision process` first.")
        return

    profile = load_profile(args.profile) if args.profile else None
    exclude = {int(n) for n in (args.exclude_jersey or [])}

    print("=== soccer-vision enroll ===")
    print(f"Run:    {run_dir}")

    if args.dump_crops:
        _dump_crops(run_dir, tracks_path, proxy_path, Path(args.dump_crops), args)
        return

    print("Loading re-id backbone...")
    embedder = ReIDEmbedder.from_pretrained(weights=args.weights, device=args.device)

    reader = VideoReader(proxy_path)
    try:
        if args.from_crops:
            import cv2

            labelled = crops_from_directory(args.from_crops)
            print(f"Source: labelled crop folders — {len(labelled)} crops, "
                  f"{len(set(n for _, n in labelled))} players")
            images = [(cv2.imread(str(p)), n) for p, n in labelled]
            images = [(img, n) for img, n in images if img is not None]
            embeddings = (embedder.embed([img for img, _ in images]) if images
                          else np.zeros((0, 512), dtype=np.float32))
            names = [n for _, n in images]
        elif args.from_label_studio:
            export = json.loads(Path(args.from_label_studio).read_text())
            boxes = boxes_from_label_studio(export)
            print(f"Source: Label Studio — {len(boxes)} labelled boxes")
            embeddings, names = _embed_boxes(boxes, embedder, reader, crop_player, np)
        else:
            jerseys_path = run_dir / "jerseys.json"
            if not jerseys_path.exists():
                print(f"No jerseys.json in {run_dir} — run `soccer-vision identify` "
                      "first, or enrol from annotations with --from-label-studio.")
                return
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
            per_track = embed_tracks(
                load_track_boxes(tracks_path), embedder, reader,
                max_samples_per_track=args.max_samples,
                track_ids=set(track_names),
            )
            embeddings = (np.concatenate(list(per_track.values())) if per_track
                          else np.zeros((0, 512), dtype=np.float32))
            names = [track_names[t] for t, e in per_track.items() for _ in range(len(e))]
    finally:
        reader.close()

    if len(embeddings) == 0:
        print("No crops could be embedded — nothing enrolled.")
        return

    gallery = build_gallery(embeddings, names, max_per_player=args.max_per_player)
    if args.append and out_path.exists():
        existing = load_gallery(out_path)
        gallery = merge_galleries(existing, gallery, max_per_player=args.max_per_player)
        print(f"Appended to existing gallery ({len(existing['names'])} players in it)")

    save_gallery(gallery, out_path)

    counts = Counter(gallery["names"][i] for i in gallery["label"])
    print(f"\nGallery: {len(gallery['names'])} players, {len(gallery['emb'])} exemplars")
    for name, n in counts.most_common():
        print(f"  {name:<24} {n} exemplars")
    print(f"Saved: {out_path}")
    print(f"Next: soccer-vision identify --run {run_dir} --method reid --gallery {out_path}")


def _dump_crops(run_dir: Path, tracks_path: Path, proxy_path: Path, out_dir: Path, args):
    """Write one folder of crops per track, for you to rename to player names.

    Folders are named ``track_<id>`` — plus the jersey number OCR voted, when
    there is one, as a hint while labelling. Rename a folder to the player and
    it enrols; leave the prefix and it's skipped.

    Only the ``--max-tracks`` **longest** lanes are dumped. A match fragments
    into hundreds of lanes (1,995 on the Saints U11 full match, 1,335 of them
    over 40 frames), which is far more than anyone will label by hand — and
    unnecessary, since enrolment wants a few good exemplars per player, not
    coverage. The longest lanes are also the ones that saw the player from the
    most angles, so they make the best gallery entries.
    """
    import cv2

    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.identify.reid import crop_player
    from soccer_vision.io.video import VideoReader

    hints = {}
    jerseys_path = run_dir / "jerseys.json"
    if jerseys_path.exists():
        for tid, info in json.loads(jerseys_path.read_text()).get("tracks", {}).items():
            if info.get("jersey") is not None:
                hints[int(tid)] = f"__ocr{info['jersey']}"

    tracks = [(t, s) for t, s in load_track_boxes(tracks_path).items()
              if len(s) >= args.min_track_frames]
    tracks.sort(key=lambda ts: -len(ts[1]))

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(proxy_path)

    if args.team:
        teams = _resolve_track_teams(run_dir, tracks_path, tracks, reader, args)
        wanted = args.team.strip().lower()
        kept = [(t, s) for t, s in tracks if (teams.get(t) or "").lower() == wanted]
        unknown = sum(1 for t, _ in tracks if teams.get(t) is None)
        print(f"Team filter '{wanted}': {len(kept)} of {len(tracks)} lanes "
              f"({unknown} unclassified, dropped)")
        if not kept:
            print(f"  No lanes on team '{wanted}'. Kit colours seen: "
                  f"{sorted({v for v in teams.values() if v})}")
            reader.close()
            return
        tracks = kept

    print(f"Dumping the {min(args.max_tracks, len(tracks))} longest of "
          f"{len(tracks)} tracks over {args.min_track_frames} frames...")
    written = 0
    # Context tiles are built in the same pass and kept in memory only. They must
    # never land in a player folder: they show neighbouring players too, so
    # enrolling one would bank someone else's appearance under this player's name.
    context: dict[str, list] = {}
    # The sheet wants a handful of large tiles; the gallery wants many crops.
    # Tying both to --max-samples forces a choice between a readable sheet and a
    # well-covered player, so the sheet takes an evenly-spaced subset.
    sheet_samples = max(1, getattr(args, "sheet_samples", 6))
    try:
        for tid, samples in tracks[:args.max_tracks]:
            step = max(1, len(samples) // args.max_samples)
            name = f"track_{tid:04d}{hints.get(tid, '')}"
            folder = out_dir / name
            kept = samples[::step][:args.max_samples]
            tile_stride = max(1, len(kept) // sheet_samples)
            for i, (frame_no, bbox) in enumerate(kept):
                frame = reader.read_frame(int(frame_no))
                if frame is None:
                    continue
                crop = crop_player(frame, bbox)
                if crop is None:
                    continue
                folder.mkdir(exist_ok=True)
                cv2.imwrite(str(folder / f"{int(frame_no):06d}.jpg"), crop)
                written += 1
                if i % tile_stride == 0 and len(context.get(name, ())) < sheet_samples:
                    context.setdefault(name, []).append(
                        _context_tile(frame, bbox, args.context_pad, args.context_min)
                    )
    finally:
        reader.close()

    folders = sorted(p for p in out_dir.iterdir() if p.is_dir())
    sheets = _write_index_sheets(out_dir, folders, context,
                                 tile_h=getattr(args, "sheet_tile_height", TILE_H))
    print(f"\nWrote {written} crops across {len(folders)} track folders → {out_dir}")
    print(f"Review sheets ({len(sheets)}): {sheets[0].parent}/index_*.jpg")
    print("Next: rename the folders you recognise to player names "
          "(delete the rest), then:")
    print(f"  soccer-vision enroll --run {run_dir} --from-crops {out_dir} "
          f"--out {run_dir / 'gallery.npz'}")


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

    ``process`` masks the colour sample with SAM3's per-player segmentation; a
    run without one leaves the rectangular torso patch, which on an overhead
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


TILE_H = 180          # crops are ~50px tall on an overhead camera; upscale to see them
TRACKS_PER_SHEET = 12
SHEET_MAX_PIXELS = 40_000_000   # keep a sheet openable in an ordinary image viewer
LABEL_W = 260


def _context_tile(frame, bbox, pad: float, min_px: int):
    """A wide view around the player, with their box outlined.

    The tight crop that feeds the model is useless for *recognising* who someone
    is — an overhead camera renders a player in about 50x21 px with no
    surroundings. Pulling back to a fixed window and marking the target lets you
    use position on the pitch, who they're next to, and which way play is going,
    which is how you actually tell youth players apart at this resolution.
    """
    import cv2
    import numpy as np

    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) * (0.5 + pad), min_px / 2)
    half_h = max((y2 - y1) * (0.5 + pad), min_px / 2)

    H, W = frame.shape[:2]
    wx1, wy1 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    wx2, wy2 = int(min(W, cx + half_w)), int(min(H, cy + half_h))
    tile = frame[wy1:wy2, wx1:wx2].copy()
    if tile.size == 0:
        return np.zeros((min_px, min_px, 3), dtype=np.uint8)

    # The marker has to scale with the window, not sit at a fixed 2px: pulled out
    # to a full frame, a hairline box around a 29px-tall child is invisible, and
    # a view you can't locate the player in is no more use than no view at all.
    th, tw = tile.shape[:2]
    thick = max(2, int(round(min(th, tw) / 260)))
    px1, py1 = int(x1) - wx1, int(y1) - wy1
    px2, py2 = int(x2) - wx1, int(y2) - wy1
    cv2.rectangle(tile, (px1, py1), (px2, py2), (0, 255, 255), thick)

    # Crosshair from the tile edges, stopping short of the box so it points at
    # the player without covering them — findable at a glance on a wide view.
    mx, my = (px1 + px2) // 2, (py1 + py2) // 2
    gap_x = int((px2 - px1) * 3.0 + thick * 8)
    gap_y = int((py2 - py1) * 1.8 + thick * 8)
    for a, b in (((0, my), (mx - gap_x, my)), ((tw, my), (mx + gap_x, my)),
                 ((mx, 0), (mx, my - gap_y)), ((mx, th), (mx, my + gap_y))):
        cv2.line(tile, a, b, (0, 255, 255), thick, cv2.LINE_AA)
    return tile


def _write_index_sheets(out_dir: Path, folders: list[Path], context: dict,
                        tile_h: int = TILE_H) -> list[Path]:
    """One image per dozen tracks: a labelled strip of context views per track.

    Without this the folders are unlabellable — the tight crops are ~50x21 px,
    which no one can put a name to in a file browser.

    ``tile_h`` has to grow with ``--context-pad``: every tile is scaled to this
    height, so a wider window rendered at the same height just shrinks the player
    back to where we started. Pulling the view out without raising it cancels
    itself out.
    """
    import cv2
    import numpy as np

    strips = []
    for folder in folders:
        tiles = []
        for c in context.get(folder.name, []):
            if c is None or c.size == 0:
                continue
            scale = tile_h / c.shape[0]
            tiles.append(cv2.resize(c, (max(1, int(c.shape[1] * scale)), tile_h)))
        if not tiles:
            continue
        label = np.zeros((tile_h, LABEL_W, 3), dtype=np.uint8)
        cv2.putText(label, folder.name[:22], (8, tile_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        strips.append(np.hstack([label] + tiles))

    if not strips:
        return []

    width = max(s.shape[1] for s in strips)
    padded = [np.pad(s, ((0, 2), (0, width - s.shape[1]), (0, 0))) for s in strips]

    # A dozen full-resolution rows makes a 100+ megapixel JPEG that image viewers
    # choke on, so the row count follows the pixel budget rather than a constant.
    per_sheet = max(1, min(TRACKS_PER_SHEET, int(SHEET_MAX_PIXELS // (width * (tile_h + 2)))))

    paths = []
    for i in range(0, len(padded), per_sheet):
        sheet = np.vstack(padded[i:i + per_sheet])
        path = out_dir / f"index_{i // per_sheet + 1:03d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(path)
    return paths


def _embed_boxes(boxes, embedder, reader, crop_player, np):
    """Embed ``(frame, bbox, name)`` triples, dropping boxes too small to crop."""
    crops, names = [], []
    for frame_no, bbox, name in boxes:
        frame = reader.read_frame(int(frame_no))
        if frame is None:
            continue
        crop = crop_player(frame, bbox)
        if crop is not None:
            crops.append(crop)
            names.append(name)
    return (embedder.embed(crops) if crops else np.zeros((0, 512))), names
