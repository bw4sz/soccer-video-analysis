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
    print(f"Dumping the {min(args.max_tracks, len(tracks))} longest of "
          f"{len(tracks)} tracks over {args.min_track_frames} frames...")

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(proxy_path)
    written = 0
    # Context tiles are built in the same pass and kept in memory only. They must
    # never land in a player folder: they show neighbouring players too, so
    # enrolling one would bank someone else's appearance under this player's name.
    context: dict[str, list] = {}
    try:
        for tid, samples in tracks[:args.max_tracks]:
            step = max(1, len(samples) // args.max_samples)
            name = f"track_{tid:04d}{hints.get(tid, '')}"
            folder = out_dir / name
            for frame_no, bbox in samples[::step][:args.max_samples]:
                frame = reader.read_frame(int(frame_no))
                if frame is None:
                    continue
                crop = crop_player(frame, bbox)
                if crop is None:
                    continue
                folder.mkdir(exist_ok=True)
                cv2.imwrite(str(folder / f"{int(frame_no):06d}.jpg"), crop)
                written += 1
                context.setdefault(name, []).append(
                    _context_tile(frame, bbox, args.context_pad, args.context_min)
                )
    finally:
        reader.close()

    folders = sorted(p for p in out_dir.iterdir() if p.is_dir())
    sheets = _write_index_sheets(out_dir, folders, context)
    print(f"\nWrote {written} crops across {len(folders)} track folders → {out_dir}")
    print(f"Review sheets ({len(sheets)}): {sheets[0].parent}/index_*.jpg")
    print("Next: rename the folders you recognise to player names "
          "(delete the rest), then:")
    print(f"  soccer-vision enroll --run {run_dir} --from-crops {out_dir} "
          f"--out {run_dir / 'gallery.npz'}")


TILE_H = 180          # crops are ~50px tall on an overhead camera; upscale to see them
TRACKS_PER_SHEET = 12
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

    cv2.rectangle(tile, (int(x1) - wx1, int(y1) - wy1), (int(x2) - wx1, int(y2) - wy1),
                  (0, 255, 255), 2)
    return tile


def _write_index_sheets(out_dir: Path, folders: list[Path], context: dict) -> list[Path]:
    """One image per dozen tracks: a labelled strip of context views per track.

    Without this the folders are unlabellable — the tight crops are ~50x21 px,
    which no one can put a name to in a file browser.
    """
    import cv2
    import numpy as np

    strips = []
    for folder in folders:
        tiles = []
        for c in context.get(folder.name, []):
            if c is None or c.size == 0:
                continue
            scale = TILE_H / c.shape[0]
            tiles.append(cv2.resize(c, (max(1, int(c.shape[1] * scale)), TILE_H)))
        if not tiles:
            continue
        label = np.zeros((TILE_H, LABEL_W, 3), dtype=np.uint8)
        cv2.putText(label, folder.name[:22], (8, TILE_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        strips.append(np.hstack([label] + tiles))

    if not strips:
        return []

    width = max(s.shape[1] for s in strips)
    padded = [np.pad(s, ((0, 2), (0, width - s.shape[1]), (0, 0))) for s in strips]

    paths = []
    for i in range(0, len(padded), TRACKS_PER_SHEET):
        sheet = np.vstack(padded[i:i + TRACKS_PER_SHEET])
        path = out_dir / f"index_{i // TRACKS_PER_SHEET + 1:03d}.jpg"
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
