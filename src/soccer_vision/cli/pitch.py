"""`soccer-vision pitch-region` — name which pitch is ours, once per match.

Three ways in, in order of how likely you are to use them here:

1. ``--export-frame`` writes a still with a labelled 0-1 grid on it. Read the
   corners off that image (or hand it to a vision model) and pass them back with
   ``--points``. This is the headless path — it needs no display at all.
2. ``--points`` writes the region file directly, from coordinates you already
   have. Repeat ``--frame``/``--points`` to key the polygon at several times and
   have it interpolated between them.
3. ``--interactive`` opens the frame in an OpenCV window and you click the
   corners. Only works where there is a display.

``--preview`` renders the stored region back onto a frame so you can check it
before committing an hour of GPU to it; ``--check`` additionally runs the
detector and colours which players the region keeps and drops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _resolve_frame_no(reader, args) -> int:
    if getattr(args, "at", None) is not None:
        return max(0, int(round(args.at * (reader.fps or 30.0))))
    return int(args.frame[0] if isinstance(args.frame, list) else args.frame or 0)


def _read_frame(video: Path, frame_no: int):
    from soccer_vision.io.video import VideoReader

    reader = VideoReader(video)
    frame = reader.read_frame(frame_no)
    if frame is None:
        reader.close()
        raise SystemExit(f"Could not read frame {frame_no} of {video}")
    return reader, frame


def _export_frame(args) -> None:
    import cv2

    from soccer_vision.detection.pitch_region import draw_coordinate_grid

    video = Path(args.video)
    reader, frame = _read_frame(video, 0)
    frame_no = _resolve_frame_no(reader, args)
    if frame_no:
        frame = reader.read_frame(frame_no)
    h, w = frame.shape[:2]
    fps = reader.fps or 30.0
    reader.close()

    out = Path(args.export_frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = frame if args.no_grid else draw_coordinate_grid(frame, step=args.grid_step)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"Reference frame {frame_no} ({frame_no / fps:.1f}s, {w}x{h}) -> {out.resolve()}")
    if not args.no_grid:
        print("Gridlines are normalised frame coordinates: x=0 left, x=1 right, "
              "y=0 top, y=1 bottom.")
    print("\nRead the four corners of OUR pitch off that image, then:")
    print(f"  soccer-vision pitch-region --video {video} --frame {frame_no} \\")
    print("      --points '0.05,0.45 0.95,0.42 0.98,0.99 0.02,0.99' \\")
    print(f"      --out {args.out or 'pitch_region.json'} --preview region_preview.jpg")
    print("\nPixel coordinates work too — anything outside 0-1 is read as pixels.")


def _click_polygon(frame, window: str = "our pitch"):
    """Collect polygon corners by clicking; returns pixel points or None."""
    import cv2

    from soccer_vision.detection.pitch_region import draw_region

    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print("Click the corners of our pitch (right-click undoes). "
          "ENTER accepts, ESC cancels.")
    while True:
        shown = draw_region(frame, np.array(pts, float), label=f"{len(pts)} corners") \
            if len(pts) >= 3 else frame.copy()
        for x, y in pts[: 0 if len(pts) >= 3 else len(pts)]:
            cv2.circle(shown, (x, y), 7, (0, 220, 60), -1)
        cv2.imshow(window, shown)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(window)
            return None
        if key in (13, 10):  # ENTER
            cv2.destroyWindow(window)
            return np.array(pts, dtype=float) if len(pts) >= 3 else None


def _preview(region, video: Path, frame_no: int, out_path: Path, args) -> None:
    import cv2

    from soccer_vision.detection.pitch_region import PitchRegionTracker, draw_region

    reader, frame = _read_frame(video, frame_no)
    reader.close()
    tracker = PitchRegionTracker(region, video_path=video, track_pan=args.track_pan)
    poly = tracker.polygon_for(frame_no, frame=frame)
    img = draw_region(frame, poly, label=f"frame {frame_no}")

    kept = dropped = 0
    if args.check:
        from soccer_vision.detection.field_filter import _box_foot_point
        from soccer_vision.detection.pitch_region import points_in_polygon
        from soccer_vision.detection.rfdetr import ALL_PERSON_CLASS_IDS, RFDETRSoccerDetector

        det = RFDETRSoccerDetector.from_pretrained(device=args.device)
        det.conf_threshold = args.conf_threshold
        dets = det.predict(frame)
        dets = dets[np.isin(dets.class_id, list(ALL_PERSON_CLASS_IDS))]
        feet = _box_foot_point(dets.xyxy)
        inside = points_in_polygon(feet, poly)
        for box, ok in zip(dets.xyxy.astype(int), inside):
            colour = (0, 220, 60) if ok else (60, 60, 235)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), colour, 2)
        kept, dropped = int(inside.sum()), int((~inside).sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Preview -> {out_path.resolve()}")
    if args.check:
        print(f"  Players kept (green): {kept}   dropped as another pitch/sideline "
              f"(red): {dropped}")


def run_pitch_region(args) -> None:
    from soccer_vision.detection.pitch_region import (
        PitchRegion,
        PitchRegionError,
        parse_polygon_spec,
        polygon_below_line,
    )

    video = Path(args.video) if args.video else None

    if args.export_frame:
        if video is None:
            raise SystemExit("--export-frame needs --video")
        _export_frame(args)
        return

    # Preview-only: an existing region file, no new points.
    if not args.points and not args.below and not args.interactive:
        if not args.region:
            raise SystemExit(
                "Nothing to do. Give --points (or --interactive) to write a region, "
                "--export-frame to get a gridded still to read coordinates off, or "
                "--region FILE --preview OUT.jpg to check an existing one.")
        region = PitchRegion.load(args.region)
        if not args.preview:
            print(f"{args.region}: {len(region.keyframes)} keyframe(s), "
                  f"{len(region.keyframes[0].polygon)} corners, "
                  f"track_pan={region.track_pan}, margin={region.margin}")
            for kf in region.keyframes:
                print(f"  frame {kf.frame}: "
                      + " ".join(f"{x:.3f},{y:.3f}" for x, y in kf.polygon))
            return
        if video is None:
            video = Path(region.video) if region.video else None
        if video is None:
            raise SystemExit("--preview needs --video (the region file records none)")
        _preview(region, video, args.frame[0] if args.frame else region.keyframes[0].frame,
                 Path(args.preview), args)
        return

    if video is None:
        raise SystemExit("--video is required to write a pitch region")

    reader, _ = _read_frame(video, 0)
    width, height, fps = reader.width, reader.height, reader.fps or 30.0
    reader.close()

    frames = [int(f) for f in (args.frame or [0])]

    if args.interactive:
        if not sys.stdout.isatty():
            print("Warning: --interactive needs a display; on a headless machine use "
                  "--export-frame and --points instead.", file=sys.stderr)
        polys = []
        for fn in frames:
            reader, frame = _read_frame(video, fn)
            reader.close()
            clicked = _click_polygon(frame, window=f"our pitch — frame {fn}")
            if clicked is None:
                raise SystemExit("Cancelled — no region written.")
            polys.append(clicked / np.array([float(width), float(height)]))
    else:
        # --below names one boundary line (our far touchline); the region is
        # everything on our side of it, run past the frame edges so a zoom-out
        # doesn't cut off our own players. --points names the full polygon.
        flag, specs = ("--below", args.below) if args.below else ("--points", args.points)
        if len(specs) != len(frames) and len(frames) != 1:
            raise SystemExit(
                f"--frame given {len(frames)} times but {flag} {len(specs)} times; "
                f"pair them one to one (each {flag} is the region at the preceding "
                "--frame)")
        if len(frames) == 1 and len(specs) > 1:
            raise SystemExit(f"Several {flag} need a --frame each to interpolate between")
        try:
            if args.below:
                polys = []
                for spec in specs:
                    line = parse_polygon_spec(spec, width, height, mode=args.coords,
                                              min_points=2)
                    if len(line) != 2:
                        raise PitchRegionError(
                            f"--below takes exactly 2 points on the line, got {len(line)}")
                    polys.append(polygon_below_line(line[0], line[1]))
            else:
                polys = [parse_polygon_spec(p, width, height, mode=args.coords)
                         for p in specs]
        except PitchRegionError as exc:
            raise SystemExit(f"Bad {flag}: {exc}") from exc

    base = PitchRegion.load(args.region) if (args.region and Path(args.region).exists()) else None
    try:
        if base is not None:
            region = base
            for fn, poly in zip(frames, polys):
                region = region.with_keyframe(poly, fn)
            region.track_pan = args.track_pan if args.track_pan is not None else region.track_pan
            region.margin = args.margin if args.margin is not None else region.margin
        else:
            from soccer_vision.detection.pitch_region import PitchKeyframe

            region = PitchRegion(
                keyframes=[PitchKeyframe(frame=fn, polygon=poly)
                           for fn, poly in zip(frames, polys)],
                video=str(video),
                width=width,
                height=height,
                track_pan=True if args.track_pan is None else args.track_pan,
                margin=0.0 if args.margin is None else args.margin,
                notes=args.notes,
            )
    except PitchRegionError as exc:
        raise SystemExit(str(exc)) from exc

    out = Path(args.out or args.region or "pitch_region.json")
    region.save(out)
    print(f"Pitch region -> {out.resolve()}")
    for kf in region.keyframes:
        print(f"  frame {kf.frame} ({kf.frame / fps:.1f}s): "
              + " ".join(f"{x:.3f},{y:.3f}" for x, y in kf.polygon))
    print(f"  track_pan={region.track_pan}  margin={region.margin}  "
          f"({len(region.keyframes)} keyframe(s))")

    if args.preview:
        _preview(region, video, frames[0], Path(args.preview), args)

    print("\nUse it:")
    print(f"  soccer-vision process {video} --pitch-region {out}")
