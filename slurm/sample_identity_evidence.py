"""Show the evidence behind one lane's identity — the crops, the digits, both scores.

The identity decisions in this pipeline are currently argued from counts, and
counts cannot settle the question that matters: when OCR reads `1` on a lane
re-id called Gia, is that a real shirt number or PARSeq hallucinating on a blur?
That is a question about pixels, so this renders them.

Per lane it draws:

- the **whole frame** at the lane's midpoint with the box marked, because a
  50x100 px crop in isolation is unnameable — position, neighbours and kit are
  what a person actually reads a player from;
- a strip of **player crops** across the lane, so you can see whether it is one
  person throughout;
- under each, the **torso window PARSeq actually sees** (``crop_number_region``),
  upscaled nearest-neighbour so no interpolation invents a digit that wasn't
  there, captioned with what the recognizer read on that exact crop.

Cases are chosen to put the live decisions side by side: lanes the jersey reads
disproved, lanes they corroborated, the high-similarity lanes they could not
rule on, and — separately — the lanes OCR *named on its own*, which is the path
that produced 982 lanes reading `1` on a roster where nobody wears 1.

    python slurm/sample_identity_evidence.py runs/saints-u14g-full \
        --out runs/saints-u14g-full/identity_evidence

Reads the video once in frame order (`VideoReader.read_frames`), so adding lanes
costs little; the per-crop PARSeq forwards run on CPU by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from soccer_vision.clips.halo import load_track_boxes
from soccer_vision.identify.jersey_ocr import (
    JerseyNumberRecognizer,
    crop_number_region,
    is_legible,
)
from soccer_vision.io.video import VideoReader

TILE_H = 210          # player crop height in the strip
STRIP_H = 84          # number-region height under it
CAPTION_H = 36
MIN_TILE_W = 124
HEADER_H = 118
CONTEXT_W = 900
PAD = 12
BG = (18, 18, 20)
FG = (238, 238, 238)
DIM = (150, 150, 155)
GOOD = (120, 220, 140)
BAD = (255, 120, 110)


def draw_text(d: ImageDraw.ImageDraw, xy, text, fill=FG, size=15):
    """Text with whatever font PIL can find; size is a hint, layout is fixed."""
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf", size)
    except Exception:
        font = None
    d.text(xy, text, fill=fill, font=font)


def pick_cases(after: dict, before: dict, teams: dict, per_case: int) -> list[tuple[str, int]]:
    """A few lanes from each decision the pipeline is currently making."""
    conflict, agree, unruled, ocr_one, ocr_roster = [], [], [], [], []
    for tid, r in after.items():
        sim = r.get("similarity") or 0.0
        if r.get("conflict"):
            conflict.append((-sim, int(tid)))
        elif r.get("crosscheck") == "agree":
            agree.append((-sim, int(tid)))
        elif r.get("source") == "reid" and r.get("crosscheck") == "no_evidence":
            unruled.append((-sim, int(tid)))
        elif r.get("source") == "ocr":
            key = (-(r.get("n_obs") or 0), int(tid))
            (ocr_one if r.get("jersey") == 1 else ocr_roster).append(key)

    out = []
    for label, pool in (("conflict", conflict), ("agree", agree),
                        ("reid-unruled", unruled), ("ocr-named-1", ocr_one),
                        ("ocr-named", ocr_roster)):
        for _, tid in sorted(pool)[:per_case]:
            out.append((label, tid))
    return out


def lane_frames(samples, n: int) -> list[int]:
    """``n`` frames spread evenly across the lane (its whole life, not its start)."""
    if len(samples) <= n:
        return [int(f) for f, _ in samples]
    idx = np.linspace(0, len(samples) - 1, n).round().astype(int)
    return [int(samples[i][0]) for i in idx]


def choose_tiles(tiles: list[dict], n: int) -> list[dict]:
    """The frames that *drove* the decision, plus enough spread to see the lane.

    An even sample mostly shows the illegible majority and misses the handful of
    confident reads a veto actually rests on — which is the opposite of what
    someone judging the veto needs to look at. So the best-read frames come
    first, then an even spread fills the rest.
    """
    if len(tiles) <= n:
        return tiles
    by_conf = sorted(tiles, key=lambda t: -t["conf"])
    keep = {id(t) for t in by_conf[:3] if t["conf"] > 0}
    spread = np.linspace(0, len(tiles) - 1, n).round().astype(int)
    for i in spread:
        if len(keep) >= n:
            break
        keep.add(id(tiles[i]))
    for t in tiles:                       # top up if the spread hit duplicates
        if len(keep) >= n:
            break
        keep.add(id(t))
    return [t for t in tiles if id(t) in keep][:n]


def render(lane, out_path: Path):
    """One sheet: header, whole-frame context, then the crop/digit strip."""
    tiles = lane["tiles"]
    strip_w = sum(t["player"].width + PAD for t in tiles) + PAD
    width = max(CONTEXT_W + 2 * PAD, strip_w, 900)
    ctx_h = lane["context"].height if lane["context"] else 0
    height = HEADER_H + ctx_h + PAD + TILE_H + STRIP_H + CAPTION_H + 3 * PAD

    sheet = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(sheet)

    for i, (line, colour, size) in enumerate(lane["header"]):
        draw_text(d, (PAD, PAD + i * 22), line, fill=colour, size=size)

    y = HEADER_H
    if lane["context"]:
        sheet.paste(lane["context"], (PAD, y))
        y += ctx_h + PAD

    x = PAD
    for t in tiles:
        sheet.paste(t["player"], (x, y))
        sheet.paste(t["strip"], (x, y + TILE_H + 4))
        cy = y + TILE_H + STRIP_H + 10
        draw_text(d, (x, cy), t["caption"], fill=t["colour"], size=13)
        draw_text(d, (x, cy + 15), t["sub"], fill=DIM, size=12)
        x += t["player"].width + PAD

    sheet.save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--out", help="Output directory (default: <run>/identity_evidence)")
    ap.add_argument("--tracks", nargs="+", type=int, help="Explicit lane ids instead of samples")
    ap.add_argument("--per-case", type=int, default=2)
    ap.add_argument("--samples", type=int, default=7, help="Crops shown per lane (default: 7)")
    ap.add_argument("--scan", type=int, default=24,
                    help="Frames read per lane before choosing which to show (default: 24)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run = Path(args.run)
    out_dir = Path(args.out) if args.out else run / "identity_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    after = json.loads((run / "jerseys.json").read_text())["tracks"]
    before_p = run / "jerseys.reid-only.json"
    before = json.loads(before_p.read_text())["tracks"] if before_p.exists() else {}
    tracks_doc = json.loads((run / "tracks.json").read_text())
    teams = tracks_doc.get("teams", {})
    boxes = load_track_boxes(run / "tracks.json")
    fps = tracks_doc.get("fps") or 30.0

    cases = ([("requested", t) for t in args.tracks] if args.tracks
             else pick_cases(after, before, teams, args.per_case))
    print(f"{len(cases)} lanes: " + ", ".join(f"{c}#{t}" for c, t in cases))

    plan: dict[int, list] = {}
    context_frame: dict[int, int] = {}
    for _, tid in cases:
        scanned = lane_frames(boxes.get(tid, []), args.scan)
        for f in scanned:
            plan.setdefault(f, []).append(tid)
        if scanned:
            context_frame[tid] = scanned[len(scanned) // 2]

    print("Loading recognizer...")
    rec = JerseyNumberRecognizer.from_pretrained(device=args.device)

    lanes: dict[int, dict] = {tid: {"tiles": [], "context": None} for _, tid in cases}
    box_at = {tid: dict((int(f), b) for f, b in boxes.get(tid, [])) for _, tid in cases}

    print(f"Reading {len(plan)} frames...")
    reader = VideoReader(run / "broadcast_proxy.mp4")
    try:
        for frame_no, frame in reader.read_frames(plan.keys()):
            for tid in plan[frame_no]:
                bbox = box_at[tid].get(frame_no)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue

                region = crop_number_region(frame, bbox)
                legible = is_legible(region)
                read = rec.predict(region) if legible else None

                player = Image.fromarray(crop[:, :, ::-1])
                w = max(40, int(player.width * TILE_H / max(player.height, 1)))
                player = player.resize((min(w, 200), TILE_H), Image.LANCZOS)
                if player.width < MIN_TILE_W:  # keep captions from colliding
                    canvas = Image.new("RGB", (MIN_TILE_W, TILE_H), BG)
                    canvas.paste(player, ((MIN_TILE_W - player.width) // 2, 0))
                    player = canvas

                if region is not None and region.size:
                    strip = Image.fromarray(region[:, :, ::-1]).resize(
                        (player.width, STRIP_H), Image.NEAREST)  # honest pixels
                else:
                    strip = Image.new("RGB", (player.width, STRIP_H), (40, 40, 44))

                if read is not None and read.number is not None:
                    caption = f"#{read.number} @{read.confidence:.2f}"
                    strong = read.confidence >= 0.7
                    colour = GOOD if strong else BAD
                    sub = "counts to veto" if strong else "under floor"
                    number, conf = read.number, read.confidence
                elif legible:
                    caption, sub, colour = "no digits", "model read none", DIM
                    number, conf = None, 0.0
                else:
                    caption, sub, colour = "illegible", "gated out", DIM
                    number, conf = None, 0.0

                lanes[tid]["tiles"].append(
                    {"player": player, "strip": strip, "caption": caption,
                     "sub": sub, "colour": colour, "frame": frame_no,
                     "number": number, "conf": conf, "legible": legible})

                if context_frame.get(tid) == frame_no:
                    ctx = frame.copy()
                    import cv2
                    cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    ctx = Image.fromarray(ctx[:, :, ::-1])
                    ctx = ctx.resize(
                        (CONTEXT_W, int(ctx.height * CONTEXT_W / ctx.width)),
                        Image.LANCZOS)
                    lanes[tid]["context"] = ctx
    finally:
        reader.close()

    written = []
    for case, tid in cases:
        r = after.get(str(tid), {})
        lane = lanes[tid]
        scanned = lane["tiles"]
        best: dict[int, float] = {}
        counts: dict[int, int] = {}
        for t in scanned:
            if t["number"] is not None:
                counts[t["number"]] = counts.get(t["number"], 0) + 1
                best[t["number"]] = max(best.get(t["number"], 0.0), t["conf"])
        hist = " · ".join(
            f"#{n} x{c} (best {best[n]:.2f})"
            for n, c in sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        ) or "nothing read"
        illegible = sum(1 for t in scanned if not t["legible"])
        hist_line = (f"reads over {len(scanned)} scanned frames: {hist}"
                     f"   |   {illegible} illegible, "
                     f"{sum(1 for t in scanned if t['conf'] >= 0.7)} at/over the 0.70 floor")

        lane["tiles"] = choose_tiles(scanned, args.samples)
        for t in lane["tiles"]:
            t["sub"] = f"f{t['frame']} {t['sub']}"
        samples = boxes.get(tid, [])
        span = ((samples[-1][0] - samples[0][0]) / fps) if len(samples) > 1 else 0.0

        conflict = r.get("conflict")
        if conflict:
            verdict = (f"VETOED — re-id said {conflict['reid_name']} "
                       f"(#{conflict['reid_jersey']}), {conflict['n_obs']} strong reads "
                       f"say #{conflict['ocr_jersey']} @{conflict['ocr_confidence']}")
            colour = BAD
        elif r.get("crosscheck") == "agree":
            verdict = f"CORROBORATED — reads agree with #{r.get('jersey')}"
            colour = GOOD
        elif r.get("source") == "ocr":
            verdict = (f"NAMED BY OCR ALONE — vote #{r.get('jersey')} "
                       f"conf {r.get('confidence')} over {r.get('n_obs')} reads "
                       f"(no confidence floor applied on this path)")
            colour = FG
        elif r.get("source") == "reid":
            verdict = "re-id name stands — no legible number anywhere in the lane"
            colour = FG
        else:
            verdict = "unnamed"
            colour = DIM

        # What re-id said *before* OCR touched anything — a lane OCR named on its
        # own must not read as though re-id had proposed that name.
        b = before.get(str(tid), {})
        reid_name = (b.get("name")
                     or (r.get("name") if r.get("source") == "reid" else None)
                     or (r.get("conflict") or {}).get("reid_name")
                     or "(abstained)")
        lane["header"] = [
            (f"[{case}]  track {tid}   kit: {teams.get(str(tid), '?')}   "
             f"{len(samples)} detections over {span:.1f}s", DIM, 16),
            (f"re-id: {reid_name}   similarity {r.get('similarity')}", FG, 16),
            (hist_line, DIM, 14),
            (verdict, colour, 16),
        ]
        path = out_dir / f"{case}_track{tid}.png"
        render(lane, path)
        written.append(path)
        print(f"  {path}")

    print(f"\n{len(written)} sheets in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
