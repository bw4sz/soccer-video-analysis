"""Smoke test: run SoccerChat on a few 10s clips cut straight from a match.

Isolates the SoccerChat integration (model load + ms-swift inference in
``soccer_vision.verify.soccerchat``) from the full pipeline, so we can see
whether a pro-trained soccer VLM makes sense of youth/Veo footage without first
running the heavy ``process`` pipeline.

Cuts ``--n`` evenly spaced 10s clips, then prints/saves each clip's SoccerChat
class, mapped soccer-vision label, confidence, and caption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Match video to sample clips from")
    ap.add_argument("--n", type=int, default=6, help="Number of clips to sample")
    ap.add_argument("--clip-s", type=float, default=10.0, help="Clip length (SoccerChat uses 10s)")
    ap.add_argument("--out", default="soccerchat_smoke", help="Output dir")
    ap.add_argument("--timestamps", type=float, nargs="*",
                    help="Explicit clip start times (s); overrides --n even spacing")
    ap.add_argument("--margin-s", type=float, default=60.0,
                    help="Skip this many seconds at each end when auto-spacing")
    args = ap.parse_args()

    from soccer_vision.io.video import VideoReader, ffmpeg_extract_clip
    from soccer_vision.verify.soccerchat import ADAPTER_ID, SoccerChatModel, is_available

    if not is_available():
        raise SystemExit("ms-swift not installed — run with the 'soccerchat' extra.")

    out = Path(args.out)
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(args.video)
    duration = reader.duration_s
    reader.close()
    print(f"Video: {args.video}  ({duration / 60:.1f} min)")

    if args.timestamps:
        starts = list(args.timestamps)
    elif args.n <= 1:
        starts = [max(0.0, duration / 2)]
    else:
        span = max(0.0, duration - 2 * args.margin_s - args.clip_s)
        starts = [args.margin_s + span * i / (args.n - 1) for i in range(args.n)]

    print(f"Loading SoccerChat ({ADAPTER_ID}) — first inference loads ~16GB of weights...\n")
    model = SoccerChatModel()

    results = []
    for i, start in enumerate(starts, 1):
        clip = clips_dir / f"smoke_{i:02d}_{int(start)}s.mp4"
        ffmpeg_extract_clip(args.video, start, args.clip_s, clip, reencode=True)
        cls = model.classify(clip)
        caption = model.describe(clip)
        row = {
            "i": i,
            "start_s": round(start, 1),
            "sc_class": cls["sc_class"],
            "label": cls["label"],
            "confidence": cls["confidence"],
            "raw": cls["raw"],
            "caption": caption,
            "clip": str(clip),
        }
        results.append(row)
        print(f"[{i}/{len(starts)}] t={start:7.1f}s  "
              f"class={str(cls['sc_class']):<18} label={str(cls['label']):<12} "
              f"conf={cls['confidence']}")
        print(f"        caption: {caption}\n")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    labels = [r["label"] for r in results if r["label"]]
    print(f"Saved {out / 'results.json'}")
    print(f"Mapped labels: {labels or '(none mapped)'}")


if __name__ == "__main__":
    main()
