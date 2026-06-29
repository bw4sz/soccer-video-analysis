"""
extract_clips.py

Pass 2/3 of the pipeline: extract video clips around candidate timestamps,
with an option to concatenate all clips into a single highlight reel.

Reads candidates from a JSON file produced by detect_actions.py, or accepts
timestamps directly on the command line.

Usage:
    # From candidates JSON (produced by detect_actions.py or Claude verification)
    python extract_clips.py --video match.mp4 --candidates candidates.json

    # Manual timestamps
    python extract_clips.py --video match.mp4 --timestamps 2369 2469 3203

    # Concatenate into a single highlight reel
    python extract_clips.py --video match.mp4 --candidates candidates.json --concat

Options:
    --video         Path to input video
    --candidates    JSON file from detect_actions.py (or Claude-verified JSON)
    --timestamps    Space-separated list of timestamps in seconds
    --pre           Seconds before each timestamp to include (default: 5)
    --post          Seconds after each timestamp to include (default: 30)
    --out-dir       Output directory (default: clips/)
    --prefix        Filename prefix for clips (default: clip)
    --concat        Concatenate all clips into a single file
    --concat-out    Output filename for concatenated reel (default: highlight_reel.mp4)
    --no-reencode   Use stream copy (faster, no quality loss, may have seek imprecision)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}h{m:02d}m{s:05.2f}s"


def run(cmd, check=True):
    print("  $", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(f"Command failed (exit {result.returncode})")
    return result


def extract_clip(video, start_s, duration_s, out_path, no_reencode=False):
    """Extract a clip using ffmpeg. Fast seek + optional stream copy."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0, start_s)),
        "-i", video,
        "-t", str(duration_s),
    ]
    if no_reencode:
        cmd += ["-c", "copy"]
    else:
        # Re-encode for accurate cuts and concat compatibility
        cmd += [
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac",
        ]
    cmd.append(str(out_path))
    run(cmd)


def concat_clips(clip_paths, out_path):
    """Concatenate clips using ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_file = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        str(out_path),
    ]
    run(cmd)
    os.unlink(list_file)
    print(f"\nHighlight reel saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_timestamps(args):
    """Return list of (timestamp_s, label) tuples."""
    if args.candidates:
        # Load from track.py candidates.json
        with open(args.candidates) as f:
            data = json.load(f)
        frame_set = set(args.frames) if args.frames else None
        results = []
        for entry in data["candidates"]:
            if frame_set is None or entry["frame"] in frame_set:
                results.append((entry["timestamp_s"], entry["timestamp_hms"]))
        return results
    elif args.index:
        # Load from detect_actions.py index.json and filter to requested frames
        with open(args.index) as f:
            data = json.load(f)
        frame_set = set(args.frames) if args.frames else None
        results = []
        for entry in data["frames"]:
            if frame_set is None or entry["frame"] in frame_set:
                results.append((entry["timestamp_s"], entry["timestamp_hms"]))
        return results
    elif args.timestamps:
        return [(float(t), ts_to_hms(float(t))) for t in args.timestamps]
    else:
        sys.exit("Provide --candidates, --index + --frames, or --timestamps")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = load_timestamps(args)
    if not timestamps:
        sys.exit("No timestamps found.")

    duration = args.pre + args.post
    clip_paths = []

    for i, (ts, label) in enumerate(timestamps, 1):
        start = max(0.0, ts - args.pre)
        safe_label = label.replace(":", "").replace(".", "")
        out_path = out_dir / f"{args.prefix}_{i:02d}_{safe_label}.mp4"

        print(f"\n[{i}/{len(timestamps)}] Extracting clip at t={ts:.1f}s "
              f"({start:.1f}s → {start + duration:.1f}s) → {out_path.name}")
        extract_clip(args.video, start, duration, out_path, no_reencode=args.no_reencode)
        clip_paths.append(out_path)

    print(f"\n{len(clip_paths)} clip(s) saved to {out_dir}/")

    if args.concat and len(clip_paths) > 1:
        concat_out = out_dir / args.concat_out
        print(f"\nConcatenating {len(clip_paths)} clips → {concat_out.name}")
        if args.no_reencode:
            print("  WARNING: --no-reencode clips may not concat cleanly. "
                  "Re-encoding for concat...")
            # Re-extract without stream copy before concat
            reencoded = []
            for p in clip_paths:
                rp = p.with_suffix(".reenc.mp4")
                run(["ffmpeg", "-y", "-i", str(p),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-c:a", "aac", str(rp)])
                reencoded.append(rp)
            concat_clips(reencoded, concat_out)
            for rp in reencoded:
                rp.unlink()
        else:
            concat_clips(clip_paths, concat_out)


def parse_args():
    p = argparse.ArgumentParser(description="Extract and optionally concatenate video clips.")
    p.add_argument("--video", required=True)
    p.add_argument("--candidates", help="candidates.json from track.py (all candidates used)")
    p.add_argument("--index", help="index.json from detect_actions.py")
    p.add_argument("--frames", nargs="+", type=int,
                   help="Frame numbers to extract (from sheet labels, e.g. 14500 33500)")
    p.add_argument("--timestamps", nargs="+", type=float,
                   help="Raw timestamps in seconds (alternative to --index/--frames)")
    p.add_argument("--pre", type=float, default=5,
                   help="Seconds before timestamp (default 5)")
    p.add_argument("--post", type=float, default=30,
                   help="Seconds after timestamp (default 30)")
    p.add_argument("--out-dir", default="clips")
    p.add_argument("--prefix", default="clip")
    p.add_argument("--concat", action="store_true")
    p.add_argument("--concat-out", default="highlight_reel.mp4")
    p.add_argument("--no-reencode", action="store_true",
                   help="Stream copy (faster but imprecise cuts, not concat-safe)")
    return p.parse_args()


if __name__ == "__main__":
    main()
