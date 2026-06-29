"""soccer-vision CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="soccer-vision",
        description="Open-source soccer video analysis toolkit",
    )
    subparsers = parser.add_subparsers(dest="command")

    # process
    p_process = subparsers.add_parser("process", help="Run full pipeline on a match video")
    p_process.add_argument("video", help="Path to input video file")
    p_process.add_argument("--config", help="Pipeline config YAML")
    p_process.add_argument("--profile", help="Project profile YAML")
    p_process.add_argument("--out-dir", default="runs", help="Output base directory")
    p_process.add_argument("--match-id", help="Match identifier (auto-generated if omitted)")
    p_process.add_argument("--device", default=None, help="PyTorch device: cpu / cuda / mps")

    # broadcast
    p_broadcast = subparsers.add_parser("broadcast", help="Generate broadcast proxy only")
    p_broadcast.add_argument("video", help="Path to input video file")
    p_broadcast.add_argument("--out", help="Output directory")
    p_broadcast.add_argument("--config", help="Broadcast config YAML")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract clips from a processed run")
    p_extract.add_argument("--run", required=True, help="Run directory path")
    p_extract.add_argument("--events", nargs="+", help="Event labels to extract")
    p_extract.add_argument("--pre", type=float, default=5.0)
    p_extract.add_argument("--post", type=float, default=30.0)

    # reel
    p_reel = subparsers.add_parser("reel", help="Build highlight reel")
    p_reel.add_argument("--run", required=True)
    p_reel.add_argument("--event", help="Filter by event label")
    p_reel.add_argument("--player", help="Filter by player name")
    p_reel.add_argument("--out", default="highlight_reel.mp4")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "process":
        from soccer_vision.cli.process import run_pipeline
        run_pipeline(args)
    elif args.command == "broadcast":
        from soccer_vision.cli.process import run_broadcast_only
        run_broadcast_only(args)
    elif args.command == "extract":
        from soccer_vision.cli.extract import run_extract
        run_extract(args)
    elif args.command == "reel":
        from soccer_vision.cli.extract import run_reel
        run_reel(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
