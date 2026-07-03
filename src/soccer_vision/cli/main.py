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
    p_process.add_argument(
        "--action-engine", nargs="+", metavar="ENGINE",
        help="Action-detection engine(s) to run: rules (default) / learned / vlm. "
             "Overrides the config; engines without a runtime/checkpoint are skipped.",
    )

    # broadcast
    p_broadcast = subparsers.add_parser("broadcast", help="Generate broadcast proxy only")
    p_broadcast.add_argument("video", help="Path to input video file")
    p_broadcast.add_argument("--out", help="Output directory")
    p_broadcast.add_argument("--config", help="Broadcast config YAML")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract clips from a processed run")
    p_extract.add_argument("--run", required=True, help="Run directory path")
    p_extract.add_argument("--events", nargs="+", help="Event labels to extract")
    p_extract.add_argument("--team", help="Filter by team colour (e.g. blue)")
    p_extract.add_argument("--track", type=int, help="Filter by player track id")
    p_extract.add_argument("--pre", type=float, default=5.0)
    p_extract.add_argument("--post", type=float, default=30.0)

    # reel
    p_reel = subparsers.add_parser("reel", help="Build highlight reel")
    p_reel.add_argument("--run", required=True)
    p_reel.add_argument("--event", help="Filter by event label")
    p_reel.add_argument("--team", help="Filter by team colour (e.g. blue)")
    p_reel.add_argument("--track", type=int, help="Filter by player track id")
    p_reel.add_argument("--player", help="Filter by player name (needs roster jersey mapping)")
    p_reel.add_argument("--out", default="highlight_reel.mp4")

    # trim-empty
    p_trim = subparsers.add_parser(
        "trim-empty",
        help="Cut dead time (ball offscreen / not moving) into a new clip",
    )
    p_trim.add_argument("video", help="Path to input video file")
    p_trim.add_argument("--track", help="Ball-track JSON (built from detector if omitted)")
    p_trim.add_argument("--out", help="Output video path (default: <video>.trimmed.mp4)")
    p_trim.add_argument("--edl", help="Edit-decision-list JSON path (default: <video>.trim.json)")
    p_trim.add_argument("--save-track", help="Where to save an auto-built ball track")
    p_trim.add_argument("--sample-fps", type=float, default=5.0,
                        help="Sample rate when building a track (default: 5)")
    p_trim.add_argument("--min-dead", type=float, default=5.0,
                        help="Min seconds of dead time before a span is cut (default: 5)")
    p_trim.add_argument("--stationary-px", type=float, default=40.0,
                        help="Max pixel drift to count as 'not moving' (default: 40)")
    p_trim.add_argument("--pad", type=float, default=0.5,
                        help="Seconds of context kept around each cut (default: 0.5)")
    p_trim.add_argument("--copy", action="store_true",
                        help="Stream-copy segments instead of re-encoding (faster, less precise)")
    p_trim.add_argument("--dry-run", action="store_true",
                        help="Write the edit-decision list only; render no video")
    p_trim.add_argument("--device", default=None, help="PyTorch device for the detector")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify events with Claude")
    p_verify.add_argument("--run", required=True, help="Run directory path")
    p_verify.add_argument("--profile", help="Project profile YAML")

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask Claude about a processed match")
    p_ask.add_argument("question", help="Natural language question")
    p_ask.add_argument("--run", required=True, help="Run directory path")
    p_ask.add_argument("--profile", help="Project profile YAML")

    # describe (SoccerChat local VLM over event clips)
    p_describe = subparsers.add_parser(
        "describe", help="Caption/verify a run's event clips with SoccerChat (local VLM)"
    )
    p_describe.add_argument("--run", required=True, help="Run directory path")
    p_describe.add_argument("--profile", help="Project profile YAML (optional)")
    p_describe.add_argument("--adapter", default="SimulaMet/SoccerChat-qwen2-vl-7b",
                            help="HuggingFace LoRA adapter id")
    p_describe.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct",
                            help="Base model id")
    p_describe.add_argument("--max-frames", type=int, default=24,
                            help="Frames sampled per 10s clip")
    p_describe.add_argument("--limit", type=int, default=0,
                            help="Only process the first N clips (0 = all)")
    p_describe.add_argument("--no-caption", action="store_true",
                            help="Classify only; skip natural-language captions")

    # annotate (Label Studio project build / fine-tune export)
    p_annotate = subparsers.add_parser(
        "annotate", help="Build a Label Studio review project or export fine-tune data"
    )
    p_annotate.add_argument("--run", help="Run directory path (build mode)")
    p_annotate.add_argument("--out", help="Where to write config + tasks (default: run dir)")
    p_annotate.add_argument("--serve-root",
                            help="Label Studio LOCAL_FILES_DOCUMENT_ROOT (default: runs base)")
    p_annotate.add_argument("--push", action="store_true",
                            help="Create the project on a running Label Studio server")
    p_annotate.add_argument("--ls-url", help="Label Studio URL (with --push)")
    p_annotate.add_argument("--ls-key", help="Label Studio API token (with --push)")
    p_annotate.add_argument("--title", help="Project title (with --push)")
    p_annotate.add_argument("--export",
                            help="Convert a Label Studio export JSON into fine-tune JSONL")
    p_annotate.add_argument("--finetune-out", help="Output JSONL path (with --export)")
    p_annotate.add_argument("--clips-root", help="Clips dir to resolve clip paths (with --export)")

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
    elif args.command == "trim-empty":
        from soccer_vision.cli.trim import run_trim_empty
        run_trim_empty(args)
    elif args.command == "verify":
        from soccer_vision.cli.ask import run_verify
        run_verify(args)
    elif args.command == "ask":
        from soccer_vision.cli.ask import run_ask
        run_ask(args)
    elif args.command == "describe":
        from soccer_vision.cli.describe import run_describe
        run_describe(args)
    elif args.command == "annotate":
        from soccer_vision.cli.annotate import run_annotate
        run_annotate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
