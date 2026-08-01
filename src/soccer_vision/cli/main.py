"""soccer-vision CLI entry point."""

from __future__ import annotations

import argparse
import sys


def _add_on_ball_args(parser: argparse.ArgumentParser) -> None:
    """Shared on-ball fallback options for `extract` and `reel`.

    Asking for one player's clips usually returns nothing from the event stream
    alone — the set-piece detector fires a handful of times a match. So when a
    player selection matches no events, both commands fall back to the spans
    where that player was the ball's nearest player.
    """
    parser.add_argument(
        "--no-on-ball", dest="on_ball", action="store_false",
        help="Don't fall back to ball-proximity spans when a --player/--number/"
             "--track selection matches no detected events (report nothing instead).")
    parser.set_defaults(on_ball=True)
    parser.add_argument(
        "--on-ball", dest="on_ball_force", action="store_true",
        help="Always cut ball-proximity spans for the selected player, even when "
             "an explicit event label was requested (which normally suppresses "
             "the fallback so --events pass can't silently return touches).")
    parser.add_argument(
        "--on-ball-dist", type=float, default=90.0, metavar="PX",
        help="Max pixel distance from ball to player's feet to count as on the "
             "ball (default: 90)")
    parser.add_argument(
        "--on-ball-min-span", type=float, default=0.4, metavar="SEC",
        help="Drop on-ball spans shorter than this as incidental (default: 0.4)")


def _add_field_filter_args(parser: argparse.ArgumentParser) -> None:
    """Shared on-field cut for `process` and `enroll --dump-frames`.

    The cut is asymmetric: these cameras sit close to the touchline, so the
    bottom and sides of the frame are our own pitch and only the top holds other
    people's matches. See :mod:`soccer_vision.detection.field_filter`.
    """
    from soccer_vision.detection.field_filter import (
        DEFAULT_BOTTOM_FRAC, DEFAULT_SIDE_FRAC, DEFAULT_TOP_FRAC,
    )

    parser.add_argument(
        "--field-top", type=float, default=DEFAULT_TOP_FRAC, metavar="FRAC",
        help="Drop detections whose feet are in the top FRAC of the frame — sky, "
             "trees, rooftops (default: %(default)s). It will not separate the "
             "far-touchline crowd or a neighbouring pitch, which sit below that "
             "line among our own far-side players.")
    parser.add_argument(
        "--field-sides", type=float, default=DEFAULT_SIDE_FRAC, metavar="FRAC",
        help="Drop detections within FRAC of the left/right edge (default: "
             "%(default)s — a wide frame is one pitch across, so cutting the "
             "sides only loses real players).")
    parser.add_argument(
        "--field-bottom", type=float, default=DEFAULT_BOTTOM_FRAC, metavar="FRAC",
        help="Drop detections within FRAC of the bottom edge (default: "
             "%(default)s — with the camera at the touchline there is nobody "
             "between it and the pitch). Raise it only for a camera set well back.")


def field_filter_kwargs(args) -> dict:
    """``--field-*`` as keyword arguments for ``filter_spectators``.

    Falls back to the module defaults so a caller assembling ``args`` by hand
    (tests, notebooks) gets the shipped cut rather than ``None``.
    """
    from soccer_vision.detection.field_filter import (
        DEFAULT_BOTTOM_FRAC, DEFAULT_SIDE_FRAC, DEFAULT_TOP_FRAC,
    )

    # `is None`, not `or`: --field-top 0 is a legitimate "cut nothing".
    def _pick(name, default):
        value = getattr(args, name, None)
        return default if value is None else float(value)

    return {
        "top_frac": _pick("field_top", DEFAULT_TOP_FRAC),
        "side_frac": _pick("field_sides", DEFAULT_SIDE_FRAC),
        "bottom_frac": _pick("field_bottom", DEFAULT_BOTTOM_FRAC),
    }


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
    _add_field_filter_args(p_process)
    p_process.add_argument(
        "--broadcast", action="store_true",
        help="Crop wide/zoomed-out footage into a steadied, followed 16:9 view "
             "before processing. Off by default; most footage doesn't need it, "
             "and it's a pan-only crop (no zoom) that hasn't been validated on "
             "real matches end-to-end. Use `soccer-vision broadcast` to preview "
             "it on a clip before turning it on here.",
    )

    # broadcast
    p_broadcast = subparsers.add_parser("broadcast", help="Generate broadcast proxy only")
    p_broadcast.add_argument("video", help="Path to input video file")
    p_broadcast.add_argument("--out", help="Output directory")
    p_broadcast.add_argument("--config", help="Broadcast config YAML")

    # identify
    p_identify = subparsers.add_parser(
        "identify", help="Read jersey numbers per track (individual-player pathway)"
    )
    p_identify.add_argument("--run", required=True, help="Run directory path")
    p_identify.add_argument("--profile", help="Project profile YAML (maps jersey → name)")
    p_identify.add_argument("--model", help="Recognizer checkpoint id (default: parseq)")
    p_identify.add_argument("--device", default=None, help="PyTorch device: cpu / cuda")
    p_identify.add_argument("--max-samples", type=int, default=40,
                            help="Max frames sampled per track (default: 40)")
    p_identify.add_argument("--min-votes", type=int, default=3,
                            help="Min legible reads to name a track (default: 3)")
    p_identify.add_argument("--min-share", type=float, default=0.5,
                            help="Winning number's min share of vote weight (default: 0.5)")
    p_identify.add_argument("--min-margin", type=float, default=0.15,
                            help="Min weight-share lead over runner-up (default: 0.15)")
    p_identify.add_argument("--method", default="auto",
                            choices=["auto", "ocr", "reid", "reid+ocr"],
                            help="How to name tracks: appearance re-id, jersey OCR, "
                                 "or re-id with OCR filling the abstentions "
                                 "(default: auto — reid+ocr if a gallery is available)")
    p_identify.add_argument("--gallery", help="Player appearance gallery from `enroll` "
                                              "(default: <run>/gallery.npz, or the profile's)")
    p_identify.add_argument("--min-similarity", type=float, default=None,
                            help="Min gallery cosine similarity to name a track (default: 0.5)")
    p_identify.add_argument("--min-reid-margin", type=float, default=None,
                            help="Min similarity lead over the runner-up player (default: 0.05)")

    # link-tracks
    p_link = subparsers.add_parser(
        "link-tracks",
        help="Rejoin fragmented ByteTrack lanes across detector dropouts")
    p_link.add_argument("--run", required=True, help="Run directory path")
    p_link.add_argument("--max-gap", type=float, default=1.5,
                        help="Max dropout to bridge, seconds (default: 1.5)")
    p_link.add_argument("--max-dist", type=float, default=150.0,
                        help="Max px between where motion says the player should "
                             "be and where the next lane starts (default: 150)")
    p_link.add_argument("--no-motion", action="store_true",
                        help="Compare raw positions instead of extrapolating "
                             "velocity across the gap (worse; for comparison)")
    p_link.add_argument("--ignore-kit", action="store_true",
                        help="Allow links between different kit colours")
    p_link.add_argument("--appearance", action="store_true",
                        help="Also require the two lane edges to look like the "
                             "same person. Needs the proxy video; rejects 85%% of "
                             "the links made when no true continuation exists")
    p_link.add_argument("--min-appearance", type=float, default=0.70,
                        help="Cosine similarity floor for --appearance (default: 0.70)")
    p_link.add_argument("--no-interpolate", action="store_true",
                        help="Don't fill positions across bridged gaps")
    p_link.add_argument("--in-place", action="store_true",
                        help="Overwrite tracks.json/jerseys.json so reel and "
                             "extract use the linked ones (originals kept as "
                             "*.unlinked.json)")
    p_link.add_argument("--device", default=None, help="PyTorch device for --appearance")

    # enroll
    p_enroll = subparsers.add_parser(
        "enroll", help="Bank a team's appearances into a re-id gallery (carried between matches)"
    )
    p_enroll.add_argument("--run", help="Run directory path")
    p_enroll.add_argument("--video",
                          help="Export labelling frames straight from a video, detecting "
                               "only on the exported frames — no `process` run needed "
                               "(with --dump-frames)")
    p_enroll.add_argument("--profile", help="Project profile YAML (maps jersey → name)")
    p_enroll.add_argument("--out",
                          help="Gallery path (default: gallery.npz beside the run, or "
                               "beside a --from-label-studio export)")
    p_enroll.add_argument("--append", action="store_true",
                          help="Merge into the existing gallery instead of replacing it")
    p_enroll.add_argument("--dump-frames", metavar="DIR",
                          help="Write whole frames + a Label Studio project for naming "
                               "players on them, then exit. Boxes come pre-drawn from "
                               "tracks.json; enrol the export with --from-label-studio")
    p_enroll.add_argument("--dump-tracklets", metavar="DIR",
                          help="Write windows of play as clips with every tracked "
                               "player ringed and numbered, plus a Label Studio "
                               "project naming them. One decision per lane harvests "
                               "every crop in it; enrol with --from-tracklets")
    p_enroll.add_argument("--from-tracklets", metavar="JSON",
                          help="Enrol from a --dump-tracklets export (needs the "
                               "tracklets.json manifest beside it, and --run for the "
                               "video the crops are cut from)")
    p_enroll.add_argument("--manifest", metavar="JSON",
                          help="tracklets.json for --from-tracklets, if it isn't "
                               "beside the export")
    p_enroll.add_argument("--serve-url", metavar="BASE",
                          help="Write absolute clip URLs against this base (e.g. "
                               "http://localhost:8000) instead of Label Studio's "
                               "/data/local-files/ endpoint — serve the folder with "
                               "`python -m http.server` when local-files serving "
                               "won't cooperate")
    p_enroll.add_argument("--window", type=float, default=20.0, metavar="SEC",
                          help="Seconds per tracklet window (default: 20)")
    p_enroll.add_argument("--n-windows", type=int, default=8,
                          help="Windows to render, spread across the match (default: 8). "
                               "Spread beats length — a gallery wants varied views, and "
                               "one lane's crops are 1.4s of near-duplicates")
    p_enroll.add_argument("--max-lanes", type=int, default=12,
                          help="Lanes ringed per window, longest first (default: 12). "
                               "Fixes the number of dropdowns in the config")
    p_enroll.add_argument("--n-frames", type=int, default=20,
                          help="Frames to export for labelling, spread across the match "
                               "(default: 20)")
    p_enroll.add_argument("--min-players", type=int, default=4,
                          help="Skip frames showing fewer than this many players of the "
                               "selected team (default: 4)")
    p_enroll.add_argument("--min-y-frac", type=float, default=0.0,
                          help="Keep only detections whose feet are below this fraction of "
                               "frame height — the way to exclude a neighbouring pitch's "
                               "match at a multi-field complex (e.g. 0.45; 0 keeps all)")
    _add_field_filter_args(p_enroll)
    p_enroll.add_argument("--min-motion", type=float, default=6.0,
                          help="Drop detections that barely move between frames half a "
                               "second apart — the seated crowd and the next pitch over "
                               "(default: 6.0 grey levels; 0 disables)")
    p_enroll.add_argument("--serve-root",
                          help="Label Studio LOCAL_FILES_DOCUMENT_ROOT the frame paths are "
                               "written relative to (default: the runs/ base)")
    p_enroll.add_argument("--from-label-studio", metavar="JSON",
                          help="Enrol from a Label Studio rectanglelabels export — the "
                               "labelled frames beside it are the crops, so no run or "
                               "video is needed (default: bootstrap from jerseys.json)")
    p_enroll.add_argument("--frames", metavar="DIR",
                          help="Where the frames a --from-label-studio export was drawn "
                               "on live (default: frames/ beside the export)")
    p_enroll.add_argument("--min-track-frames", type=int, default=20,
                          help="Skip tracks shorter than this when dumping (default: 20)")
    p_enroll.add_argument("--team",
                          help="Only dump lanes on this kit colour (e.g. black) — usually "
                               "you enrol one squad, not both. Uses the `teams` block of "
                               "tracks.json, else classifies kit colour and caches it")
    p_enroll.add_argument("--team-frames", type=int, default=300,
                          help="Frames sampled to classify kit colour when tracks.json "
                               "carries no `teams` block (default: 300)")
    p_enroll.add_argument("--weights", help="Re-id checkpoint (default: sportsreid OSNet_x1_0)")
    p_enroll.add_argument("--device", default=None, help="PyTorch device: cpu / cuda")
    p_enroll.add_argument("--max-samples", type=int, default=20,
                          help="Max frames sampled per track (default: 20)")
    p_enroll.add_argument("--max-per-player", type=int, default=64,
                          help="Max exemplars kept per player (default: 64)")
    p_enroll.add_argument("--min-confidence", type=float, default=0.8,
                          help="Min OCR vote confidence to enrol a track (default: 0.8)")
    p_enroll.add_argument("--min-obs", type=int, default=5,
                          help="Min legible OCR reads to enrol a track (default: 5)")
    p_enroll.add_argument("--exclude-jersey", nargs="+", type=int,
                          help="Jersey numbers to never enrol (OCR hallucination classes)")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract clips from a processed run")
    p_extract.add_argument("--run", required=True, help="Run directory path")
    p_extract.add_argument("--events", nargs="+", help="Event labels to extract")
    p_extract.add_argument("--team", help="Filter by team colour (e.g. blue)")
    p_extract.add_argument("--track", type=int, help="Filter by player track id")
    p_extract.add_argument("--player", help="Filter by player name (needs `identify` + profile)")
    p_extract.add_argument("--number", type=int, help="Filter by jersey number (needs `identify`)")
    p_extract.add_argument("--profile", help="Project profile YAML (resolves --player → jersey)")
    p_extract.add_argument("--pre", type=float, default=5.0)
    p_extract.add_argument("--post", type=float, default=30.0)
    p_extract.add_argument(
        "--halo", nargs="?", const="ellipse", choices=["ellipse", "circle"],
        help="Draw a gentle spotlight on each clip's player track "
             "(needs tracks.json from `process`). Default style: ellipse.")
    _add_on_ball_args(p_extract)

    # reel
    p_reel = subparsers.add_parser("reel", help="Build highlight reel")
    p_reel.add_argument("--run", required=True)
    p_reel.add_argument("--event", help="Filter by event label")
    p_reel.add_argument("--team", help="Filter by team colour (e.g. blue)")
    p_reel.add_argument("--track", type=int, help="Filter by player track id")
    p_reel.add_argument("--player", help="Filter by player name (needs `identify` + profile)")
    p_reel.add_argument("--number", type=int, help="Filter by jersey number (needs `identify`)")
    p_reel.add_argument("--profile", help="Project profile YAML (resolves --player name → jersey)")
    p_reel.add_argument("--out", default="highlight_reel.mp4")
    p_reel.add_argument(
        "--halo", nargs="?", const="ellipse", choices=["ellipse", "circle"],
        help="Spotlight each clip's player track (needs tracks.json from `process`).")
    _add_on_ball_args(p_reel)

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
    p_trim.add_argument("--sample-fps", type=float, default=15.0,
                        help="Sample rate when building a track (default: 15; the "
                             "flickery ball detector needs a dense track for Kalman "
                             "smoothing to lock on — lower rates over-reject)")
    p_trim.add_argument("--min-dead", type=float, default=5.0,
                        help="Min seconds of dead time before a span is cut (default: 5)")
    p_trim.add_argument("--stationary-px", type=float, default=40.0,
                        help="Max pixel drift to count as 'not moving' (default: 40)")
    p_trim.add_argument("--pad", type=float, default=0.5,
                        help="Seconds of context kept around each cut (default: 0.5)")
    p_trim.add_argument("--no-smooth", dest="smooth", action="store_false",
                        help="Skip Kalman smoothing of an auto-built ball track "
                             "(keep the raw, flickery detections)")
    p_trim.set_defaults(smooth=True)
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

    # harvest (pull CC-BY youth-soccer clips from YouTube)
    p_harvest = subparsers.add_parser(
        "harvest",
        help="Download short CC-BY youth-soccer clips from YouTube for annotation",
    )
    p_harvest.add_argument("--out-dir", default="data/youth_clips",
                           help="Output directory for clips + manifest")
    p_harvest.add_argument("-n", type=int, default=200,
                           help="Target number of clips/games (default: 200)")
    p_harvest.add_argument("--clip-len", type=float, default=10.0,
                           help="Clip length in seconds (default: 10)")
    p_harvest.add_argument("--position", choices=["middle", "random"], default="middle",
                           help="Where in the match to clip (default: middle)")
    p_harvest.add_argument("--position-frac", type=float, default=0.6,
                           help="Fraction of the match to centre the clip on with "
                                "--position middle (default: 0.6 = mid-second-half, "
                                "past the halftime kickoff; use 0.5 for true centre)")
    p_harvest.add_argument("--max-per-channel", type=int, default=2,
                           help="Cap clips per channel for diversity (default: 2)")
    p_harvest.add_argument("--per-query", type=int, default=50,
                           help="YouTube results fetched per search query (default: 50)")
    p_harvest.add_argument("--min-duration", type=float, default=300.0,
                           help="Skip videos shorter than this many seconds (default: 300)")
    p_harvest.add_argument("--max-height", type=int, default=720,
                           help="Cap clip resolution height (default: 720)")
    p_harvest.add_argument("--queries", nargs="+",
                           help="Override the default search queries")
    p_harvest.add_argument("--queries-file",
                           help="Read search queries from a file (one per line)")
    p_harvest.add_argument("--dry-run", action="store_true",
                           help="Search + licence-filter only; download nothing")

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
    elif args.command == "identify":
        from soccer_vision.cli.identify import run_identify
        run_identify(args)
    elif args.command == "link-tracks":
        from soccer_vision.cli.link import run_link_tracks
        run_link_tracks(args)
    elif args.command == "enroll":
        from soccer_vision.cli.enroll import run_enroll
        run_enroll(args)
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
    elif args.command == "harvest":
        from soccer_vision.cli.harvest import run_harvest
        run_harvest(args)
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
