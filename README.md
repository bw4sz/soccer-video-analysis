# soccer-vision

Open-source soccer video analysis toolkit. Replicates capabilities of subscription products (Trace, Veo Editor, LongoMatch) without vendor lock-in — designed for coaches and analysts who want programmatic control over their match footage.

**Target audience:** coaches, sports analysts, and developers working with single-camera match video (Veo, TrackMan, or any fixed/overhead camera). Python 3.12+, CPU-viable, no cloud dependency.

Built on [OpenSportsLib](https://github.com/OpenSportsLab/opensportslib) and [supervision](https://github.com/roboflow/supervision). Uses Transformers-based models (RF-DETR, SAM3) — no Ultralytics/YOLO.

## Install

```bash
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# With desktop reviewer
pip install -e ".[gui]"
```

**System requirement:** `ffmpeg` on PATH.

## Usage

```bash
# Full pipeline: broadcast proxy → detection → tracking → events → metrics → clips
soccer-vision process match.mp4

# With config
soccer-vision process match.mp4 --config examples/process_match.yaml

# Generate broadcast proxy only
soccer-vision broadcast match.mp4

# Extract clips from a processed run
soccer-vision extract --run runs/match_001/ --events goal_kick corner_kick

# Build highlight reel
soccer-vision reel --event goal_kick --run runs/match_001/ --out goal_kicks.mp4
```

## Pipeline

Every match video runs through a single canonical workflow:

1. **Load raw video**
2. **Virtual broadcast** — follow-cam 16:9 proxy from wide-angle single-camera footage
3. **Ball detection** — RF-DETR fine-tuned on SoccerNet
4. **Player tracking** — ByteTrack via supervision
5. **Field registration** — Hough-line homography (sn-calibration fallback)
6. **Event detection** — set-piece heuristics + spotting model adapters
7. **Team metrics** — distance, possession, shots, heatmaps
8. **Database logging** — SQLite + OSL JSON 2.0
9. **Clip creation** — ffmpeg extraction + highlight reels

## Outputs

Each processed match produces:

```
runs/{match_id}/
├── broadcast_proxy.mp4    # 16:9 follow-cam proxy
├── annotations.json       # OSL JSON 2.0 events
├── stats.json             # team metrics
├── clips/                 # extracted clips
└── sheets/                # contact sheets for review
```

## Project Profiles

Customize for your team with a YAML profile:

```bash
soccer-vision process match.mp4 --profile examples/profiles/saints-u10.yaml
```

## Dependencies

- Python 3.12+
- `opencv-python`, `numpy`, `torch`, `transformers`, `supervision`
- `ffmpeg` on PATH

## Quick two-script usage (no install)

If you just want to extract highlight clips without the full pipeline:

```bash
pip install opencv-python numpy   # ffmpeg must be on PATH

# Detect goal-kick candidates (CPU, ~2–4 min per 60-min match)
python detect_actions.py --video match.mp4 --action goal_kick --preview --out candidates.json

# Review candidates_sheet.jpg, then extract clips
python extract_clips.py --video match.mp4 --candidates candidates.json \
  --pre 5 --post 30 --concat --concat-out goal_kicks.mp4
```

For YOLO-assisted detection with field registration, see [`track.py`](track.py) and [`register.py`](register.py).

## Contributing

Issues and PRs welcome. The project is early-stage — the most useful contributions right now are:

- Additional event detectors (free kicks, throw-ins, substitutions)
- Field registration improvements for non-broadcast cameras
- Test fixtures (short anonymized video clips)

## License

AGPL-3.0 (matches opensportslib upstream).
