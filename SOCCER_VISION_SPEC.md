# soccer-vision — Implementation Specification

> **Purpose:** Hand this document to Claude Code to refactor `soccer-video-analysis` into a modular Python package and desktop application. This spec consolidates the full architecture plan from the design conversation (June 2026).

---

## Executive Summary

Build **soccer-vision**: an open-source soccer video analysis toolkit that replicates capabilities of subscription products (Trace, Veo Editor, LongoMatch) without vendor lock-in. The package sits on top of [OpenSportsLab/opensportslib](https://github.com/OpenSportsLab/opensportslib) and uses SoccerNet baselines where possible.

**Product name:** `soccer-vision` (PyPI: `soccer-vision`, import: `soccer_vision`)

**Not Saints-specific.** Saints U10 Azul is one optional **project profile** (roster, IDPs). Core library is generic.

**Do not use "Veo" as a camera type anywhere.** Veo is a commercial vendor we are replacing. Use: *raw match footage*, *single-camera*, *wide-angle*, *panoramic*.

**Do not use Ultralytics/YOLO.** Use Transformers-based models (RF-DETR, opensportslib, SAM3) + [supervision](https://github.com/roboflow/supervision).

---

## Goals

Replicate subscription soccer video tools with open models:

| Commercial feature | soccer-vision equivalent |
|---|---|
| Trace — per-player highlight reels | Jersey tracking + ball proximity + clip DB + reels |
| Veo Editor — AI events, team stats | Action spotting + possession/distance metrics |
| LongoMatch — manual tagging, playlists | Desktop reviewer with timeline, tags, clip browser |

**Outputs:**
- Clips of a detected player merged into highlight reels
- Team clips filtered by event type (goal kicks, corners, throw-ins, etc.)
- Game statistics: shots, possession, distance covered, heatmaps

**Principles:**
- Simpler over complex
- Existing CV tools over new code
- Thin adapters, not reimplementations
- OSL JSON 2.0 as canonical interchange format
- Python best practices: pyproject.toml, pytest, GitHub Actions, Read the Docs, PyPI

---

## Canonical Pipeline (linear, one path)

Every match video runs through this single workflow:

```
1. Load raw video
      ↓
2. Create virtual broadcast (follow-cam proxy)
      ↓
3. Ball detection
      ↓
4. Player tracking
      ↓
5. Field registration (calibration)
      ↓
6. Event detection
      ↓
7. Team metrics
      ↓
8. Database logging
      ↓
9. Highlight and clip creation
```

**Why step 2 exists:** SoccerNet models (calibration, spotting, game-state) are trained on broadcast-style framing. Raw wide-angle single-camera footage gets normalized into a 16:9 follow-cam proxy so one model stack works for all footage. Reference: [AutoCam-AI](https://github.com/chele-s/AutoCam-AI) (RF-DETR + Kalman + virtual camera).

There is **no dual-path** (raw vs broadcast). Always create broadcast proxy first; downstream steps read the proxy.

---

## Current Repo State (starting point)

Flat repo with 4 CLI scripts:

| File | Purpose |
|---|---|
| `detect_actions.py` | Pipeline A: frame sampling → contact sheets + index.json (no ML) |
| `track.py` | Pipeline B: YOLO ball + Hough homography → goal-kick candidates |
| `extract_clips.py` | ffmpeg clip cut + concat |
| `register.py` | Optional KpSFR homography pre-compute |

**Migrate and refactor** this logic into `soccer_vision` package modules. Preserve behavior where possible; replace YOLO with RF-DETR.

**Remove/replace:**
- `yolov8n.pt` bundled weights
- All Ultralytics imports
- All references to "Veo" as camera type (grep and fix)

---

## Package Structure

```
soccer-video-analysis/          # repo root; consider renaming to soccer-vision
├── pyproject.toml              # hatchling, ruff, pytest, optional extras
├── README.md
├── docs/                       # Sphinx → Read the Docs
│   ├── conf.py
│   ├── index.rst
│   ├── quickstart.rst
│   ├── pipeline.rst
│   └── profiles.rst
├── .github/workflows/
│   ├── ci.yml                  # lint + unit tests (no GPU)
│   └── docs.yml
├── src/soccer_vision/
│   ├── __init__.py
│   ├── cli/
│   │   ├── main.py             # entry: soccer-vision
│   │   ├── process.py          # full pipeline
│   │   ├── extract.py          # clip extraction
│   │   └── ask.py              # Claude query CLI
│   ├── io/
│   │   ├── video.py            # load, sample, ffmpeg helpers
│   │   ├── osl.py              # OSL JSON 2.0 read/write
│   │   └── project.py          # project folder layout
│   ├── broadcast/
│   │   └── virtual_cam.py      # STEP 2: follow-cam proxy generation
│   ├── detection/
│   │   ├── rfdetr.py           # RF-DETR wrapper → sv.Detections
│   │   └── ball.py             # ball-specific detection
│   ├── tracking/
│   │   ├── bytetrack.py        # supervision ByteTrack wrapper
│   │   ├── sam3.py             # SAM3 player masks (optional GPU)
│   │   └── gamestate.py        # sn-gamestate / TrackLab adapter
│   ├── registration/
│   │   ├── hough.py            # Hough-line homography fallback
│   │   ├── kpsfr.py            # optional KpSFR subprocess
│   │   └── sn_calib.py         # sn-calibration adapter
│   ├── events/
│   │   ├── set_piece.py        # goal kick / corner / throw-in heuristics
│   │   ├── spotting.py         # opensportslib LocalizationModel + sn-teamspotting
│   │   └── phases.py           # in-play vs dead-ball (rule-based)
│   ├── metrics/
│   │   ├── distance.py
│   │   ├── possession.py
│   │   ├── shots.py
│   │   └── heatmap.py
│   ├── store/
│   │   ├── db.py               # SQLite: matches, players, events, clips
│   │   └── schema.sql
│   ├── clips/
│   │   ├── extract.py          # ffmpeg cut + concat
│   │   └── reels.py            # per-player / per-event highlight merge
│   ├── verify/
│   │   ├── sheets.py           # contact sheets for human/Claude review
│   │   └── claude.py           # Claude API integration
│   ├── profiles/
│   │   └── loader.py           # YAML project profiles (roster, IDP paths)
│   └── gui/                    # optional [gui] extra
│       ├── app.py              # PySide6: soccer-vision review
│       ├── browser.py
│       ├── player.py
│       ├── idp_panel.py
│       └── claude_panel.py
├── training/                   # only where public weights missing
│   ├── sn_calib/
│   └── README.md
├── tests/
│   ├── test_osl.py
│   ├── test_set_piece.py
│   ├── test_possession.py
│   ├── test_virtual_cam.py
│   └── fixtures/
├── examples/
│   ├── process_match.yaml
│   └── profiles/
│       └── saints-u10.yaml       # example profile, not core logic
└── legacy/                       # optional: keep old scripts as shims during migration
    ├── detect_actions.py
    ├── track.py
    └── extract_clips.py
```

---

## Dependencies

```toml
[project]
name = "soccer-vision"
requires-python = ">=3.12"   # match opensportslib
dependencies = [
  "opensportslib>=0.2.0",
  "opencv-python>=4.8",
  "numpy>=1.24",
  "transformers>=4.45",
  "torch>=2.2",
  "supervision>=0.25",
  "pyyaml>=6.0",
  "sqlalchemy>=2.0",
  "anthropic>=0.40",          # Claude API
]

[project.optional-dependencies]
gpu = ["inference[gpu]"]       # SAM3 streaming
gui = ["pyside6>=6.6", "supervision[desktop]"]
broadcast = ["tracklab"]       # sn-gamestate; heavy, install on demand
train = ["lightning"]
dev = ["pytest", "ruff", "sphinx"]
```

**System requirement:** `ffmpeg` on PATH.

**Do NOT include:** `ultralytics`, bundled YOLO weights.

---

## Model Stack (Transformers, not Ultralytics)

| Task | Model | Source |
|---|---|---|
| Ball + players + GK + referee | RF-DETR fine-tuned on SoccerNet | `julianzu9612/RFDETR-Soccernet` (HF) |
| Small ball fallback | RF-DETR ball-only | `eeeeeeeeeeeeee3/soccer-ball-detection` (HF) |
| Player segmentation / tracking | SAM3 | `transformers` / Roboflow inference |
| Multi-object IDs | ByteTrack | `supervision` |
| Field calibration | sn-calibration | [Google Drive weights](https://drive.google.com/file/d/1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM) |
| Player positions + jersey + team | sn-gamestate / TrackLab | Zenodo auto-download |
| Action / event spotting | sn-teamspotting (T-DEED) + opensportslib `LocalizationModel` | GitHub + HF |

**Internal representation:** always `supervision.Detections`. Convert at model boundaries only.

**Inference pattern:**

```python
import supervision as sv
from soccer_vision.detection.rfdetr import RFDETRSoccerDetector

detector = RFDETRSoccerDetector.from_pretrained("julianzu9612/RFDETR-Soccernet")
detections = detector.predict(frame)  # → sv.Detections
tracker = sv.ByteTrack()
tracked = tracker.update_with_detections(detections)
```

---

## Step 2: Virtual Broadcast (detail)

**Module:** `soccer_vision.broadcast.virtual_cam`

**Input:** raw match video (any wide-angle single-camera source)  
**Output:** `{run_dir}/broadcast_proxy.mp4` + per-frame crop metadata JSON

**Algorithm v1:**

1. Decode video; downsample frames to 1080p for inference.
2. Run RF-DETR at `detect_fps` (default 5 fps) for ball + players.
3. Compute action centroid: ball position; if ball missing, cluster centroid of players.
4. Smooth crop window with One-Euro filter or Kalman filter.
5. Dynamic zoom: tighter when action localized, wider at restarts/set pieces.
6. Render full-resolution crop to 16:9 proxy; interpolate crop between detection frames.

**Config (YAML):**

```yaml
broadcast:
  output_aspect: "16:9"
  output_resolution: [1920, 1080]
  smooth_window_s: 0.4
  min_zoom: 1.0       # show full width
  max_zoom: 2.5       # tight on action
  detect_fps: 5
```

**Reference implementation to study (do not copy wholesale):** [chele-s/AutoCam-AI](https://github.com/chele-s/AutoCam-AI)

All downstream pipeline steps read `broadcast_proxy.mp4`, not the raw file. Keep raw for archival and optional full-field heatmaps.

---

## OSL JSON 2.0 Integration

Use [OSL JSON format](https://opensportslab.github.io/opensportslib/data/osl-json-format/) for all annotations, predictions, and pipeline outputs.

Each processed match produces:

```
runs/{match_id}/
├── raw.mp4                    # symlink or copy of input
├── broadcast_proxy.mp4
├── annotations.json           # OSL JSON with events, metadata
├── tracking.parquet           # optional tracking data
├── stats.json                 # team metrics
├── clips/                     # extracted clip files
└── sheets/                    # contact sheets for verify step
```

**Event labels (localization `events[]`):**

```
goal_kick, corner_kick, throw_in, free_kick, kickoff,
shot, save, pass, foul, substitution, halftime
```

Map SoccerNet/sn-teamspotting labels → this taxonomy in `events/spotting.py`.

**Action detector, team & player attribution (action-agnostic):**

Detection is decoupled from attribution and clip selection via a pluggable
`ActionDetector` interface (`events/sources.py`): `is_available()` / `detect()`.
Each implementation is an *engine* named by what it does, not the model behind it,
and selected by plain name (`--action-engine`, or `action_engines:` in config):
`RulesActionDetector` = `rules` (live), `LearnedActionDetector` = `learned`
(interface until a checkpoint is passed), `VLMActionDetector` = `vlm` (opt-in).
All emit the same event dicts, so new engines add labels (`tackle`, `goal`,
`losing_the_ball`, ...) without touching downstream code. (Pre-rename names
`EventSource` / `SetPieceSource` / `TackleSource` / `SoccerChatSource` and the
`sources`/`set_piece`/`tackle`/`soccerchat` config keys remain as back-compat
aliases.)

- **Team assignment** is v1 by **jersey colour** (`tracking/teams.py`): cluster
  tracked players into two teams and name each cluster (blue / white / ...), so
  events are filterable by `--team blue`. Jersey OCR → named roster (sn-jersey /
  sn-gamestate) and SAM3 masklet identity are later phases behind the same seam.
- **Association** (`events/associate.py`) tags each event with the nearest
  player's `track_id` and their `team`; persisted on the `events` table and in
  OSL. Clips are then selectable per team or per player (track), composable with
  the event label, via `soccer-vision extract` / `reel` (`--team`, `--track`,
  `--event`).

---

## Metrics Engine

All metrics from **field coordinates** (homography) + **tracker states**. No custom deep models for stats.

| Metric | Method |
|---|---|
| **Distance covered** | Sum of √(Δx²+Δy²) per track_id in field metres |
| **Possession %** | Per in-play frame: ball within 3 m of nearest player → assign to team |
| **Shots** | spotting `shot` class OR ball speed toward goal mouth |
| **Touches** | Ball within 2 m of player for ≥2 consecutive frames |
| **Sprints** | Speed > 5 m/s for ≥1 s on field coords |
| **Heatmaps** | 2D histogram of player field positions |

Export: `stats.json` + OSL metadata.

**Youth 7v7 field model** (from existing `track.py`):

```python
FIELD_W_M = 55.0   # touchline
FIELD_H_M = 36.0   # goal line to goal line
BOX_DEPTH_M = 5.5  # goal area depth
```

---

## Set-Piece Heuristics (preserve from track.py)

Migrate existing goal-kick detection logic to `events/set_piece.py`:

- Sample ball position on broadcast proxy
- Project to field coordinates via homography
- Flag when ball is stationary (≥3 consecutive samples, ≤40 px drift) inside goal-area zone
- Deduplicate within 5 s window
- Merge with spotting model outputs (spotting takes precedence when confident; heuristics fill gaps)

Also extend heuristics for corner kicks (ball in corner arc + clustered players) and throw-ins (ball near touchline, stationary).

---

## Database Schema (SQLite)

```sql
-- matches
CREATE TABLE matches (
  id TEXT PRIMARY KEY,
  raw_path TEXT,
  proxy_path TEXT,
  processed_at TIMESTAMP,
  osl_path TEXT,
  stats_path TEXT
);

-- players (from project profile)
CREATE TABLE players (
  id TEXT PRIMARY KEY,
  name TEXT,
  jersey INTEGER,
  team TEXT,
  profile_id TEXT
);

-- events
CREATE TABLE events (
  id TEXT PRIMARY KEY,
  match_id TEXT,
  label TEXT,
  position_ms INTEGER,
  frame INTEGER,
  confidence REAL,
  verified BOOLEAN DEFAULT FALSE,
  FOREIGN KEY (match_id) REFERENCES matches(id)
);

-- clips
CREATE TABLE clips (
  id TEXT PRIMARY KEY,
  match_id TEXT,
  event_id TEXT,
  player_id TEXT,       -- nullable for team clips
  path TEXT,
  pre_s REAL,
  post_s REAL,
  FOREIGN KEY (match_id) REFERENCES matches(id)
);

-- idp_links (connect clips to development plan focus areas)
CREATE TABLE idp_links (
  player_id TEXT,
  focus_skill TEXT,
  clip_id TEXT,
  notes TEXT
);
```

---

## Claude Integration

**Module:** `soccer_vision.integrations.claude` (or `verify/claude.py`)

**Not Saints-specific.** Claude reads structured data + optional project profile.

### Project Profile (YAML)

```yaml
# examples/profiles/saints-u10.yaml
team_name: "Saints U10 Azul"
season: "2026-spring"
profile_id: "saints-u10"

roster:
  - name: Noah
    jersey: 7
    role: holding_mid
    idp_focus: "stay high for back diamond"
  - name: Bryce
    jersey: 9
    role: striker
    idp_focus: "lateral movement for top diamond"
  # ... full roster

idp_source: "file:///Users/benweinstein/Dropbox/Saints/idps/"

claude:
  enabled: true
  # ANTHROPIC_API_KEY from environment
```

### Claude use cases

| Command | What it does |
|---|---|
| `soccer-vision verify --run RUN_ID` | Send contact sheet + candidate events → accept/reject/patch OSL |
| `soccer-vision ask "..."` | Natural language query over DB + OSL + profile |
| GUI Claude panel | Same, interactive |

**Rules:**
- Send OSL JSON + stats + profile context, not raw video
- Claude returns structured JSON patches to OSL `events[]`
- API key: `ANTHROPIC_API_KEY` env var

**Example verify prompt structure:**

```
You are verifying soccer event detections.
Roster: {profile.roster}
Candidate events: {osl.events}
Contact sheet: [image attachment]

Return JSON: { "verified": [...], "rejected": [...], "reasons": {...} }
```

---

## Desktop Application

**Stack:** PySide6 + OpenCV/Qt video widget + SQLite  
**Entry:** `soccer-vision review --project ./my-team/`

| Panel | Function |
|---|---|
| File browser | Projects → matches → events → clips |
| Video player | Scrub timeline; overlay boxes via supervision annotators |
| Event lane | Color-coded markers from OSL events |
| Player filter | Roster dropdown from active profile |
| Tag editor | Manual corrections → write back to OSL JSON |
| Clip bin | Select events → export reel (ffmpeg concat) |
| IDP tab | Player → focus skill → linked clips |
| Stats tab | Possession bar, distance table, shot map |
| Claude panel | Verify / query |

Install: `pip install soccer-vision[gui]`

---

## SoccerNet Repo Inventory

| Repo | Use | Weights |
|---|---|---|
| [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) | Player track + jersey + minimap | Auto-download Zenodo |
| [sn-calibration](https://github.com/SoccerNet/sn-calibration) | Camera calibration | Google Drive |
| [sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting) | Team ball action spotting (2025) | T-DEED baseline in repo |
| [sn-spotting](https://github.com/SoccerNet/sn-spotting) | Classic action spotting | Baseline in repo |
| [sn-jersey](https://github.com/SoccerNet/sn-jersey) | Jersey OCR | Via gamestate MMOCR |
| [sn-tracking](https://github.com/SoccerNet/sn-tracking) | MOT benchmarks | Evaluation only |
| [sn-reid](https://github.com/SoccerNet/sn-reid) | Re-ID | Via gamestate |
| [sn-caption](https://github.com/SoccerNet/sn-caption) | Dense captions | Optional Claude context |
| [sn-grounding](https://github.com/SoccerNet/sn-grounding) | Replay grounding | Phase 3+ |
| [sn-mvfoul](https://github.com/SoccerNet/sn-mvfoul) | Multi-view fouls | Defer |
| [sn-depth](https://github.com/SoccerNet/sn-depth) | Depth estimation | Defer |

**Training scripts:** only create in `training/` where public weights are missing or broken (primarily sn-calibration backup). Do not write training scripts for gamestate/teamspotting until fine-tune is scoped.

---

## OpenSportsLib Upstream PRs

Contribute reusable pieces to [opensportslib](https://github.com/OpenSportsLab/opensportslib) (branch from `dev`, PR to `dev`):

| PR | Content |
|---|---|
| PR #1 | Virtual broadcast proxy as OSL input type (`broadcast_proxy` in `inputs[]`) |
| PR #2 | Set-piece heuristic localizer (goal kick, corner, throw-in) |
| PR #3 | Single-camera Hough registration fallback module |
| PR #4 | RF-DETR → OSL detection adapter |

Keep in soccer-vision repo: SQLite DB, GUI, Claude integration, clip reels, project profiles.

---

## CLI Commands

```bash
# Full pipeline
soccer-vision process match.mp4 --config examples/process_match.yaml
soccer-vision process match.mp4 --profile examples/profiles/saints-u10.yaml

# Individual steps
soccer-vision broadcast match.mp4 --out runs/match_001/
soccer-vision extract --run runs/match_001/ --events goal_kick corner_kick

# Verify with Claude
soccer-vision verify --run runs/match_001/ --profile saints-u10.yaml

# Natural language query
soccer-vision ask "Show clips where player 7 stayed high" --run runs/match_001/

# Desktop reviewer
soccer-vision review --project ./my-team/

# Highlight reels
soccer-vision reel --player "Noah" --run runs/match_001/ --out noah_highlights.mp4
soccer-vision reel --event goal_kick --run runs/match_001/ --out goal_kicks.mp4
```

---

## Testing

| Test file | Coverage |
|---|---|
| `test_osl.py` | OSL JSON round-trip read/write |
| `test_set_piece.py` | Synthetic ball positions → goal-kick detection |
| `test_possession.py` | Known positions → possession split |
| `test_virtual_cam.py` | Crop window smoothing (synthetic frames) |
| `test_extract.py` | Clip timecode math (mock ffmpeg) |

**CI:** ruff + pytest on every PR. No GPU, no model weights in CI.

Integration tests with real video: manual / optional nightly workflow.

---

## Implementation Phases

### Phase 0 — Foundation (do first)
- [ ] Create `pyproject.toml` + `src/soccer_vision/` layout
- [ ] Migrate `track.py`, `detect_actions.py`, `extract_clips.py`, `register.py` into package
- [ ] Replace Ultralytics with RF-DETR detector wrapper
- [ ] OSL JSON read/write + unit tests
- [ ] GitHub Actions CI
- [ ] Remove `yolov8n.pt`; grep-remove "Veo" as camera type
- [ ] Backward-compatible CLI shims in `legacy/` if needed

### Phase 1 — Virtual Broadcast
- [ ] `broadcast/virtual_cam.py` with config YAML
- [ ] Output `broadcast_proxy.mp4` + crop metadata
- [ ] Wire into `soccer-vision process` as step 2

### Phase 2 — Full Pipeline
- [ ] Ball detection (RF-DETR) + ByteTrack
- [ ] Field registration (sn-calibration + Hough fallback)
- [ ] Event detection (set-piece heuristics + spotting adapter stub)
- [ ] Metrics (distance, possession, shots)
- [ ] SQLite DB + OSL export
- [ ] Clip extraction + reel merge

### Phase 3 — Profiles + Claude
- [ ] Project profile loader (YAML)
- [ ] `soccer-vision verify` CLI
- [ ] `soccer-vision ask` CLI
- [ ] Example `saints-u10.yaml` profile

### Phase 4 — Desktop App
- [ ] PySide6: file browser, video player, event timeline
- [ ] IDP tab, stats tab, Claude panel
- [ ] `pip install soccer-vision[gui]`

### Phase 5 — Advanced Models
- [ ] sn-gamestate / TrackLab adapter (optional `[broadcast]` extra)
- [ ] sn-teamspotting / opensportslib LocalizationModel
- [ ] SAM3 player tracking (optional `[gpu]` extra)
- [ ] opensportslib upstream PRs

### Phase 6 — Publish
- [ ] Read the Docs
- [ ] PyPI publish
- [ ] Example notebooks

---

## Migrate Existing Code — Mapping

| Current file | New module | Notes |
|---|---|---|
| `track.py` homography + goal-zone logic | `registration/hough.py` + `events/set_piece.py` | Replace YOLO with RF-DETR |
| `track.py` ball detection | `detection/ball.py` | RF-DETR, not YOLO class 32 |
| `detect_actions.py` sampling + sheets | `verify/sheets.py` + `io/video.py` | Keep contact sheet workflow |
| `extract_clips.py` | `clips/extract.py` | ffmpeg subprocess, unchanged logic |
| `register.py` | `registration/kpsfr.py` | Optional subprocess adapter |
| `CLAUDE.md` pipeline docs | `docs/pipeline.rst` | Update terminology |

---

## Terminology Rules

| ❌ Do not use | ✅ Use instead |
|---|---|
| Veo footage / Veo cameras / Veo path | wide-angle single-camera footage |
| saints-vision | soccer-vision |
| Saints-specific logic in core | project profile YAML |
| Ultralytics / YOLO | RF-DETR / transformers |
| Dual-path raw vs broadcast | Single path with virtual broadcast step |

---

## Risk Register

| Risk | Mitigation |
|---|---|
| RF-DETR slow on CPU | Downsample for detection; run broadcast step at 5 fps |
| sn-gamestate install heavy | Optional `[broadcast]` extra |
| SAM3 requires GPU | Graceful fallback to RF-DETR + ByteTrack |
| Jersey OCR fails on youth kits | Manual roster mapping in profile + GUI tag editor |
| AGPL (opensportslib) | Open-source soccer-vision; note in README |
| Single-camera calibration weak | Hough fallback; human verify via Claude contact sheets |

---

## Success Criteria

1. `pip install soccer-vision` works on macOS (CPU) and Linux (GPU optional)
2. `soccer-vision process match.mp4` produces: broadcast proxy, OSL JSON, stats, clips DB
3. Goal-kick detection works at least as well as current `track.py` on test footage
4. `soccer-vision review` opens desktop with file browser and clip playback
5. `soccer-vision verify --profile saints-u10.yaml` sends candidates to Claude and patches OSL
6. No Ultralytics dependency anywhere
7. No "Veo" as camera terminology anywhere in code or docs

---

## References

- [OpenSportsLib](https://github.com/OpenSportsLab/opensportslib) — base library
- [OSL JSON format](https://opensportslab.github.io/opensportslib/data/osl-json-format/)
- [supervision](https://github.com/roboflow/supervision)
- [RF-DETR SoccerNet](https://huggingface.co/julianzu9612/RFDETR-Soccernet)
- [sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- [sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting)
- [AutoCam-AI](https://github.com/chele-s/AutoCam-AI) — virtual broadcast reference
- [SoccerNet challenges](https://www.soccer-net.org/challenges)

---

*Generated from architecture design conversation, June 2026. Give this file to Claude Code with: "Implement Phase 0 and Phase 1 from SOCCER_VISION_SPEC.md"*
