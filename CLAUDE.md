# Soccer Video Analysis — Claude Instructions

Two pipelines are available. Use **Pipeline A** for quick visual review or when
YOLO detection is unreliable. Use **Pipeline B** when you want automated
candidate detection and only need Claude for final verification.

## SLURM job ledger

`slurm/job_ledger.md` tracks every SLURM job submitted for this project (why it
was run, its outcome, and follow-up). It's a project-scoped copy of the
cross-project `/home/b.weinstein/logs/job_ledger.md`, which mixes in other
projects. When you submit a job for this repo, append an entry here; when it
finishes, check its raw output under `/home/b.weinstein/logs/` or
`slurm/logs/<name>_<timestamp>/` and fill in the `Result` line.

---

## Pipeline A — Frame sampling (no YOLO)

Fast, always works, Claude does all the heavy lifting.

### Step 1 — Generate contact sheets

```bash
python detect_actions.py --video match.mp4 --effort medium --out-dir sheets/
```

Outputs `sheets/sheet_001.jpg` … and `sheets/index.json`.  
Each thumbnail is 320×180 px (one vision tile). Each sheet holds 30 thumbs.

| Effort | Interval | Gap | Sheets (60-min match) |
|---|---|---|---|
| `low`    | 1000 frames | ~33s | ~4  |
| `medium` | 500 frames  | ~17s | ~8  |
| `high`   | 250 frames  | ~8s  | ~16 |

Start with `medium`. Use `high` if you're missing events between sheets.

### Step 2 — Claude reads the sheets

Read each sheet image. For each thumbnail, look for:

**Goal kick**
- Camera framing one goal end
- Ball visible in or near the 6-yard box
- Goalkeeper near the ball; opposing players pulled back to penalty-area edge

**Corner kick**
- Ball in a corner arc
- Both teams clustered inside the penalty area

**Goal / near-goal**
- Keeper diving or beaten; ball crossing the line
- Celebration posture

**Halftime / break**
- Field nearly empty; spectators on pitch; players on sidelines

Output format after reading all sheets:

```
Goal kick candidates:
- F14500  (8:03)  — ball near right goal, keeper in position
- F66000  (36:42) — left goal end, static setup

False positives to exclude:
- F40500 / F41000 — halftime (field empty)
```

### Step 3 — Extract clips

```bash
python extract_clips.py \
  --video match.mp4 \
  --index sheets/index.json \
  --frames 14500 66000 90500 \
  --pre 0 --post 20 \
  --concat --concat-out goal_kick_reel.mp4
```

---

## Pipeline B — YOLO tracking → Claude verify → clips

More precise candidate detection. Claude only reads a single small contact
sheet at the end (≈ 5k tokens vs. 32k for Pipeline A).

### Step 1 — (Optional) Pre-compute field registration with KpSFR

KpSFR ([github.com/ericsujw/KpSFR](https://github.com/ericsujw/KpSFR)) is a
deep-learning model that outputs a homography matrix per frame. More robust
than the built-in Hough-line fallback when field lines are partially occluded.

**Caveat:** trained on broadcast World Cup footage, not Veo overhead cameras.
Test on a sample frame before relying on it. Skip if registration looks poor —
track.py falls back to Hough lines automatically.

```bash
git clone https://github.com/ericsujw/KpSFR kpsfr/
conda env create -f kpsfr/environment.yml && conda activate kpsfr
# Download kpsfr_finetuned.pth → kpsfr/checkpoint/
python register.py --video match.mp4 --kpsfr-repo kpsfr/ --fps 0.5
# → registrations/homographies.json
```

### Step 1b — Run ball tracker

```bash
pip install ultralytics   # one-time

# Without KpSFR (built-in Hough-line fallback):
python track.py --video match.mp4 --fps 3

# With KpSFR homography cache:
python track.py --video match.mp4 --fps 3 \
  --homography-cache registrations/homographies.json

# Options:
#   --fps 3              sample rate (default 3 fps — fast, CPU-only)
#   --stationary-frames 3  consecutive stationary detections needed
#   --stationary-px 40   max pixel drift to count as stationary
#   --half first/second  restrict to one half if known
#   --half-frame N       frame where second half starts
#   --model n/s/m        yolov8 size (n = fastest, default)
#   --device cpu/mps     cpu works fine
#   --homography-cache   path to homographies.json from register.py
```

Outputs `candidates/candidates.json` + `candidates/sheet_001.jpg`.

Typical runtime: **2–4 minutes** on CPU for a 60-min match.

### Step 2 — Claude verifies candidates

Read the contact sheet (`candidates/sheet_001.jpg`). Each thumbnail shows:
- Frame number and timestamp (top-left, cyan)
- Which goal zone: `[left]` or `[right]`
- A red circle drawn at the detected ball position

For each candidate, confirm or reject:

```
Verified goal kicks:
- F14500  (8:03)  [left]  — confirmed: ball in box, keeper setup
- F66000  (36:42) [right] — confirmed

Rejected:
- F22000  (12:12) [left]  — ball near touchline, not in box (false positive)
```

### Step 3 — Extract and concatenate clips

```bash
# Use all candidates from JSON
python extract_clips.py \
  --video match.mp4 \
  --candidates candidates/candidates.json \
  --pre 0 --post 20 \
  --concat --concat-out goal_kick_reel.mp4

# Use only specific frames Claude confirmed
python extract_clips.py \
  --video match.mp4 \
  --candidates candidates/candidates.json \
  --frames 14500 66000 \
  --pre 0 --post 20 \
  --concat --concat-out goal_kick_reel.mp4
```

---

## Trim empty — cut dead time into a shorter clip

Youth matches are mostly dead time (ball out of play, or sitting still while
players reposition). `trim-empty` removes spans where the ball is **offscreen**
or **not moving** for longer than `--min-dead` seconds, splicing the rest into a
new file. **The original is never modified.**

```bash
# Auto-build a ball track from the RF-DETR detector, then trim:
soccer-vision trim-empty match.mp4 --save-track ball_track.json

# Reuse a precomputed track (from the detector, or track.py in future):
soccer-vision trim-empty match.mp4 --track ball_track.json --out match.short.mp4

# Preview the cut list without rendering:
soccer-vision trim-empty match.mp4 --track ball_track.json --dry-run
```

Outputs `match.trimmed.mp4` (or `--out`) plus `match.trim.json`, an
edit-decision list recording every kept/removed span and why.

**Ball-track schema** (the input; the tracker is optional — supply your own
until it lands). Defined in `soccer_vision.events.deadball`:

```json
{
  "video": "match.mp4", "fps": 30.0, "sample_fps": 5.0,
  "width": 1920, "height": 1080, "total_frames": 108000,
  "samples": [
    {"frame": 0, "timestamp_s": 0.0, "visible": true,
     "pixel_x": 950.0, "pixel_y": 540.0, "confidence": 0.82},
    {"frame": 6, "timestamp_s": 0.2, "visible": false,
     "pixel_x": null, "pixel_y": null, "confidence": 0.0}
  ]
}
```

Key options: `--min-dead 5` (dead-span threshold), `--stationary-px 40` (max
drift to count as "not moving"), `--pad 0.5` (context kept around each cut),
`--no-smooth` (skip Kalman smoothing — see below).

**Kalman smoothing (auto-built tracks).** There is only one ball and it moves
smoothly, but the RF-DETR detector *flickers* — it latches onto a jersey number
or sponsor logo for a frame or two, so the reported position teleports and snaps
back. When `trim-empty` builds a track itself it runs the detections through a
constant-velocity Kalman filter (`soccer_vision.tracking.ball_kalman`) that
smooths jitter and gates out those jumps: a detection is rejected when its
Mahalanobis distance from the predicted position exceeds a χ² threshold, and the
ball coasts on the prediction instead. Several rejects in a row, or a long
offscreen gap, re-lock the filter onto the latest detection (so a genuine
relocation isn't fought forever). Smoothed samples keep the original detection
under `raw_pixel_x`/`raw_pixel_y` and gain `smoothed: true` (rejected jumps also
get `outlier: true`); offscreen samples pass through untouched so real
out-of-play gaps still read as dead time. Pass `--no-smooth` to keep raw
detections, or call `smooth_ball_track(track)` directly on a precomputed track.

Auto-built tracks default to `--sample-fps 15` because this detector is flickery
enough that the filter needs a dense track to lock on: on the saints validation
clip the raw ball teleports with a p95 frame-to-frame jump of ~850px, and at 5
fps the ball moves too far between samples for the gate to tell real motion from
a false positive (it over-rejects, ~39% of frames). At 15 fps rejection drops to
~32% and mean jump is more than halved; 30 fps (the FOOTPASS h5 export rate) is
better still.

---

## Harvest — build a diverse annotation set from YouTube

`harvest` pulls short, openly-licensed youth-soccer clips off YouTube to seed an
annotation set that's *broad* (many cameras, countries, kit colours) rather than
deep on any one team or camera. For each match it keeps a single ~10s clip of
live open play — sampled at 60% of the match by default (`--position-frac`),
*not* the exact centre, because a video's midpoint lands on the halftime /
second-half kickoff. Only videos the uploader released under **Creative Commons
Attribution (CC BY)** — the one reusable YouTube licence — are kept, and every
clip's provenance is logged for attribution (a CC BY duty).

```bash
pip install 'soccer-vision[harvest]'   # yt-dlp; needs ffmpeg (module load ffmpeg)

# Preview yield without downloading:
soccer-vision harvest --dry-run -n 200

# Harvest 200 clips into data/youth_clips/ (resumable — re-run to top up):
soccer-vision harvest --out-dir data/youth_clips -n 200
```

Outputs `clips/<video_id>.mp4`, `manifest.jsonl` (one provenance line per clip:
id, url, title, channel, licence, duration, clip window, query), and
`ATTRIBUTION.md`. Re-running skips video-ids already in the manifest, so a
200-clip set can be built over several sessions. Clips feed straight into the
`annotate` (Label Studio) flow for player / team / field-position labelling.

Key options: `-n 200` (target games), `--clip-len 10`, `--position-frac 0.6`
(where in the match to clip; 0.5 = true centre, 0.35 = mid-first-half),
`--max-per-channel 2` (diversity cap so one uploader can't dominate),
`--min-duration 300` (skip highlights/shorts), `--queries`/`--queries-file`
(override the default multi-lingual query list in `soccer_vision.harvest.queries`).

**Note:** downloading from YouTube is contrary to its ToS even for CC-BY content;
CC BY covers content *reuse*, not retrieval method. This mirrors how academic
vision datasets are built — keep the attribution manifest with any release.

---

## Token cost reference

| What | Reads | Est. tokens |
|---|---|---|
| Pipeline A: 8 medium sheets (30 thumbs × 320×180) | 8 | ~32k |
| Pipeline A: full pipeline w/ 5 verify reads | ~13 | ~72k |
| Pipeline B: 1 candidate contact sheet (≤30 thumbs) | 1 | ~5k |
| Single full-res frame for verification | 1 | ~8k |

Pipeline B is ~6× cheaper on Claude tokens if YOLO finds candidates reliably.
If ball detection is poor (bad lighting, camera angle), fall back to Pipeline A.

---

## Tips

- **Veo cameras** pan and zoom constantly — optical flow is unreliable. Visual
  inspection is more accurate than motion-based heuristics.
- **Team colours**: note which team wears which colour in the first sheet.
- **Which end**: note which goal each team defends in the first half; they swap
  at halftime. Use `--half first` / `--half second` to restrict tracking.
- **Halftime**: both pipelines may surface an empty-field segment. Exclude it.
- **YOLO false positives**: the ball detector sometimes picks up sponsor logos
  or white jersey numbers. The stationary-frames filter (default 3) suppresses
  most of these, but Claude's verify step catches the rest.
- **YOLO misses**: if the ball is partially occluded or very small, detections
  drop. Increase `--fps` to 5 or fall back to Pipeline A for that half.
