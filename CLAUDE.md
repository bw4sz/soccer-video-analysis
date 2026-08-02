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

### Step 1 — Run ball tracker

```bash
pip install ultralytics   # one-time

python track.py --video match.mp4 --fps 3

# Options:
#   --fps 3              sample rate (default 3 fps — fast, CPU-only)
#   --stationary-frames 3  consecutive stationary detections needed
#   --stationary-px 40   max pixel drift to count as stationary
#   --half first/second  restrict to one half if known
#   --half-frame N       frame where second half starts
#   --model n/s/m        yolov8 size (n = fastest, default)
#   --device cpu/mps     cpu works fine
#   --homography-cache   legacy; leave unset (see the caveat below)
```

> **`track.py` is a legacy standalone script**, not part of the `soccer-vision`
> package, and its goal-zone logic depends on the Hough homography that
> **does not work on Veo footage** — so its `[left]`/`[right]` zone labels and
> any `field_x_m` / `field_y_m` it writes are unreliable. Field registration was
> removed from the real pipeline for exactly this reason (see *Field registration*
> below); `register.py` and the KpSFR homography cache it fed are gone. Treat
> Pipeline B's candidates as "the ball sat still here" and let Claude's verify
> step in Step 2 decide whether it's a goal kick.

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

## Detector — RF-DETR

`process` detects players and the ball with **RF-DETR**
(`julianzu9612/RFDETR-Soccernet`, wired in `detection/rfdetr.py`). It is the only
detector; SAM3 was evaluated and **removed on 2026-07-28** (see below). Submit a
match with `slurm/submit_process.sh`, which defaults to
`examples/process_match.yaml`.

```bash
sbatch slurm/submit_process.sh data/<match>.mp4 <match-id> examples/profiles/<team>.yaml
```

**Measured behaviour**, across all four of our cameras plus four harvested youth
clips (jobs 38223174 / 38223284):

| | RF-DETR |
|---|---|
| Weights | public, no HF gate |
| s/detection-frame, 1080p / 720p | **0.045** / 0.038 |
| Scaling in object count | flat from 15 to 30 objects/frame |
| Full 60-min match | ~1.6 h, **mostly video decoding** (~0.2 h is detection) |
| FOOTPASS broadcast F1 @IoU 0.5 | 0.902 |
| FOOTPASS broadcast recall | 0.977 |

**Detect on every frame. `--detect-fps` is a speed knob with a measured accuracy
cost, not a free one.** Job 38526638 ran the same 3-min U14G clip at both rates
through identical code:

| | 30 fps | 5 fps |
|---|---|---|
| tracked player-seconds | **4,576** | 2,616 |
| share of tracked time in lanes ≥10 s | **52.7%** | 20.9% |
| longest lane | **114.1 s** | 27.6 s |
| detections/frame carrying a track id | 25.4 | 14.5 |

**Read the lane *count* as a trap**: 30 fps produces *more* lanes (2,169 vs
1,226) with a *shorter* median (0.30 s vs 1.00 s), because it mints 1,262
sub-half-second fragments that together hold 4% of the tracked time. Weighted by
coverage the picture inverts. Lane length is what gates identity — a lane too
short to carry a confident re-id vote can never be named however good the gallery
is — so this is an identity lever as much as a tracking one.

Cost is ~2.2 h for a 60-min match. It was ~4.3 h until `predict_split` stopped
`process` running the full RF-DETR forward twice per frame (once for players,
once inside `detect_ball_position` for the ball); both classes come out of one
pass now, verified byte-identical on 2,169 lanes and 5,394 ball samples.

If a run comes back thin, try `conf_threshold: 0.15`
(`examples/saints-u11-0.15-threshold.yaml`). RF-DETR is also **fine-tunable**
(`rfdetr`'s `train()`; our checkpoint is already a SoccerNet fine-tune), which is
the real lever for overhead/Veo footage — the labelled frames from `enroll` are
training data. Its numbers above are unoptimized; `optimize_for_inference(
dtype=float16)` is an untouched speed lever.

### Why SAM3 was removed — do not reintroduce it without new evidence

SAM3 (`facebook/sam3`) was briefly the default and is now deleted from the repo.
The case for it collapsed on every axis:

- **The count that motivated it never reproduced.** It was adopted on a Veo
  player *count* (5-6/frame for RF-DETR vs 20-22 for SAM3) that was never
  ground-truthed. Job 38162552 measured RF-DETR at 20.4 detections/frame against
  SAM3's 13.3 on the same clip; the figure had conflated two different videos.
- **RF-DETR is the better detector where ground truth exists** — F1 0.902 vs
  0.835, recall 0.977 vs 0.925 on FOOTPASS broadcast (job 38133841).
- **It cost 29-39x per frame on every video tested**, 1.08-1.70 s/frame against
  RF-DETR's 0.038-0.046, projecting to 5.4-8.5 h per match against ~0.2 h. The
  gap is a property of the models, not the footage.
- **The weights are HF-gated**, so a fresh clone or a new collaborator fails in a
  way no caching fixes — against an ethos of *quick, dirty and easy*.

Its one genuine advantage was a steadier ball track (p95 jump 208px vs RF-DETR's
905px, job 37883252) and longer-lived object ids. Both are worth fixing on the
RF-DETR path rather than paying 30x compute for.

**Two known costs of the RF-DETR path**, both measured on a 3-min U14G Veo clip
(job 38178685, `runs/u14g-smoke-rfdetr`, 2m45s):

1. **Ball jitter — fixed 2026-08-02, see *Ball smoothing* below.** RF-DETR's
   ball is flickery on overhead footage: 84.5% of frames detected, but median
   frame-to-frame jump 54px and **p95 905px** on a 1920px-wide frame. `process`
   used to write `ball_track.json` **raw**, so on-ball spans — the only working
   selection pathway — inherited that jitter. It now gates the flicker out
   (p95 707px → 32px on the validation clip).

2. **Track fragmentation.** ByteTrack ids are ephemeral: **856 lanes** in three
   minutes, median lane length 7 detection-frames (~1.4 s), only 32 lanes
   reaching 50 frames. This is the fragmentation `enroll`/`identify` already
   exist to paper over — merging lanes per player. It costs enrolment nothing now
   that the gallery is built from labelled *frames* rather than from whole lanes,
   but it does mean OCR bootstrapping (`enroll` off `jerseys.json`) has fewer
   long lanes to vote on.

Both are worth fixing on the RF-DETR path, where the fixes are cheap and reusable.

### Team colour without a segmentation mask — fixed, and how

A third cost showed up on the same clip and has been dealt with, but the reasoning
is worth keeping because it will resurface on any new venue.

RF-DETR returns boxes, not the per-player mask `sample_jersey_bgr` had been
using to sample kit colour from player pixels only. The bbox fallback averaged
kit with turf and shadow and stamped **624 tracks `black` against 38 `white`** —
`--team` and every kit-aware query were simply wrong. Two things were going on,
and only the second one matters:

1. **Turf contamination** — real, and fixed by rejecting grass-hued pixels
   (`turf_pixels`) inside the torso window. Worth doing, but it was the minor part.
2. **Shadow** — the actual cause. Under a low sun the local turf ranges over
   L\* 50–101, so **a white kit in shade is darker than a black kit in sun**. No
   amount of turf rejection helps: absolute lightness is not the kit's property.

The fix is to judge a player against the grass they are standing on
(`estimate_local_illuminant` samples a turf ring around the box). A dark kit
reflects less than that grass, a light kit more, so the *sign* of
torso-minus-turf lightness names the team. That gave **419 black / 243 white**
where the old path gave 624/38.

**Clustering cannot find this boundary, so don't try.** The relative-lightness
histogram is unimodal with a long sunlit-white tail — the two kits abut rather
than separate — and both k-means and Otsu cut at +53, isolating 12 bright shirts
out of 188. Zero is the boundary for a physical reason, not a statistical one.

It applies only when the profile's declared kits **straddle** the turf in
lightness (`lightness_split_kits`): black/white and blue/white qualify, red/blue
does not, and there the code falls back to colour clustering, where hue separates
them. Tracks that never see grass (about 1 in 662) are placed by nearest cluster
colour. `process` prints which route it took as `Team split by:`.

---

## Ball smoothing — gate the flicker, don't model the motion

`process` now writes a **gated** `ball_track.json`
(`soccer_vision.tracking.ball_smooth`). `--no-smooth-ball` restores the old raw
file; the track records `"smoothed": true` either way so a run is never
ambiguous.

**The raw track is bimodal, and that is the whole story.** On the 30 fps U14G
clip the median step between frames is 4px — the ball really does move smoothly —
but **18% of steps exceed 70px/frame**, which is impossible (a 30 m/s ball is
~35px/frame on this framing). Looking at those frames, the detector has latched
onto a white boot, a jersey number, a line marking, or someone in the far crowd.
The excursions are **short**: median 2 frames, longest 0.6s, and 219 of 470
episodes are a single frame.

**A causal filter cannot exploit that, which is why `ball_kalman` barely helped.**
It must decide *at* the excursion whether the ball relocated or the detector
lied, and the evidence arrives afterwards; its `reacquire_after` guess re-locks
onto a 3-frame flicker. Measured on the same clip, with a reference set of
detections independently within 60px of their local median:

| method | p95 step | >70px/frame | coverage |
|---|---|---|---|
| raw (what `process` used to write) | 707px | 18.2% | 86.0% |
| causal Kalman (`ball_kalman`) | 322px | 8.3% | 86.0% |
| **median gate (`ball_smooth`)** | **32px** | **1.4%** | 78.4% |

Reproduced on the full 60-minute match (657px → 46px, 18.1% → 3.1%).

**The motion model is worth nothing here — do not add one back.** A
Rauch-Tung-Striebel smoother on top of the gate made every metric *worse*
(median step 2.7px → 4.5px: it injects wobble between detections). Rejecting
outliers is the entire gain.

**It is a pure post-process**, so an existing run is fixed in seconds rather
than by re-running ~2h of detection — and the raw file is kept:

```bash
python scripts/smooth_ball_track.py runs/<match> --dry-run   # report only
python scripts/smooth_ball_track.py runs/<match>             # → ball_track.raw.json backup
```

**What it costs, and what is still open.** Coverage falls ~8pp because rejected
detections are dropped rather than replaced (short gaps ≤0.5s are interpolated;
longer ones stay `visible: false`, or trimming would lose real dead time). About
**6% of rejections are consistent with a straight-line continuation of the
ball's velocity**, i.e. some genuine fast motion is cut with the flicker. Fixing
that needs the detector to emit **more than one ball candidate per frame** —
`soccer_vision.detection.ball.ball_position_from` keeps only the argmax, so when
the top box is a jersey number the real ball is thrown away before any filter
sees it. With top-k this gate becomes a shortest-path problem over candidates,
which is the principled version. Note also that pixel speed includes **camera
motion** (Veo pans and zooms), so a step is never purely the ball's.

**5 fps is much harder and stays that way**: 41.6% → 11.2% unphysical, coverage
84.5% → 62.2%, because the ball genuinely moves ~200px between samples. One more
reason to detect on every frame.

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

## Identify — read jersey numbers for the individual-player pathway

There are two ways to slice a processed match into clips:

- **Team-level** — *"the black team building out of the back."* Uses jersey
  **colour** (already assigned by `process`); filter with `--team black`. No OCR
  needed.
- **Individual-player** — *"all the passing actions by number six"*,
  *"highlights from number six."* Needs stable player identity, which comes from
  **reading the jersey number** off each track. This is what `identify` adds.

Note that the *action* half of both examples (*"throw-ins"*, *"passing
actions"*) is not detectable today — see *Field registration* and *On-ball
spans* below. `--player` / `--number` / `--team` all work; `--events <label>`
has nothing to match against yet.

A ByteTrack id is an ephemeral lane, not a person — one player fragments into
many lanes across a match, and the lane number is unrelated to the jersey.
`identify` reads the number so `--player <name>` / `--number <n>` can select the
*person*.

```bash
pip install 'soccer-vision[identify]'   # PARSeq recognizer via torch.hub

# Read jersey numbers on an already-processed run (writes jerseys.json):
soccer-vision identify --run runs/<match_id> --profile team.yaml

# Then cut clips for a player by name (needs the profile roster) or number:
soccer-vision reel --run runs/<match_id> --player Simon --profile team.yaml --halo
soccer-vision extract --run runs/<match_id> --events pass --number 6
```

**How it works.** Per-frame OCR flickers (motion blur, players facing away,
angled digits), but a jersey number is constant across a track. So `identify`
reads the number on every legible frame of each track, then takes a
**confidence-weighted majority vote** per track (`soccer_vision.identify.vote`),
writing `unknown` when support is weak or the vote is split — better to abstain
than mislabel a clip. It's a separate opt-in step (not part of `process`) so it
stays re-runnable and doesn't slow every run.

`jerseys.json` records, per track id: the voted `jersey` (or `null`),
`confidence` (winner's share of vote weight), `n_obs` (legible reads),
`legible_frac`, and the roster `name` when a `--profile` is given. `--player` /
`--number` union *all* lanes voting that number, so a player fragmented across
several ByteTrack lanes is still fully selected (and each lane halos correctly).

Key options: `--max-samples 40` (frames sampled per track), `--min-votes 3`
(legible reads required to name a track), `--min-share 0.5` / `--min-margin 0.15`
(vote-confidence guards), `--model <hf-id>` (override the recognizer checkpoint —
e.g. a jersey-fine-tuned PARSeq).

**Caveat (Veo / overhead cameras).** Players are small and often seen from
behind, so many crops carry no legible number; expect a meaningful fraction of
tracks to come back `unknown`. Validate yield on a sample before relying on
per-player selection — where OCR can't read a number, fall back to `--team` or a
raw `--track <id>`. Better still, enrol the squad once and skip OCR — see below.

---

## Enroll — carry a team's appearances instead of re-reading jerseys

For a team you film every week, jersey OCR is the wrong tool to lean on: it
re-derives identity from scratch each match, and on Veo footage it can only read
a number on ~34% of crops. `enroll` banks each player's **appearance** once into
a gallery you carry between matches, and `identify --method reid` then names a
track by nearest-neighbour lookup in that gallery — no legible number required,
so it still works on backs, blurs and distant figures.

**Weights.** Crops are embedded by
[sportsreid](https://github.com/shallowlearn/sportsreid)'s `OSNet_x1_0` (MIT;
2nd place, SoccerNet 2022 re-ID challenge; 83.4 mAP / 78.0 rank-1), fetched from
its Google Drive on first use into `~/.cache/soccer_vision/reid/`. The published
checkpoint is ~1 GB because it carries a 161k-identity classifier head; we strip
that to a ~9 MB backbone and cache it, since those identities aren't our players
— **our** identities live in the gallery, not the weights. No extra install: the
`identify` extra isn't needed for the re-id path, only for the OCR fallback.

```bash
# Label a squad by hand — export frames, name the boxes in Label Studio, enrol.
soccer-vision enroll --video data/<match>.mp4 --dump-frames label_frames/ \
                     --profile team.yaml --n-frames 24
#   → label_frames/{frames/,labeling_config.xml,label_studio_tasks.json}
#   label on a laptop, drop the export back beside the frames, then:
soccer-vision enroll --from-label-studio label_frames/annotations.json \
                     --profile team.yaml --out galleries/saints-u11.npz

# Or cold start from OCR: read numbers once, enrol from the reads it got right.
soccer-vision identify --run runs/<match> --method ocr --profile team.yaml
soccer-vision enroll   --run runs/<match> --profile team.yaml \
                       --out galleries/saints-u11.npz --exclude-jersey 1

# Every match after: no OCR needed.
soccer-vision identify --run runs/<next> --method reid \
                       --gallery galleries/saints-u11.npz --profile team.yaml

# Top up the gallery after each match (it improves all season):
soccer-vision enroll --run runs/<next> --append --out galleries/saints-u11.npz
```

**Three enrolment sources.**

0. **Label Studio tracklets** (`--dump-tracklets` → label → `--from-tracklets`) —
   the highest-yield route, and the one to reach for when a gallery is thin.
   Renders windows of play with every tracked player ringed and numbered, and
   asks for a name per number; **one decision harvests every crop in that lane**
   instead of one crop per box. It also gives the annotator motion and pitch
   position, which is how people actually tell youth players apart ("Morgan
   plays centre mid") and which no still frame carries. Needs a `process` run
   for `tracks.json`. See *Tracklet labelling* below for the two things that
   bite: slot numbers are per window, and a lane's crops are near-duplicates.
1. **Label Studio frames** (`--dump-frames` → label → `--from-label-studio`) —
   the no-tracking route: works straight off a raw video with no `process` run,
   which is why it exists. Every detected player arrives pre-boxed and labelled
   `unknown`; you name the ones who are yours and delete the rest. Labelling
   happens on the **full frame**, which is the whole point: an overhead camera
   renders a player in about 50×21 px, so a crop in isolation is unnameable at
   any zoom, while on the frame you have position, neighbours and the direction
   of play to go on. The box round-trips as percentages and converts back to the
   detector's exact pixels, so the model still crops at the size it trains on.
   Enrolment reads the JPEGs sitting beside the export, so it needs **no
   processed run and no video** — `--dump-frames` works straight off `--video`.
   Boxes left `unknown` are skipped rather than banked under a shared identity.
2. **Bootstrap from OCR** (`--run`, no other source) — reuse the high-confidence
   votes in `jerseys.json` (`--min-confidence 0.8 --min-obs 5`). Free, but a
   confident-wrong read enrols the wrong player, so pass `--exclude-jersey 1`
   (PARSeq's hallucination class on this footage).

Labelling folders of track crops (`--dump-crops` / `--from-crops`, with
`index_*.jpg` review sheets) was the third route and is **gone** — deleted
2026-07-28. It asked people to name a player from a contact sheet of wide
context tiles instead of from the picture itself, and every knob added to make
those tiles readable (`--context-pad`, `--sheet-tile-height`) was working around
the fact that the crop is the wrong thing to look at. Label Studio on frames
does the same job better; don't reintroduce it.

### Tracklet labelling — one decision per lane

```bash
# Render windows of play with every lane of our squad ringed and numbered:
soccer-vision enroll --run runs/<match> --dump-tracklets runs/<match>/tracklets \
    --profile examples/profiles/<team>.yaml --team black \
    --window 20 --n-windows 8 --max-lanes 12

# Label in Label Studio, drop the export back beside the manifest, then:
soccer-vision enroll --run runs/<match> \
    --from-tracklets runs/<match>/tracklets/annotations.json \
    --out galleries/<team>.npz --append
```

**Labelling one stretch completely, to measure track linking instead.** The
defaults above sample the match, which is what a gallery wants. Asking whether
the linker rejoined a player needs the opposite — one contiguous stretch with
nothing left out:

```bash
soccer-vision enroll --run runs/<match> --dump-tracklets runs/<match>/link_gt \
    --profile examples/profiles/<team>.yaml --team black \
    --at 1200 --window 20 --n-windows 1 --all-lanes --max-lanes 12

python slurm/eval_link_ground_truth.py --run runs/<match> \
    --tracklets runs/<match>/link_gt
```

`--at SEC` pins the windows (with `--n-windows` they run back to back from
there). `--all-lanes` rings *every* lane, paging them `--max-lanes` at a time
across several passes over the same footage — truncating is not a neutral sample
here, because the lanes it drops are the short ones linking exists to join, so a
recall number off a truncated window flatters the linker exactly where it is
weakest. Pages are longest-lane-first, so stopping early leaves a gap of known
size. Two lanes labelled with the same name are the same player, which is the
ground truth `eval_link_ground_truth.py` scores precision and recall against.

**Slot numbers are per window, and the manifest is not optional.** Label Studio
fixes its labelling config for a whole project, but ByteTrack ids differ in every
window — so a config naming real track ids cannot exist. Each window ranks its
lanes and hands out slots 1..N, the config declares N dropdowns once, and
`tracklets.json` records what each slot meant. Lose that file and the export is
uninterpretable; `--manifest` points at it if it isn't beside the export.

**A lane is worth many crops but few *views*.** Median lane on the U14G RF-DETR
run is 7 detection-frames (~1.4 s), so its crops are one pose in one light —
`--max-samples` caps how many are taken per lane, because 129 near-duplicates
from one lane would swamp a gallery built from a dozen genuine views. The answer
to a thin gallery is **more windows and more videos**, not longer ones, which is
why `--n-windows` spreads evenly across the match rather than seeking out busy
passages.

`not ours` and `unsure` are first-class options in every dropdown and enrol
nothing. Both exist so an annotator clearing a form never has to guess — a guess
banks the wrong appearance under a real player's name, which is worse than a gap.

Slots are numbered **in the order they first appear**, not by lane length. Ranking
by length is still how the lanes are *chosen* — the longest carry the most crops —
but numbering by it scattered the sequence (slot 10 on screen at 0.0 s, slot 1 at
1.8 s), and an annotator scrubbing through reasonably read that as broken. Each
task also carries an `onscreen` string ("Player 1: 0-11s · …") because a 20 s
window holds ten lanes but rarely three at once, and it names the slots the clip
doesn't use so a six-lane window against ten dropdowns doesn't look wrong.

**Measured: 260 tracklet crops did not improve the gallery.** First real batch
(4 windows, 13 lanes named, `runs/u14g_tracklets`, 2026-07-29) enrolled 260
crops, taking the U14G gallery from 47 to 299 exemplars. Leave-one-frame-out on
the *same* 45 held-out crops: 19/45 before, 17/45 after — no measurable change
(that difference is inside noise at n=45), and precision when named went 5/6 to
4/6. Two reasons, both fixable:

- **The crops came from one 3-minute stretch** (`u14g_smoke180.mp4`, cut from
  15:00), so one sun angle and one patch of pitch. The gallery was short of
  *views*, and this added almost none.
- **It deepened the imbalance.** Only 7 of 11 players appear in those windows,
  and mostly the already-dominant ones: the gallery went to Morgan 64 /
  Eveleigh 64 / Morrighan 48 / Gia 45 against Ila 5, Leire 4, Izabelle 2,
  Lainey 1, Riley 1 — and Morgan was already the attractor in 13 of 26
  confusions.

So the workflow works and the yield is real (13 decisions → 260 crops), but
**crops from one window are not the constraint**. Spread windows across a full
match and across matches — which needs a full `process` run, not a smoke clip.
Keep `galleries/*.frames-only.npz` and `*.with-tracklets.npz` side by side and
A/B any new batch on a fixed held-out set before adopting it.

**Measured: spreading windows across the full match didn't help either
(2026-08-01).** That "one sun angle" diagnosis was tested and **does not hold**.
A second batch (`runs/saints-u14g-full/tracklets`, 12 windows spread evenly over
the whole 60-minute match, 11 annotated, 31 lanes named) enrolled **620 crops
covering all 11 players** — the diverse, balanced batch the first one wasn't.
Same 45-odd held-out frame crops, `slurm/ab_gallery_fullmatch.py`:

| gallery | exemplars | rank-1 (no abstention) | precision @0.05 |
|---|---|---|---|
| A: frames only | 47 | 19/45 (42%) | 5/6 |
| B: + smoke-clip tracklets | 307 | 17/45 (38%) | 4/6 |
| C: + full-match tracklets | 667 | 17/46 (37%) | 4/8 |
| D: + both | 927 | 20/46 (43%) | 4/4 |

**A 20x bigger gallery bought nothing.** Every row sits inside noise of 42%, so
the ceiling is a property of the embedding, not of how much or how varied the
tracklet data is. (Denominators differ by one because Riley, with a single frame
exemplar, only becomes testable once tracklets put her in the gallery.)

**The crops are not the problem — that was checked.** Gallery built from the
full-match tracklets *alone* names a frame query 16/46 (35%) against 9% chance,
and a tracklet crop matched against other tracklet frames scores 106/120 (88%,
inflated by same-lane near-duplicates but conclusive that boxes, labels and
frame-seeking are all aligned).

So **do not spend more annotation effort on tracklets hoping to cross 42%.**
Filed as issue #25.

**And the scoring lead is now closed too (2026-08-02).** "The correct player is
in the top-3 exemplars 62% of the time" looked like signal the scoring wasn't
extracting. It isn't. Same 45-46 held-out crops, same embeddings, only the
scoring rule varied (`slurm/../scratchpad/reid_scoring.py`):

| rule | frames-only | + tracklet crops |
|---|---|---|
| current (mean of top-3) | 19/45 (42%) | 17/46 |
| top-1 | 19/45 | 17/46 |
| player centroid | 18/45 | 22/46 |
| hubness centering | 16/45 | 21/46 |
| closed-set Hungarian (one identity per frame) | 19/45 | 17/46 |
| k-reciprocal re-ranking (Zhong CVPR'17) | 20/45 | 17/46 |
| k-reciprocal + Hungarian | 21/45 (47%) | 17/46 |

Everything sits inside noise of 42%. **Hubness correction and re-ranking are the
two standard fixes for exactly this symptom and they buy nothing**, and neither
does exploiting the closed-set constraint. That leaves one lead: a backbone
fine-tuned on these players rather than one trained to separate people by
clothing. [PRTreID](https://github.com/VlSomers/prtreid) (part-based, jointly
trained for re-id + team + role) is the candidate.

Read the ceiling narrowly, though — it is about telling *teammates* apart, and
that is not the biggest identity loss in the pipeline. See *Identity coverage*
below.

**Nicknames.** A roster entry may carry `nickname: Mo`, which replaces the first
name in the annotator's label list — a squad clicking "Mo" twenty times a frame
shouldn't have to translate "Morrighan" each time. Enrolment maps it back to the
full name, so the gallery is keyed consistently and `--player Mo`,
`--player Morrighan` and `--number 21` all reach the same person. A label that
resolves to nobody on the roster is still enrolled (it may be a hand-typed
opponent) but is **reported at enrolment** — usually it means a nickname is
missing from the profile, which would otherwise split one player into two
gallery entries.

**Config / fallback.** `identify --method` takes `auto` (default — `reid+ocr`
when a gallery is present, else `ocr`), `ocr`, `reid`, or `reid+ocr`. `reid+ocr`
matches on appearance first and sends the tracks the gallery *abstained* on to
OCR: the gallery can't name a player it never enrolled (an opponent, a referee),
and abstaining is deliberate — mislabelling a clip is worse than leaving it
unnamed. Thresholds are `--min-similarity` (0.5) and `--min-reid-margin` (0.05,
the winner's lead over the runner-up). Settle them once in the profile and drop
the flags:

```yaml
reid:
  gallery: galleries/saints-u11.npz
  min_similarity: 0.5
  min_margin: 0.05
```

`jerseys.json` gains `source` (`"reid"` / `"ocr"` / `null`) and `similarity` per
track, so which route named a clip is always auditable.

### Always pass `--team` on a match with two squads on screen

**A gallery holds one squad and has no way to answer "none of the above."**
`match_track` scores a crop against our eleven players *only*, so handed a
referee, an opponent or someone on the next pitch it returns whichever of ours is
nearest and the margin test sees an ordinary win. Measured on
`runs/saints-u14g-full` (gallery: 620 black-kit crops of the Saints U14G squad),
naming with no kit gate put **878 of 1595 names on the white kit against 256 on
our own** — the opposing squad, the yellow-shirted officials, and players on the
neighbouring pitch, all confidently named after somebody's daughter. Gia Olson
alone took 511 white-kit lanes to 56 black. `runs/saints-u14g-full-linked/named_white_lanes.jpg`
is a contact sheet of two dozen of them.

Do not read this as a re-id accuracy problem. It is a **missing constraint**: the
kit colour `process` already stamps into `tracks.json` settles it for free, and a
lane in the opponent's colours cannot be one of our players whatever the
embedding thinks.

```bash
soccer-vision identify --run runs/<match> --method reid \
    --gallery galleries/saints-u14g.npz --profile <team>.yaml --team black
```

The gate runs **before any model does**, so the excluded lanes cost no re-id
forward passes and no OCR either — it makes the step faster, not slower. Excluded
lanes stay in `jerseys.json` carrying `"excluded": "kit"` (and every lane now
records its `"kit"`), so a lane that went unnamed can always be explained.

**A missing kit is not the wrong kit.** About a third of lanes get no colour at
all — too short, or never seen against grass — and by default those stay
eligible, the same abstention logic the OCR veto uses. `--team-strict` holds them
back too, trading reach for precision (on the U14G run: 7064 eligible lanes
against 3396 strict).

`reid: team: black` works in the profile, but think before setting it — **the kit
is a property of the match, not of the squad.** Saints run black away and white
home, so a profile-level default is wrong half the season. Prefer the flag.

Two things this gate cannot do, both still open. It can't separate our players
from the **neighbouring pitch** when that pitch's squad happens to wear our
colours (issue #21 — no horizontal cut separates them either). And the kit
classifier itself errs: a few plainly black-kit lanes are stamped `white` and are
now excluded, which is the recall this buys its precision with.

### OCR vetoes re-id, it never renames it

`reid+ocr` also sends the tracks re-id *did* name to OCR, to **cross-check**
them (`soccer_vision.identify.crosscheck`). Re-id is right about 42% of the time
on teammates in one kit and confidently wrong the rest, which is how another
child's clip lands in a reel; OCR reads nothing on most crops, but several
high-confidence reads agreeing on a number across one lane are near-proof of what
that shirt says. So they are combined **asymmetrically** — re-id names, OCR is
only ever allowed to *veto*:

- **Conflict** (strong reads back a number that isn't the named player's) — the
  contradiction is **recorded, and the name is kept**. `--drop-on-conflict`
  unnames the track instead. Read *Measured: the veto is 0 for 2* below before
  turning that on.
- **Agree / no evidence** — the re-id name stands. OCR abstaining is the normal
  case and means nothing.

### Measured: the veto is 0 for 2 against hand-verified truth

Drop-on-conflict shipped as the default on 2026-08-02 and was **switched off the
same day**, on the first look at the actual pixels
(`slurm/sample_identity_evidence.py`, sheets and labels in
`runs/saints-u14g-full/identity_evidence/`). Both vetoes in the sample killed a
*correct* re-id name on a high-similarity lane:

| lane | truth (hand-verified) | re-id | OCR |
|---|---|---|---|
| 4623 | **Gia Olson**, #7 plainly on her back | Gia, 0.866 — right | `#4` x10, **best 0.98** — vetoed her |
| 5188 | **Morgan Lobey**, facing camera, number never visible | Morgan, 0.859 — right | `#1` x14, best 0.80 — vetoed her |

**A per-read confidence floor cannot fix this**: lane 4623's wrong `4` was read
at 0.98, above every genuine read on the corroborated lanes. The blurred **7** on
a running player *is* a confident 4 to a scene-text model, which also explains
`#4 x380` across the match — Gia is the most-tracked player and #4 is Morgan.
The cause is upstream: `is_legible` is a grayscale-variance gate, so an empty
chest and a smeared shoulder both reach PARSeq, and PARSeq always returns
something. This is the case for importing
[jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)'s
legibility classifier and pose-based torso localisation, not for tuning
thresholds.

**Where OCR is genuinely better than re-id**, from the same sheets: a sharp,
back-on number reads 0.90–1.00 and re-id often has nothing (lane 6047, `#20`
read 18 times at 1.00). **Agreement is the reliable signal, contradiction is
not** — treat a `crosscheck: "agree"` lane as near-certain identity and prefer
those lanes when building a reel.

**Two constraints from that session that no model can be blamed for:**

- **Guest players exist.** Lane 6047 is one of ours wearing #20 — on no roster,
  in no gallery. So "that number isn't on our roster" is **not** an eligibility
  test for whether a lane is our player, and a squad gallery will always abstain
  on a guest.
- **The keeper wears no number.** Izzy in goal (lane 13) is unreadable by
  construction, in a kit that also differs from the outfield black the gallery
  was enrolled from. Her identity has to come from re-id with the keeper kit
  enrolled, or from pitch position — never from OCR.

Every checked track records `crosscheck` (`"agree"` / `"conflict"` /
`"no_evidence"`) in `jerseys.json`, and a dropped one keeps a `conflict` block
(`reid_name`, `reid_jersey`, `ocr_jersey`, `ocr_confidence`, `n_obs`) plus its
original `similarity`, so no identity vanishes unexplained.

**The veto bar sits far above the bar for naming a track from OCR** (3 reads /
0.5 share / 0.15 margin). Reads are first floored at `--conflict-min-read-conf`
0.7 — low-confidence PARSeq output on this footage hallucinates digits, notably
`1` — and the survivors must number `--conflict-min-reads` 4 and hold
`--conflict-min-share` 0.75 of the weight. `--conflict-exclude-jersey 1` bars a
number from ever vetoing. A wrong veto costs one dropped clip; a missed veto puts
the wrong child in a parent's reel, so this is asymmetric on purpose. The same
keys work in the profile's `reid:` block (`conflict_min_reads`,
`conflict_min_read_conf`, `conflict_min_share`, `conflict_exclude_jersey`).

The pass costs OCR over the re-id-named tracks as well as the abstained ones —
roughly double the OCR work. `--no-ocr-verify` skips it.

**Validation.** `slurm/validate_reid.py` does leave-one-track-out on a processed
run: hold out one ByteTrack lane, build the gallery from the others, and see if
it's named correctly. That's the production case (enrol from past matches, name
a fresh lane), not the trivial one of matching a track to itself. It also checks
**stranger rejection** — numbers seen on exactly one track are held out of the
gallery entirely, where the correct answer is to abstain.

Measured on `runs/saints-u11-sam3-full` (47 lanes, 8 players, labels = the
high-confidence OCR votes, so a proxy for ground truth rather than truth):

| min_margin | named | correct when named | strangers rejected |
|---|---|---|---|
| 0.05 (default) | 83% | 39/39 | 5/6 |
| 0.10 | 68% | 32/32 | 5/6 |
| 0.15 | 34% | 16/16 | 5/6 |

Two things to know. **`min_similarity` is nearly inert** — 0.0 and 0.7 give the
same answer, because re-ID cosine similarities bunch high even between different
people; the *margin* between first and second place is what actually decides.
Tune `min_margin`, not `min_similarity`. And **the abstentions are not noise** —
raising the margin costs recall fast without buying precision, because at 0.05
precision is already 100%. Leave it at 0.05 and let OCR pick up the rest.

Read those numbers narrowly. That test holds out a *lane* and matches it against
other lanes of the same player minutes away in the same match — one player
against 7, with a mean over many crops each.

**The harder test, and it does not pass yet.** On the U14G gallery built from a
Label Studio export (`galleries/saints-u14g.npz`, 47 exemplars over 11 players,
2026-07-28) leave-one-**frame**-out — build the gallery from the rest of the
match, then name a player in a frame they were not enrolled from:

| min_margin | named of 45 | correct when named |
|---|---|---|
| 0.00 (no abstention) | 45 | **19/45 (42%)** |
| 0.02 | 24 | 12/24 |
| 0.05 (default) | 6 | 5/6 |
| 0.10 | 2 | 2/2 |

**42% at zero margin is the ceiling** — that is pure nearest neighbour with
nothing thrown away, so no threshold can do better, and it is not a tuning
problem. Chance is 9%, so the embedding carries real signal and nowhere near
enough of it. The abstention machinery is doing its job (5/6 at the default,
protecting you from a 58% error rate) but 13% recall is not usable.

The deeper reason this is harder than the SoccerNet task the weights were trained
on: re-ID normally separates people by **clothing**, and teammates wear an
identical kit. At ~53x76 px all that's left is build, hair and gait. Expect this
to need far more labelled frames than the broadcast literature implies — and
measure with leave-one-frame-out before trusting a gallery on a new venue.

**What's been ruled out, so nobody re-treads it** (all on the same 45 held-out
crops, `slurm/validate_reid_frames.py`):

- **Capping exemplars per player does not fix the imbalance.** Rank-1 is
  unchanged (19/45 → 19/45 at cap 2, 18/45 at cap 3) and precision drops —
  because capping discards a player's *good* crops as readily as her bad ones.
  Read this narrowly: it rules out the cap, **not** the imbalance, which does
  bite (below).
- **`top_k` trades recall for precision, it doesn't add accuracy.** At
  `min_margin` 0.05: `top_k=1` names 13 at 9/13, `top_k=2` names 11 at 9/11,
  `top_k=3` (default) names 6 at 5/6. Zero-margin accuracy is 19/45 for all
  three. Pick a point on that curve; there is no free win on it.

**How the imbalance actually bites**, since the counts alone don't show it. A
player's score is the **mean of her top-3 exemplars**, so few exemplars means
averaging two excellent matches against one bad one, while a player with seven
draws three consistent-but-mediocre ones. On frame 8952 the query is Leire, her
two nearest exemplars in the whole gallery are Leire at 0.792 and 0.787 — and
she ties Morrighan at 0.701 because her third is 0.524 against Morrighan's
0.728/0.712/0.664. Margin 0.000, abstain. Same on frame 4476, where Catherine's
0.703 rank-1 exemplar leaves her outside the top four *players*. `top_k=1` names
both correctly; that is the 13-vs-6 in the table above, not a coincidence.
Distinguish this from an honest near-miss: on the same frame Morrighan's own
query wins at 0.736 against 0.709 and abstains purely because 0.028 < 0.05.
- **Absolute brightness is not the problem** — dark and bright halves score
  10/22 and 9/23.

**Two leads that did survive.** Rank-1 is 42% but the correct player is in the
**top 3 exemplars 62%** of the time, so there is signal the current scoring
doesn't extract. And accuracy tracks **contrast and box size**, not brightness:
the low-contrast half scores 7/22 against 12/23 for the high-contrast half, same
for small vs large boxes, and the late backlit frames where the black kit
silhouettes (83552-101456) manage 3/13 against 16/32 earlier in the match. On 45
crops those splits are suggestive, not conclusive — worth re-checking on a bigger
annotation set before building on them.

**Caveat.** The gallery is kit- and season-specific. A team with two kits (Saints
run black away / white home) needs both enrolled, or a home gallery will abstain
on every away track — enrol from one match of each and `--append`.

---

## Field registration — removed, and why not to bring it back as-is

**There is no pixel→metres registration in this pipeline.** `registration/`
(Hough, KpSFR, sn-calib), `register.py`, and `events/set_piece.py` were deleted
on 2026-07-27. Nothing produces `field_x` / `field_y` any more; every spatial
question is answered in **pixel space**.

**Why.** Evaluated on 6 frames across the Saints Veo match, both estimators
failed on **all 6**:

- **Hough** (`registration/hough.py`, the only one ever wired in) has no concept
  of "field" — it takes the outermost strong lines, which on this footage are
  apartment rooftops, stadium walls, the horizon and tree lines, never the faint
  pitch lines.
- **DeepLabv3 sn-calib** emitted diffuse per-pixel noise with no coherent line
  masks — broadcast domain shift. Never wired in.
- **KpSFR** was never run; same broadcast-training problem expected.

**It failed destructively, not visibly.** Hough returned `ok=True` with a garbage
matrix, so bad metres flowed downstream wearing a valid-looking type:

| Symptom | Evidence |
|---|---|
| Frame centre projected off-pitch | `(14284, -14)` and `(-196656, -889)` on a 55×36 m pitch |
| Every player rejected as a spectator | detections 20 → 0, 52 track-frames where ~6000 expected (job 37877533) |
| A flood of confident false events | 236 events, 100% `throw_in`, 100% false — into clips, sheets, and pre-filled Label Studio predictions |
| No event could get a team | 0/8 events had a `track_id` (job 37879440) — bogus metres beat the working pixel path |

**Venue-specific killer:** the home turf is a shared multi-use complex with
**blue, red and white** lines from several overlapping pitches painted at once.
Even a perfect line detector cannot say which touchline is *the* touchline. This
is a strong argument against line-identity and keypoint homography here, not just
against these two implementations.

**The direction, if you want to solve it: turf mask, not lines.** The green turf
separates cleanly from buildings, walls, track and trees — HSV green +
morphology + largest connected component → boundary polygon, anchored on
keyframes and propagated between them by frame-to-frame optical flow (the camera
pans and zooms about a roughly fixed point). No lines, no homography, no
pretrained model, no domain shift. That gives on-field/off-field (feeding
`detection/field_filter.py`, which is currently a single horizontal cut — see
below) and rough zones — enough for set-piece detection in pixel space. Defer
metric pixel→metres entirely unless a downstream metric truly needs metres.

A VLM can pre-propose the turf boundary polygon (coarse region tracing is
something it does well) to bootstrap Label Studio annotation; it is not reliable
as a per-frame sub-pixel estimator.

### The on-field cut is a top cut, not a box

Until that lands, `detection/field_filter.py` is all the "on field" there is, and
it is **asymmetric on purpose**. It kept the middle 70% of width *and* height —
a centred rectangle — which does not match how these cameras are set up:

- **The bottom of the frame is our own pitch.** The camera sits at the touchline,
  so the near half of the field runs off the bottom edge. Nobody stands between
  the camera and the pitch.
- **The sides are our pitch too.** A wide Veo frame is one pitch across; other
  matches are *beyond* the far touchline, not left and right of it.
- **The top is where the intruders are** — far-touchline crowd, next pitch, car
  park, trees.

So cutting the sides and bottom discarded real players and removed nothing. Job
38180242 measured the old rectangle dropping **36% of detected people** (22/frame
→ 14/frame) while keeping actual spectators on the far touchline, and every
player box in `runs/saints-u14g-full` is clipped into x ∈ [288, 1632], y ≤ 918 —
which is why a player in the near corner never got a track id and so never
appeared in a tracklet labelling clip to be named.

Defaults are now `--field-top 0.15 --field-sides 0 --field-bottom 0`; `process`
prints the cut it used. The side and bottom knobs remain for a camera set well
back from the touchline. **Runs made before 2026-07-31 carry the old rectangle**,
so a thin edge of the frame is empty in them by construction — don't read that as
the detector missing players.

What the top cut *cannot* do: the far-touchline crowd and the neighbouring match
sit below that line, mixed in with our own far-side players. No horizontal cut
separates them — that is issue #21, not this function.

**Consequences to keep in mind when reading the code:**

- `soccer_vision/pitch.py` holds nominal `FIELD_W_M` / `FIELD_H_M` (55×36) for
  *declaring* pitch size in the OSL export, and nothing else. They are not
  measurements.
- **Every metre-denominated metric was deleted, not left unreachable** —
  `metrics/heatmap.py` (whole module), `metrics/shots.py::is_shot_toward_goal`,
  and `distance_per_player` from `stats.json`. Leaving them importable would
  invite someone to wire them back up to coordinates that don't exist. Pixel
  displacement is not a substitute for the distance figure: the camera pans, so
  a stationary player would accumulate "distance" as the view moves past them.
- What survives in `metrics/`: `possession.py` (pixel space) and
  `shots.py::detect_shots_from_events` (a label filter, coordinate-free).
- Old `runs/` from before this change still contain `field_x` / `field_y`. They
  are ignored, not trusted — `associate.py` reads pixels only.

---

## Identity coverage — the pipeline loses the *name*, not the football

Measured on `runs/saints-u14g-full` (2026-08-02, all from saved artefacts, no
GPU). On-ball spans over every black-kit lane, no identity filter:

| | spans | seconds |
|---|---|---|
| Saints on the ball, detected and tracked | 745 | **1,065 s** |
| …of those, carrying a name | 70 | **75 s (9.4%)** |

**~91% of the touches a parent watches are already detected, tracked,
kit-classified and turned into a clean on-ball span, then discarded for want of a
label.** Detection, the ball track and the on-ball geometry are not the
bottleneck. Identity coverage is. Weigh any proposed work against this number.

The 42% teammate-separation ceiling (issue #25) is real but is *not* the dominant
loss here — only 4.1% of black-kit lanes get named at all. The losses that matter
are coverage ones:

1. **Lane length.** Re-id cannot name a lane too short to carry a confident vote.
   At 5 fps the median black lane was 2.0 s and the longest in three minutes was
   27.6 s; at native rate the longest was 114.1 s and 2.5x more of the tracked
   time sat in lanes over 10 s (job 38526638). **Detection rate is an identity
   lever, not just a tracking one.**
2. **Linking.** A chain inherits the identity of whichever member got named, so
   this is label propagation, not a tracking nicety. Loosening the gate on the
   5 fps run took named on-ball coverage from 7% to 48% — but that sweep had no
   ground truth, which is what `--all-lanes` labelling and
   `slurm/eval_link_ground_truth.py` now supply. Don't adopt a loose gate on the
   strength of the coverage number alone.
3. **Naming the wrong team**, which the kit gate addresses — see above.

**Where the field is.** This whole stack is
[SoccerNet Game State Reconstruction](https://arxiv.org/abs/2404.11335)
([sn-gamestate](https://github.com/SoccerNet/sn-gamestate) on
[TrackLab](https://github.com/TrackingLaboratory/tracklab)), whose GS-HOTA went
29.0 → 63.9 in one year, almost all of it identity rather than detection. Three
components worth importing rather than re-deriving:

- **[gta-link](https://github.com/sjc042/gta-link)**
  ([GTA, arXiv 2411.08216](https://arxiv.org/abs/2411.08216)) — what
  `tracking/link.py` wants to be. A *splitter* (DBSCAN over per-box embeddings,
  catching lanes that contain two people — ours can only merge, never split) and
  a *connector* (hierarchical clustering on appearance plus spatial constraints,
  not greedy edge-to-edge). +3.7 HOTA on SoccerNet-Tracking, and a post-pass on
  saved tracks, which is already our architecture.
- **[Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU)**
  ([arXiv 2306.13074](https://arxiv.org/abs/2306.13074)) — drops the Kalman
  filter, whose linear-motion assumption is what ByteTrack's association rests
  on and what athletes violate. 85.4 HOTA on SoccerNet-Tracking.
  [GTATrack](https://arxiv.org/abs/2602.00484) is both together.
- **[jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)**
  ([arXiv 2405.13896](https://arxiv.org/abs/2405.13896)) — puts a **legibility
  classifier and pose-based torso localisation in front of PARSeq**, then
  aggregates per tracklet. We feed PARSeq every crop including the illegible
  ones, which is where the hallucinated `1`s come from.

Also worth reading before redesigning the linker:
**[SportsSUSHI](https://github.com/mkoshkina/sports-SUSHI)**
([WACV'25](https://arxiv.org/abs/2502.21242)) feeds jersey number, team id and
field position in as *association features* over a hierarchy of graphs spanning
progressively longer gaps — the principled version of "link at 12 seconds" — and
its hockey dataset is a fixed whole-surface camera, i.e. our geometry rather than
broadcast.

### The jumping halo was never a tracking failure — measured 2026-08-02

A player reel with a halo that hops between players, and rings empty grass, looks
like bad tracking. It is not. **The tracking is good and the naming has no
one-player-one-place constraint**, and everything downstream inherits that.

**On `runs/saints-u14g-full`, "Morgan Lobey" is 275 lanes, and on 18.4% of the
frames she is named at, two to four of them are alive at once** — 166 s of the
899 s she is named across, peaking at four. There is one Morgan on the pitch, so
every one of those frames has at least one other child labelled with her name.
`reel --player` unioned all 275 lanes into one halo track and let the earliest
sample win each frame — an arbitrary pick between them, re-made at every frame
boundary. That is the jumping halo, and no clip-selection change fixes it.
`runs/saints-u14g-full/six_morgans.jpg` is one frame with four, two of them on
the *neighbouring pitch*.

**One halo, one player** (`clips/extract.py::halo_samples_for`). Lanes named for
a player are now *candidates*, not members: one joins only if it does not overlap
in time with a lane already accepted, and is near enough that a footballer could
have run there in the gap. This cannot make the halo *correct* — the name it
started from may be wrong — but it makes it coherent, which is the difference
between a viewer seeing one player and seeing a strobe.

**The tracking underneath is genuinely good, at 30 fps.** Seeded from lane 14185
of `runs/saints-u14g-full-30fps` (Morgan, hand-verified through the 5 fps run's
lane 5188), motion linking alone follows her for **23.5 s across a lane handoff
with 96% frame fill and a 6.6 px jump** — verified by eye, crop by crop, not by a
metric. `scripts/follow_player.py` is the tool: seed a lane, chain it, render one
continuous window with one halo and the lane id in the HUD.

**Detect at 30 fps for this, not 5.** The same player at `sample_interval: 6`
fragments where the 30 fps run holds her. The 30 fps run has 104 lanes over 60 s
and a 228 s maximum; 52.7% of its tracked time sits in lanes over 10 s.

### Dedup is a separate pass from linking, and has to run first

`link_tracks` requires `b.first_frame > a.last_frame`, rejecting a successor born
before its predecessor died as "overlapping in time". But that overlap is the
**most common way a long lane ends**: the detector mints a duplicate box on a
player it is already tracking, the box takes a fresh id, and the old lane dies a
frame later. It ends **9.9% of lanes lasting over 10 s** (3.6% of lanes under
1 s — the longer the lane, the more this is what kills it). It is the easiest
link in the file, and the gate refused it by construction.

`tracking/link.py::merge_duplicate_lanes` collapses those pairs, and
`link-tracks` runs it first by default (`--no-dedup` to skip). Morgan's chain
went 12.4 s → 23.5 s; median chain span over 1,003 seed lanes went 37.3 s →
39.7 s with the bad-handoff rate flat (0.50 → 0.45).

Three things that cost a run each to learn:

- **Order matters, and folding the passes together is much worse than either.**
  Unioning dedup pairs into an already-built link result put **5.6 bad handoffs
  on the average chain, one chain collecting 705**, because a single bad merge
  fuses two long chains. Dedup → link keeps it at 0.45.
- **IoU is the wrong same-player test.** The real handoff that breaks Morgan's
  chain pairs a 30x59 box with a 51x83 one — same player, IoU 0.40. IoU punishes
  that scale disagreement twice. Centre distance in box heights (0.4) plus a
  separate scale-ratio test catches it without loosening enough to swallow a
  neighbour.
- **The kit gate is not optional.** Geometry cannot separate one player under two
  ids from two players in contact. Without it, 16.3% of merges joined lanes the
  team classifier had put in *different kits*, one stitching a 62 s white-kit
  lane onto Morgan's black one.

### Do not loosen the link gate on a coverage number — it is verified garbage

Taking the gate from 2 s/150 px to 4 s/250 px grows Morgan's chain from 23.5 s to
66.4 s and the population median from 39.7 s to 45.9 s. **Both numbers are
worthless.** Rendering the 66 s chain shows it is a white-kit player for 60 s,
then the **yellow-shirted referee** for 15 s, then Morgan. At 6 s/300 px it also
picks up a 62 s white lane.

Two traps to know:

- **Handoff jump distance is not a precision metric.** Wrong links happen
  precisely when two people are *close together*, so a swap is a small jump. The
  loose chains scored a 16.7 px median jump while being three different people.
- **The referee is stamped `black`.** The team classifier judges lightness
  against turf, and a yellow shirt in shadow reads darker than the grass — so
  the kit gate that keeps our squad's lanes apart from the opposition's does
  nothing against officials. This is the same failure that put officials in
  `named_white_lanes.jpg`.

**There is no ground truth for link precision, which is why this keeps
happening.** `runs/saints-u14g-full-30fps/link_gt/` holds the `--all-lanes`
tracklet windows staged for exactly this and they are **not annotated**;
`slurm/eval_link_ground_truth.py` is waiting on them. Until they exist, judge a
gate change by rendering the chain and looking at it — `scripts/follow_player.py`
takes about a minute.

### What a rebuilt reel actually costs — job 38562933, 2026-08-02

First `identify` against `runs/saints-u14g-full-30fps`, with the ball track gated
offline (`scripts/smooth_saved_ball_track.py`), lanes deduped and linked, and the
one-lane halo. Against the same match at 5 fps:

| | 5 fps | 30 fps |
|---|---|---|
| lanes / named | 17,395 / 2,224 | 44,350 / **1,450** |
| Morgan named seconds (pre → post link) | 899 → 1,333 | 101 → 198 |
| her frames carrying 2+ lanes of her name | 18.4% → **28.4%** | 0.5% → **1.1%** |
| worst concurrency, any player | 4 → 6 | 2 → 2 |
| Morgan spans / touch seconds | 58 / 41 s | 14 / 22 s |
| reel length | 8.7 min | **2.1 min** |

**The reel is watchable now and the name collision is gone — but not because
anything fixed it.** Re-id simply names far less at 30 fps and what it names is
right, so issue #28 is *masked* here rather than solved. Read the two columns as
a precision/recall trade, not as progress on the constraint.

**A reel's length is a padding policy, not a detection result.** `reel` pads each
span 6 s before and 5 s after and merges within 2 s, so Morgan's **41 s** of
touches on the 5 fps run became an **8.7-minute** reel — 7.9% signal.
`slurm/report_on_ball_coverage.py` prints touch seconds and projected reel
minutes side by side for exactly this reason. Shorten a reel with `--pre` /
`--post` / `--merge-gap`; it is a different problem from finding the moments.

**Gating the ball changes *selection*, not just the jump statistics.** Saints
on-ball time went 2,626 s raw → **1,831 s gated** (-30%) while the span count
went **535 → 694**: the gate breaks long false spans — the ball teleporting onto
a player and staying "on" them — into shorter true ones.

**Coverage is the binding constraint, by a lot.** Ceiling 1,831 s of Saints
on-ball time; named at all 365 s (20%); Morgan **22 s (1.2%)** against a fair
share of roughly 166 s. We catch about **13% of her touches**. And only *one*
named lane in her reel came from re-id directly (the hand-verified 14185, sim
0.845) — every other carries `source: "linked"`, so the reel is name propagation
off a handful of decisions and a wrong seed takes a whole chain with it.

**Within-lane drift is real and is not measurable without labels.** Lane 37964
(50.5 s, 7 of her 14 spans) holds the right black-kit player, **drifts onto an
adjacent white-kit player at 3012 s and 3019 s**, then returns. The kit gate
cannot see it (the lane is stamped black overall) and linking did not cause it —
the raw ByteTrack lane contains two people, the case gta-link's *splitter* exists
for. Counting frames where a black lane's box overlaps a white lane's box at
IoU>0.5 **does not detect it** (lane 37964 scores 0.4%), because when the black
lane grabs the white player, that player's own lane has usually just died — which
is why the grab happened.

**Camera pan is measured and is not the lever it looks like.** This Veo camera
moves 1.35 px/frame at the median and 10.8 px/frame at p99 (~325 px/s), so every
motion model in the stack — ByteTrack's Kalman, the linker's velocity
extrapolation — is fitting camera motion as player motion. Compensating it is
nearly free (the median box displacement per frame is a robust estimate; no
homography, no optical flow, no decode pass — `slurm/eval_follow_leads.py`). But
at the safe gate it buys **+0.8 s of median chain span**, and at loose gates its
apparent gains are the garbage chains above. It is a prerequisite for a longer
gate, not a win on its own.

---

## On-ball spans — the only working selection pathway today

**There is no action detector available on a default run.** The `rules`
set-piece engine was retired with field registration, `learned` has no
checkpoint yet, and `vlm` is opt-in — so `process` detects **zero events** and
`annotations.json` comes out empty. That is expected, and `process` prints as
much rather than leaving a bare "Found 0 events" looking like a bug. Detection,
tracking, team assignment and the ball track are all unaffected.

So when `--player` / `--number` / `--track` matches no detected events, `extract`
and `reel` fall back to **on-ball spans** — the stretches where that player was
close to the ball (`soccer_vision.events.on_ball`). This is pixel-space
geometry over `ball_track.json` + `tracks.json`, so it needs neither the event
detector nor a field homography. In practice this is not a fallback any more but
*the* pathway; it is what makes a player query answerable at all.

**Proximity, not "nearest".** A span opens whenever the selected player is within
`--on-ball-dist` of the ball, *not* only when they are the closest player on the
pitch. Requiring nearest-player silently dropped every contested moment — a
tackle, a challenge, pressing an opponent — because the opponent was fractionally
closer, which is exactly the footage a parent or coach wants. The radius is tight
to pay for that: 200px let in fly-bys where the player was merely in frame, so
the default is 90px, close enough to read as a real touch or challenge.

```bash
# No pass detector yet — this cuts Simon's touches, haloed
soccer-vision reel --run runs/<match_id> --player Simon --profile team.yaml --halo

soccer-vision extract --run runs/<match_id> --number 6 --team black
soccer-vision extract --run runs/<match_id> --player Simon --events pass --on-ball
soccer-vision extract --run runs/<match_id> --player Simon --no-on-ball
```

Spans become ordinary `on_ball` events, so `--team`, `--halo`, and clip naming
all work unchanged. In `reel` the clip window follows the span's real duration
instead of the fixed 20s, so a 1s touch and a 40s dribble don't produce the same
footage.

**Two things it deliberately won't do:**

- **Won't fire when an explicit event label was given.** `--events pass` coming
  back with ball-proximity touches would answer a different question than the one
  asked, so it prints why and stops. `--on-ball` forces it; `--no-on-ball`
  disables the fallback entirely.
- **Won't fire without a player selection.** `--team blue` alone is a team query
  with no lane to anchor spans on.

**Lane handoffs.** A player fragments across ByteTrack lanes and one continuous
touch can cross a handoff. Spans are *not* split there (that would cut one action
into two clips) — each span carries every lane it covers in `track_ids` so the
halo follows through the handoff, with `track_id` being the lane that got closest
to the ball. `--team` filtering works because `process` now stamps each track's
kit colour into the `teams` block of `tracks.json`.

Key options: `--on-ball-dist 90` (max px from ball to the player's feet),
`--on-ball-min-span 0.4` (drop shorter spans as incidental).

**Caveat.** This is proximity, not action recognition — it says #6 was on the
ball, not that #6 *passed*. It's the honest answer available today; once the
learned action engine ships, real `pass` / `shot` events take over and the
fallback stops firing for those queries.

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

The default query list (`soccer_vision.harvest.queries`) leads with elevated
auto-tracking cameras (Veo / XbotGo Falcon+Chameleon / Trace / Pixellot) because
they give the high vantage point wanted for analysis and skew youth/amateur;
general youth queries follow. English queries say "soccer" (never "football") and
an American-football filter (`is_american_football`, incl. the 🏈 emoji) rejects
gridiron uploads that slip through — "football" is soccer in most of the world,
so only US-specific terms trigger it.

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

---

## Community — file issues for open problems, don't just fix and move on

This project grows through outside contributors, and the issue tracker is how
they find in. When we hit a real, scoped, unsolved problem — a known
limitation, a "here's the direction, someone should build this" — **file a
GitHub issue documenting it**, even if we route around it ourselves for now.
Do this proactively, not just when asked.

A good community issue is short and self-contained:
- **What's broken or missing**, with concrete evidence (numbers, a frame, a
  small chart) rather than a vague description.
- **What we already ruled out and why**, so nobody re-treads it.
- **The direction we think is promising**, stated as a lead not a mandate —
  leave room for a contributor to disagree with the approach.
- **Where to look**: exact file paths to the relevant module(s).
- One illustrative image when it makes the problem obvious at a glance (a
  real chart/frame from our own data beats a mockup — see the `dataviz`
  skill for how to build one cleanly).

Tag with `help wanted` and, where the fix is genuinely approachable without
deep repo context, `good first issue`. File with `gh issue create` (see below
for setup).

### GitHub CLI

`gh` is installed at `~/.local/bin/gh` (not via a module — HiPerGator has no
`gh` module, so it's a direct binary download from
`github.com/cli/cli/releases`). Auth lives in `~/.config/gh/hosts.yml`
(personal access token); if `gh auth status` reports an invalid/expired
token, ask the user for a fresh fine-grained PAT (Issues: Read/write,
Contents: Read at minimum) rather than trying to run the interactive
`gh auth login` browser flow, which doesn't work in this non-interactive
environment.
