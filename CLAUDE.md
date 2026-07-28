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

## Detector — RF-DETR by default, SAM3 opt-in

`process` detects players and the ball with **RF-DETR**
(`julianzu9612/RFDETR-Soccernet`, wired in `detection/rfdetr.py`). SAM3
(`tracking/sam3.py`) is available behind `detector.type: sam3` but is not the
default. Submit a match with `slurm/submit_process.sh`, which defaults to
`examples/process_match.yaml`.

```bash
# default — RF-DETR
sbatch slurm/submit_process.sh data/<match>.mp4 <match-id> examples/profiles/<team>.yaml

# opt into SAM3 (raise --time to 12:00:00 first, or it will not finish)
sbatch slurm/submit_process.sh data/<match>.mp4 <match-id> <profile> examples/saints-u11-sam3.yaml
```

**Why RF-DETR is the default.** Three reasons, in order of how much they matter:

| | RF-DETR | SAM3 |
|---|---|---|
| Weights | public | **HF-gated** (`facebook/sam3`) |
| s/detection-frame, 1080p, L4 | **0.062** | 1.623 |
| Full 60-min match | **~1.6 h** | ~9.4 h |
| FOOTPASS broadcast F1 @IoU 0.5 | **0.902** | 0.835 |
| FOOTPASS broadcast recall | **0.977** | 0.925 |

The gate is the practical one: it makes a fresh clone or a new collaborator's
setup fail in a way no amount of caching fixes, against an ethos of *quick,
dirty and easy*. The speed is the operational one — 9.4 h silently overran an
8 h wall (job 38162800). The quality row is the surprising one: **RF-DETR is
not the weaker detector.** SAM3 was adopted on a Veo player *count* (5-6/frame
vs 20-22) that was never ground-truthed and has since failed to reproduce —
job 38162552 measured RF-DETR at 20.4 detections/frame against SAM3's 13.3 on
the same clip, and the speed profiling saw 22-27/frame on U14G Veo footage.

Speed numbers are jobs 38176330 / 38177148; quality is job 38133841. SAM3's
cost is `0.205s + 38.7ms x n_masklets` — nearly all per-tracked-object, so
resolution is not a lever (640x360 is only 1.2x faster than 1080p) but prompt
choice is, because it changes how many objects get tracked.

**When to reach for SAM3 anyway.** Its real advantage is robustness to domain
shift, not detection quality — RF-DETR is a strong *broadcast* detector, and
the open question is how far it degrades on overhead/Veo footage where no
ground truth exists. If a run comes back thin, try `conf_threshold: 0.15`
(`examples/saints-u11-0.15-threshold.yaml`) **before** switching models.

**Two known costs of the default**, both measured on a 3-min U14G Veo clip
(job 38178685, `runs/u14g-smoke-rfdetr`, 2m45s where SAM3 timed out at 60 min):

1. **Ball jitter.** RF-DETR's ball is flickery on overhead footage: 84.5% of
   frames detected, but median frame-to-frame jump 54px and **p95 905px** on a
   1920px-wide frame. SAM3's `"soccer ball"` prompt gives p95 208px (job
   37883252). `process` writes `ball_track.json` **raw**, so on-ball spans — the
   only working selection pathway — inherit that jitter.
   `soccer_vision.tracking.ball_kalman` exists for exactly this and is *not*
   wired into `process`; see *Trim empty* below, including the caveat that it
   over-rejects at the 5 fps `process` samples at.

2. **Track fragmentation.** ByteTrack ids are far more ephemeral than SAM3's
   masklet ids: **856 lanes** in three minutes, median lane length 7
   detection-frames (~1.4 s), only 32 lanes reaching 50 frames. SAM3 gave 55
   lanes over 600 frames on comparable footage (job 38162552). This is the
   fragmentation `enroll`/`identify` already exist to paper over — merging lanes
   per player. It costs enrolment nothing now that the gallery is built from
   labelled *frames* rather than from whole lanes, but it does mean OCR
   bootstrapping (`enroll` off `jerseys.json`) has fewer long lanes to vote on.

Neither is a reason to go back to SAM3 by default; both are worth fixing on the
RF-DETR path, where the fixes are cheap and reusable.

### Team colour without a segmentation mask — fixed, and how

A third cost showed up on the same clip and has been dealt with, but the reasoning
is worth keeping because it will resurface on any new venue.

Dropping SAM3 also dropped its per-player mask, which `sample_jersey_bgr` had been
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

**Two enrolment sources.**

1. **Label Studio frames** (`--dump-frames` → label → `--from-label-studio`) —
   the way to do this. Every detected player arrives pre-boxed and labelled
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
matches on appearance first and sends only the tracks the gallery *abstained* on
to OCR: the gallery can't name a player it never enrolled (an opponent, a
referee), and abstaining is deliberate — mislabelling a clip is worse than
leaving it unnamed. Thresholds are `--min-similarity` (0.5) and
`--min-reid-margin` (0.05, the winner's lead over the runner-up). Settle them
once in the profile and drop the flags:

```yaml
reid:
  gallery: galleries/saints-u11.npz
  min_similarity: 0.5
  min_margin: 0.05
```

`jerseys.json` gains `source` (`"reid"` / `"ocr"` / `null`) and `similarity` per
track, so which route named a clip is always auditable.

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
`detection/field_filter.py`, which is currently a crude central-rectangle hull)
and rough zones — enough for set-piece detection in pixel space. Defer metric
pixel→metres entirely unless a downstream metric truly needs metres.

A VLM can pre-propose the turf boundary polygon (coarse region tracing is
something it does well) to bootstrap Label Studio annotation; it is not reliable
as a per-frame sub-pixel estimator.

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
