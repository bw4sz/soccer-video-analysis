# Annotating your own footage

Two annotation jobs feed the pipeline, and they are **not** the same task:

| | What you label | Where | Feeds |
|---|---|---|---|
| **1. Player identity (re-id)** | which player each detected box is | Label Studio | `gallery.npz` → `identify --method reid` → `--player` clips |
| **2. Events** | what happened in a clip | Label Studio | corrected event labels → action-recognition training |

Both follow the same split: **compute stays on HPC, labelling happens on your
laptop.** The artifacts you actually annotate are small (a 24-frame labelling
export is ~1 MB; clips downscale from 6.1 GB to ~350 MB); the videos and models
they come from are not (a run directory is ~10 GB). So every walkthrough below is
*generate on HPC → sync down → label locally → sync back*.

Paths on HPC are under `/orange/ewhite/b.weinstein/soccer-video-analysis/`;
local paths assume `~/soccer-annotation/` on your laptop. Substitute freely —
the only rule that matters is in [Path portability](#path-portability).

---

## 0. Before you start

**Get the code on both machines.** `enroll` and `identify --method reid` landed
in PR #16, so both ends need `master`:

```bash
git checkout master && git pull
pip install -e .            # nothing extra is needed for re-id labelling
pip install label-studio    # laptop only, for the event pathway
```

Labelling itself needs neither GPU nor models — a laptop clone is only there to
run Label Studio and, optionally, `annotate --export`.

**Each team needs a processed run.** Crops and clips both come out of
`runs/<match_id>/`, so a squad you haven't processed yet (U10B, U14G) needs one
`soccer-vision process` pass first — on HPC, via SLURM, since detection wants a
GPU:

```bash
sbatch slurm/submit_process.sh data/<match>.mp4 <team>-<date> \
  examples/profiles/<team>.yaml
```

That defaults to RF-DETR (~1.6 h for a 60-min match). See
[Detector](../CLAUDE.md#detector--rf-detr-by-default-sam3-opt-in) for when to
opt into SAM3 instead — it is ~6x slower and HF-gated.

For **enrolment only**, you don't need the whole match. A gallery wants a few
good exemplars per player, not coverage — process a 5–10 minute segment per
team and dump crops from that. It turns an hours-long job into a short one, and
the gallery it produces is just as usable.

---

## 1. Player identity — build a re-id gallery

The goal is a `gallery.npz` per squad: each player's appearance banked once, so
`identify --method reid` can name a track without reading a jersey number
(OCR only manages ~34% of crops on overhead footage). Full background:
[Enroll](../CLAUDE.md#enroll--carry-a-teams-appearances-instead-of-re-reading-jerseys).

**Two ways to produce the labels.** Naming boxes on whole frames (1a-1c) is the
recommended one; labelling folders of track crops is the older route, kept below
for when you'd rather accept or reject a whole lane at a time.

| | Frames in Label Studio | Track-crop folders |
|---|---|---|
| What you see | the **full frame**, zoomable, boxes pre-drawn | 4 wide views per lane on a contact sheet |
| What you do | pick a name per box; delete the rest | rename / merge / delete folders |
| Good for | recognising anyone, since you see the whole picture | bulk-accepting a lane you're sure of |
| Yield | ~20 named boxes per frame | one lane per decision |

### 1a. Export frames to label (HPC)

```bash
soccer-vision enroll --run runs/<match_id> \
  --dump-frames runs/<match_id>/label_frames \
  --profile examples/profiles/<team>.yaml --n-frames 24 --min-players 8
```

Writes `frames/*.jpg` (full resolution), `labeling_config.xml` (one label per
roster player plus `unknown`), and `label_studio_tasks.json` with **every
detected player already boxed**. A 24-frame export is ~1 MB and carries ~530
boxes.

Frames are picked one per equal time bin, taking the **busiest** frame in each —
even spacing alone lands on warm-ups and stoppages where three players stand in
one corner. Boxes that balloon across the frame (a known tracker artifact) are
dropped so they don't cover every real player in the UI.

> **Why this beats labelling crops.** A player is ~29 px tall here, so a track
> crop is 14×20 px — you cannot recognise a child from it at any zoom, and
> that's *by design*: it's the size the re-id model trains on. Labelling on the
> full frame separates the two. You identify a person using everything in the
> picture; Label Studio stores the box as percentages, `enroll` converts it back
> to exact pixels, and the model crops to whatever size it wants at enrolment
> time. Covered by a round-trip test in
> [`tests/test_enroll_team_filter.py`](../tests/test_enroll_team_filter.py).

> **Leave `--team` off here.** Filtering to one kit hides lanes the colour
> classifier got wrong, and a missing box is invisible to you — you'd never know
> a player wasn't offered. Better to see every person and delete the ones that
> aren't yours.

### 1b. Sync down and label (laptop)

```bash
mkdir -p ~/soccer-annotation/runs/<match_id>
rsync -avP b.weinstein@hpg.rc.ufl.edu:/orange/ewhite/b.weinstein/soccer-video-analysis/runs/<match_id>/label_frames/ \
  ~/soccer-annotation/runs/<match_id>/label_frames/

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=$HOME/soccer-annotation/runs
label-studio start
```

Create the project from `labeling_config.xml`, import
`label_studio_tasks.json`, then per frame: **name** each box that's one of your
players, **delete** the boxes for opponents, referees and sideline figures.
Anything left `unknown` is skipped at enrolment rather than guessed at. Zoom in
(the control is enabled in the config) — the players are small.

You don't need every box on every frame. A handful of confident names per player
across a few frames is already a working gallery.

### 1c. Enrol the export (HPC)

Export → JSON from Label Studio, copy it back, then:

```bash
soccer-vision enroll --run runs/<match_id> --from-label-studio export.json \
  --out galleries/<team>.npz          # add --append for later matches

soccer-vision identify --run runs/<match_id> --method reid \
  --gallery galleries/<team>.npz --profile examples/profiles/<team>.yaml
```

Both kits matter: a gallery built only from black-away frames will abstain on
every white-home track, so label one match of each and `--append`.

---

### Alternative route — label whole tracks instead

Faster per decision when you're confident about a lane, and it needs no Label
Studio at all. The catch is the one that sent us to frames: you're judging a
lane from wide contact-sheet views, not from the picture itself.

#### Dump crops per track (HPC)

```bash
soccer-vision enroll --run runs/<match_id> \
  --dump-crops runs/<match_id>/label_crops \
  --team black --profile examples/profiles/<team>.yaml \
  --context-pad 30 --context-min 1600 --sheet-tile-height 1080 --sheet-samples 4
```

Writes one folder per ByteTrack lane (`track_0021__ocr20/` — the `__ocr20`
suffix is the jersey number OCR voted, a hint while labelling), plus
`index_*.jpg` review sheets. Only the `--max-tracks 60` longest lanes are
dumped; a full match fragments into ~2,000 lanes (1,663 over 20 frames on the
Saints U11 match), which is far more than anyone will label and unnecessary for
a gallery.

> **Dump one team.** You normally enrol your own squad, not both, so `--team
> black` halves the pile. Kit colour comes from the `teams` block `process`
> stamps into `tracks.json`; a run made before that landed gets classified on the
> spot (300 sampled frames, turf-green pixels masked out so the median isn't
> dragged toward grass) and cached to `track_teams.json`. **It's a coarse
> pre-sort, not ground truth** — the referee's black shorts and dark-jacketed
> spectators land in "black" too, and an opposing sky-blue kit gets named after
> whichever colour your profile declares. You'll delete those off the sheet
> anyway. For a trustworthy `teams` block, re-run `process` with SAM3 masks.

> **Zoom is two knobs.** `--context-pad`/`--context-min` set how wide the window
> is; `--sheet-tile-height` sets what it's rendered at. Raising only the first
> shrinks the player straight back — every tile is scaled to that height, so a
> wider view at 180 px cancels itself out. At full frame
> (`--context-pad 30 --context-min 1600 --sheet-tile-height 1080`) you get the
> whole pitch with the target crosshaired, which is how you actually tell youth
> players apart at ~29 px tall: position, who they're next to, and which way play
> is going. `--sheet-samples` (tiles per track on the sheet) is independent of
> `--max-samples` (crops enrolled per track), so a readable sheet costs no
> gallery coverage. All of this changes the *review sheets only* — enrolled crops
> are unaffected, so re-dumping to retune costs a few minutes.

#### Sync down (laptop)

```bash
mkdir -p ~/soccer-annotation/<match_id>
rsync -avP b.weinstein@hpg.rc.ufl.edu:/orange/ewhite/b.weinstein/soccer-video-analysis/runs/<match_id>/label_crops/ \
  ~/soccer-annotation/<match_id>/label_crops/
```

~62 MB for 60 tracks at full-frame sheets (the crops themselves are ~1 MB of
that; the 15 review sheets are the rest). Drop `--sheet-tile-height` to 720 if
you want it nearer 25 MB.

#### Label by renaming folders (laptop)

Open `index_001.jpg` … side by side with the folder list, then:

- **Rename** each folder you recognise to the player's name —
  `track_0021__ocr20/` → `Simon Weinstein/`. Match the roster names in the
  team profile (`examples/profiles/saints-u11.yaml`) so `--player` resolves.
- **Merge** several lanes into one player's folder when they're the same child.
  This is encouraged — more poses per player makes a better gallery entry.
- **Delete** everything you can't identify, plus opponents, referees and
  sideline figures. The tracker picks up spectators behind the barrier and the
  ref in yellow; enrolling either poisons the gallery.
- **Leave** anything you're unsure about as `track_*`. Folders keeping that
  prefix are skipped, so a half-finished pass enrols only what you named.

Aim for every player on your squad appearing in at least one folder. Both kits
matter: a gallery built from black-away tracks will abstain on every white-home
track, so enrol one match of each and `--append`.

#### Sync back and enrol (HPC)

```bash
# laptop
rsync -avP --delete ~/soccer-annotation/<match_id>/label_crops/ \
  b.weinstein@hpg.rc.ufl.edu:/orange/ewhite/b.weinstein/soccer-video-analysis/runs/<match_id>/label_crops/

# HPC
soccer-vision enroll --run runs/<match_id> \
  --from-crops runs/<match_id>/label_crops \
  --out galleries/<team>.npz          # add --append for later matches

soccer-vision identify --run runs/<match_id> --method reid \
  --gallery galleries/<team>.npz --profile examples/profiles/<team>.yaml
```

`--delete` matters on the way back: it's what makes the folders you deleted
locally actually disappear on HPC.

---

## 2. Events — Label Studio clip review

Each task is one extracted clip, **pre-filled** with the pipeline's label (and,
if you ran `soccer-vision describe`, SoccerChat's caption + class) so you
*confirm or correct* one choice per clip instead of labelling from scratch. The
choice list is generated from
[`events/labels.py`](../src/soccer_vision/events/labels.py), so the UI never
drifts from the taxonomy the pipeline emits.

```
process ──► clips/ + annotations.json
              │
   describe   ▼ (optional, GPU)     annotate --run
  SoccerChat caption + class ──► label_studio_tasks.json + labeling_config.xml
              │                              │
              ▼                              ▼
        soccerchat.json            Label Studio: confirm/correct
                                             │
                             annotate --export ▼
                                  soccerchat_finetune.jsonl  (youth training set)
```

> **Read the pre-fill as "here's a candidate window", not as a prior.** On the
> current full Saints run every one of the 236 events came back `throw_in` — the
> set-piece heuristics gate on a homography that's degenerate on this footage.
> The clips are still worth annotating (they're real moments of play, and the
> corrections are exactly the training signal that fixes this), but expect to
> change most labels rather than confirm them.

### 2a. Build the project (HPC)

```bash
soccer-vision annotate --run runs/<match_id>
# → runs/<match_id>/labeling_config.xml + label_studio_tasks.json
```

`--serve-root` defaults to the `runs/` base — keep it that way, see
[Path portability](#path-portability).

### 2b. Shrink and sync the clips (HPC → laptop)

Clips come out 1920×1080 / 20 s / ~25 MB each — 6.1 GB for a full match, which
you do not want to pull down. Re-encode first (annotation doesn't need the
pixels):

```bash
# HPC
module load ffmpeg
mkdir -p /tmp/<match_id>_720p
for f in runs/<match_id>/clips/*.mp4; do
  ffmpeg -nostdin -loglevel error -i "$f" -vf scale=-2:720 -crf 30 -preset veryfast \
    -an "/tmp/<match_id>_720p/$(basename "$f")"
done
```

```bash
# laptop — mirror the run layout exactly
mkdir -p ~/soccer-annotation/runs/<match_id>/clips
rsync -avP b.weinstein@hpg.rc.ufl.edu:/tmp/<match_id>_720p/ \
  ~/soccer-annotation/runs/<match_id>/clips/
rsync -avP b.weinstein@hpg.rc.ufl.edu:/orange/ewhite/b.weinstein/soccer-video-analysis/runs/<match_id>/{labeling_config.xml,label_studio_tasks.json} \
  ~/soccer-annotation/runs/<match_id>/
```

### 2c. Start Label Studio (laptop)

Clips are served from disk, not uploaded. Point the document root at the
directory that holds all your match folders:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=$HOME/soccer-annotation/runs
label-studio start
```

In the UI:
1. **Create Project** → *Labeling Setup* → *Custom template* → paste
   `labeling_config.xml`.
2. **Import** → `label_studio_tasks.json`.

If the video player comes up blank, the local-files endpoint isn't serving:
check that `LOCAL_FILES_DOCUMENT_ROOT` was exported *before* `label-studio
start`, and register the same directory under **Settings → Cloud Storage → Add
Local Files** (recent versions want the storage declared as well as the env
var), then **Sync**.

Each clip opens with its label pre-selected; click through and fix the wrong
ones. The `notes` box takes a free-text reason — worth using when you correct
something, since that's the record of *why* the detector was wrong.

**If Label Studio is already running** and you have an API token, skip the file
shuffle and push directly:

```bash
soccer-vision annotate --run runs/<match_id> --push \
  --ls-url http://localhost:8080 --ls-key "$LABEL_STUDIO_TOKEN"
```

### 2d. Export corrections → training data

**Export → JSON** in the UI, then:

```bash
soccer-vision annotate --export export.json \
  --clips-root runs/<match_id>/clips \
  --finetune-out youth_soccerchat.jsonl
```

`youth_soccerchat.jsonl` holds `{"query", "response", "videos"}` records — the
schema SoccerChat's training notebooks consume, ready to LoRA-fine-tune toward
youth/Veo footage. See [`../SOCCERCHAT_INTEGRATION.md`](../SOCCERCHAT_INTEGRATION.md).

---

## Path portability

Task files store each clip as a path **relative to `--serve-root`**:

```json
"video": "/data/local-files/?d=saints-u11-sam3-full/clips/clip_001_throw_in_0s.mp4"
```

Nothing in there is HPC-specific. So a tasks file built on HPC works unchanged
on your laptop **as long as the local directory you point
`LOCAL_FILES_DOCUMENT_ROOT` at contains the same `<match_id>/clips/` layout** —
which is why 2b mirrors `runs/<match_id>/clips/` rather than flattening it.
Build with `--serve-root` = the `runs/` base and this takes care of itself.

---

## Known gap: sampling for event *detection*

The flow above only surfaces clips **where the detector already fired**, so it
can correct a wrong label but can never show you a pass it missed entirely.
That's fine for fine-tuning a clip classifier and wrong for training a temporal
action spotter (the FOOTPASS / SN-PCBAS pathway), which needs true event times
across continuous play — including all the events nothing fired on.

The direction: build tasks from a *continuous segment* of the match with Label
Studio's `<TimelineLabels>` control, letting you scrub 10–15 minutes and mark
each action's time span. That's a builder alongside
[`annotate/label_studio.py`](../src/soccer_vision/annotate/label_studio.py), not
a change to it — the clip-review project stays as-is for the SoccerChat loop.
Not built yet.

---

## Notes

- **Labels** come from one source of truth
  ([`events/labels.py`](../src/soccer_vision/events/labels.py)); regenerate the
  config after the taxonomy changes and the UI stays in sync.
- **`goal_kick`** has no SoccerChat equivalent (SoccerNet folds it into
  Kick-off / Ball-out), so `describe` marks such clips `PLAUSIBLE`, not
  `CONFIRMED`. Those are the clips worth annotating first.
- **Annotating on HPC instead.** If you'd rather not sync, run
  `label-studio start --host 0.0.0.0` on a compute node and tunnel:
  `ssh -L 8080:<node>:8080 b.weinstein@hpg.rc.ufl.edu`. Workable, but video
  scrubbing over the tunnel is slow enough that syncing down usually wins.
- **SoccerChat captions** (optional, GPU) before building the project:
  `sbatch training/slurm/soccerchat_describe.sbatch runs/<match_id>` (append a
  clip count for a smoke test).
