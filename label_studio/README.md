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

**Enrolment needs no processed run.** `enroll --video ... --dump-frames` detects
on the two dozen frames it exports and nothing else, so a squad you've never
processed (U10B, U14G) can be labelled straight off the raw file in seconds. Going
through `process` for this was a 5-GPU-hour job that never finished (38176317).

**Clips do need one**, so a match you actually want footage from gets a
`soccer-vision process` pass — on HPC, via SLURM, since detection wants a GPU:

```bash
sbatch slurm/submit_process.sh data/<match>.mp4 <team>-<date> \
  examples/profiles/<team>.yaml
```

That defaults to RF-DETR (~1.6 h for a 60-min match). See
[Detector](../CLAUDE.md#detector--rf-detr-by-default-sam3-opt-in) for when to
opt into SAM3 instead — it is ~6x slower and HF-gated.

---

## 1. Player identity — build a re-id gallery

The goal is a `gallery.npz` per squad: each player's appearance banked once, so
`identify --method reid` can name a track without reading a jersey number
(OCR only manages ~34% of crops on overhead footage). Full background:
[Enroll](../CLAUDE.md#enroll--carry-a-teams-appearances-instead-of-re-reading-jerseys).

Labelling is **naming boxes on whole frames in Label Studio** — one route, not a
choice. (Renaming folders of track crops used to be an alternative; it was
removed on 2026-07-28, see the note at the end of this section.)

### 1a. Export frames to label (HPC)

```bash
# From a processed run:
soccer-vision enroll --run runs/<match_id> \
  --dump-frames runs/<match_id>/label_frames \
  --profile examples/profiles/<team>.yaml --n-frames 24 --min-players 8

# Or straight off the video, with no `process` run at all — detection runs on
# the two dozen exported frames only, which takes seconds:
soccer-vision enroll --video data/<match>.mp4 \
  --dump-frames runs/<match_id>_label_frames \
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

**Renaming a label is fine** (a squad's nickname beats a formal first name), but
put the nickname in the profile too — `nickname: Mo` under that roster entry.
Enrolment maps it back to the full name; without it, "Mo" and "Morrighan Wright"
become two players in one gallery. `enroll` prints every label that didn't
resolve to a roster name, so check that line.

### 1c. Enrol the export (HPC)

Export → JSON from Label Studio and **drop it back into the same folder as the
frames** (`runs/<match_id>/label_frames/annotations.json`) — the crops are cut
from those JPEGs, so nothing else needs to be present:

```bash
rsync -avP ~/soccer-annotation/runs/<match_id>/label_frames/annotations.json \
  b.weinstein@hpg.rc.ufl.edu:/orange/ewhite/b.weinstein/soccer-video-analysis/runs/<match_id>/label_frames/

soccer-vision enroll --from-label-studio runs/<match_id>/label_frames/annotations.json \
  --profile examples/profiles/<team>.yaml \
  --out galleries/<team>.npz          # add --append for later matches

soccer-vision identify --run runs/<match_id> --method reid \
  --gallery galleries/<team>.npz --profile examples/profiles/<team>.yaml
```

No `--run` and no GPU: enrolment reads the labelled frames off disk and embeds a
few hundred crops on CPU in seconds. If the frames aren't beside the export, say
where they are with `--frames <dir>`, or let it decode them from the source video
with `--video`.

Both kits matter: a gallery built only from black-away frames will abstain on
every white-home track, so label one match of each and `--append`.

> **Why there's no crop-folder route any more.** `--dump-crops` wrote a folder of
> crops per ByteTrack lane for you to rename, with `index_*.jpg` review sheets so
> you could tell the lanes apart. But a player is ~29 px tall on this footage, so
> every knob it grew (`--context-pad`, `--context-min`, `--sheet-tile-height`)
> existed to pull the *view* back out to a full frame — i.e. to reconstruct what
> Label Studio shows you for free. Removed 2026-07-28 along with `--from-crops`.

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
