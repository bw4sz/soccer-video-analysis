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

Detection is RF-DETR (~1.6 h for a 60-min match, most of it video decoding).

---

## 1. Player identity — build a re-id gallery

The goal is a `gallery.npz` per squad: each player's appearance banked once, so
`identify --method reid` can name a track without reading a jersey number
(OCR only manages ~34% of crops on overhead footage). Full background:
[Enroll](../CLAUDE.md#enroll--carry-a-teams-appearances-instead-of-re-reading-jerseys).

**Two routes, and which one depends on whether the match is processed.**

| | Tracklets (1d) | Frames (1a-1c) |
|---|---|---|
| You see | a 20s clip, every player ringed and numbered | one still frame, boxes pre-drawn |
| You do | pick a name per number | pick a name per box |
| One decision buys | every crop in that lane (~7-30) | one crop |
| Cues available | position, motion, who they're next to | position, neighbours |
| Needs | a `process` run (`tracks.json`) | nothing — runs off raw video |
| Sync size | ~15 MB per window | ~1 MB for 24 frames |

Use **tracklets** when the match is processed — it is far higher yield, and the
motion is what makes a child recognisable. Use **frames** for a match you haven't
processed, or when you want a gallery started in the next ten minutes.

(Renaming folders of track crops used to be a third route; it was removed on
2026-07-28, see the note at the end of this section.)

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

### 1d. Tracklets — one decision per lane

The higher-yield route, and the one to use whenever the match is processed. Each
task is a **20 s clip of play with every lane of your squad ringed and
numbered**, and you pick a name per number — so one decision banks every crop in
that lane instead of one crop per box. It also gives you motion and pitch
position, which is how people actually tell youth players apart and which no
still frame carries.

**Generate (HPC).** Needs `tracks.json`, so a `process` run must exist:

```bash
soccer-vision enroll --run runs/<match_id> \
  --dump-tracklets runs/<match_id>/tracklets \
  --profile examples/profiles/<team>.yaml --team black \
  --window 20 --n-windows 12 --max-lanes 12
```

Writes `clips/window_*.mp4`, `labeling_config.xml`, `label_studio_tasks.json`
and **`tracklets.json`** — the slot→track-id manifest. Label Studio fixes one
labelling config per project but ByteTrack ids differ in every window, so each
window ranks its lanes and hands out slots 1..N. **Lose `tracklets.json` and the
export is uninterpretable.**

Spread beats length. Twelve windows across a full match cost ~144 MB and gave
115 lanes / 4,373 crops; four windows inside one 3-minute clip gave 260 crops
and no measurable gain, because they carried one sun angle and 7 of 11 players.

**Sync down (laptop).** The task paths are relative to the run directory
(`?d=tracklets/clips/...`), so mirror that layout exactly — the document root is
the folder *containing* `tracklets/`, and the folder must keep its name:

```bash
mkdir -p ~/soccer-annotation/runs/<match_id>
rsync -avP b.weinstein@hpg.rc.ufl.edu:/orange/ewhite/b.weinstein/soccer-video-analysis/runs/<match_id>/tracklets/ \
  ~/soccer-annotation/runs/<match_id>/tracklets/

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=$HOME/soccer-annotation/runs/<match_id>
label-studio start
```

Renaming the folder on the way down (`saints-u14g-tracklets/`) silently breaks
every video: the paths say `tracklets/clips/...` and nothing resolves. Rebuild
with `--serve-url`/`--serve-root` if you want a different layout.

**In the UI**, in this order:

1. **Create Project** → *Labeling Setup* → *Custom template* → paste
   `labeling_config.xml`.
2. **Settings → Cloud Storage → Add Source Storage → Local files**, path =
   the directory above, then **Sync**. Video tasks are *served* from disk, not
   uploaded, so this is required rather than a fallback — without it the player
   comes up blank.
3. **Import** → `label_studio_tasks.json`.

> **Check the task count matches your window count.** Local-files sync creates
> its own task per object it finds, so a 12-window export that reports 13+ tasks
> has swept in the config/manifest files or duplicated the clips. Those
> storage-made tasks carry only a video path — they lose the `onscreen` string
> and the `slots` block that the tasks JSON provides, and `onscreen` is what
> tells you a 20 s window holds ten lanes but rarely three at once. Set the
> storage **File Filter Regex** to something like `.*\.mp4$`, or leave Sync
> alone and let the imported JSON be the only source of tasks.

**Labelling.** The chip number is *not* a jersey number — slots are numbered in
the order players first appear. `not ours` (opponents, referees, spectators) and
`unsure` are first-class and enrol nothing: a guess banks the wrong appearance
under a real player's name, which is worse than a gap. Expect a referee or a
sideline figure to be ringed occasionally; that is what `not ours` is for.

**Enrol (HPC).** Drop the export back beside the manifest:

```bash
rsync -avP ~/soccer-annotation/runs/<match_id>/tracklets/annotations.json \
  b.weinstein@hpg.rc.ufl.edu:/orange/ewhite/b.weinstein/soccer-video-analysis/runs/<match_id>/tracklets/

soccer-vision enroll --run runs/<match_id> \
  --from-tracklets runs/<match_id>/tracklets/annotations.json \
  --out galleries/<team>.with-tracklets.npz --append
```

**Enrol to a new file, not over your working gallery, and A/B before adopting
it.** `--max-samples` caps crops per lane because a lane is many crops but few
*views*, and a batch can deepen an imbalance while looking like progress —
the first one took the U14G gallery 47 → 299 exemplars and moved leave-one-frame-out
19/45 → 17/45. Measure with `slurm/validate_reid_frames.py` on a fixed held-out
set, keeping `galleries/*.frames-only.npz` and `*.with-tracklets.npz` side by side.

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

In the UI, same three steps as [1d](#1d-tracklets--one-decision-per-lane):

1. **Create Project** → *Labeling Setup* → *Custom template* → paste
   `labeling_config.xml`.
2. **Settings → Cloud Storage → Add Source Storage → Local files**, path = the
   document root above, then **Sync**.
3. **Import** → `label_studio_tasks.json`.

Step 2 is not optional on current Label Studio: the env var alone doesn't serve
the files, and a project built without it shows a blank video player on every
task. The same task-count caveat applies — see the note in 1d.

If the player is still blank, `LOCAL_FILES_DOCUMENT_ROOT` was probably exported
*after* `label-studio start`; restart it.

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
