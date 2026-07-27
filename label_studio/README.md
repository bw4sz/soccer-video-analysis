# Annotating your own footage

Two annotation jobs feed the pipeline, and they are **not** the same task:

| | What you label | Where | Feeds |
|---|---|---|---|
| **1. Player identity (re-id)** | which track is which player | a file browser + review sheets — no Label Studio needed | `gallery.npz` → `identify --method reid` → `--player` clips |
| **2. Events** | what happened in a clip | Label Studio | corrected event labels → action-recognition training |

Both follow the same split: **compute stays on HPC, labelling happens on your
laptop.** The artifacts you actually annotate are small (a crops folder is
~2 MB; clips downscale to a few hundred MB); the videos and models they come
from are not (a run directory is ~10 GB). So every walkthrough below is
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
`soccer-vision process` pass first — on HPC, via SLURM, since SAM3 detection is
the expensive step:

```bash
soccer-vision process data/<match>.mp4 --out-dir runs --match-id <team>-<date> \
  --config examples/saints-u11-sam3.yaml
```

For **enrolment only**, you don't need the whole match. A gallery wants a few
good exemplars per player, not coverage — process a 5–10 minute segment per
team and dump crops from that. It turns an 8-hour job into a short one, and the
gallery it produces is just as usable.

---

## 1. Player identity — build a re-id gallery

The goal is a `gallery.npz` per squad: each player's appearance banked once, so
`identify --method reid` can name a track without reading a jersey number
(OCR only manages ~34% of crops on overhead footage). Full background:
[Enroll](../CLAUDE.md#enroll--carry-a-teams-appearances-instead-of-re-reading-jerseys).

### 1a. Dump crops (HPC)

```bash
soccer-vision enroll --run runs/<match_id> \
  --dump-crops runs/<match_id>/enroll_crops \
  --context-pad 1.0 --context-min 160
```

Writes one folder per ByteTrack lane (`track_0021__ocr20/` — the `__ocr20`
suffix is the jersey number OCR voted, a hint while labelling), plus
`index_*.jpg` review sheets. Only the `--max-tracks 60` longest lanes are
dumped; a full match fragments into ~2,000 lanes (1,663 over 20 frames on the
Saints U11 match), which is far more than anyone will label and unnecessary for
a gallery.

> **Zoom the review sheets in.** The tight crop that feeds the model is
> ~50×21 px — unidentifiable in a file browser. The sheets pull back and outline
> the target so you can use position, teammates and direction of play to tell
> who it is. The **defaults (`--context-pad 3.0 --context-min 384`) are too wide
> to recognise a child**: you see a pitch with a small yellow box on it. At
> `--context-pad 1.0 --context-min 160` the player fills the tile, kit and hair
> colour read clearly, and a jersey number is occasionally legible. Use those
> values. They change the *review sheets only* — the enrolled crops are
> unaffected, so re-dumping to retune costs nothing but a couple of minutes.

### 1b. Sync down (laptop)

```bash
mkdir -p ~/soccer-annotation/<match_id>
rsync -avP b.weinstein@hpg.rc.ufl.edu:/orange/ewhite/b.weinstein/soccer-video-analysis/runs/<match_id>/enroll_crops/ \
  ~/soccer-annotation/<match_id>/enroll_crops/
```

~2 MB for 57 tracks. Seconds, not minutes.

### 1c. Label by renaming folders (laptop)

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

### 1d. Sync back and enrol (HPC)

```bash
# laptop
rsync -avP --delete ~/soccer-annotation/<match_id>/enroll_crops/ \
  b.weinstein@hpg.rc.ufl.edu:/orange/ewhite/b.weinstein/soccer-video-analysis/runs/<match_id>/enroll_crops/

# HPC
soccer-vision enroll --run runs/<match_id> \
  --from-crops runs/<match_id>/enroll_crops \
  --out galleries/<team>.npz          # add --append for later matches

soccer-vision identify --run runs/<match_id> --method reid \
  --gallery galleries/<team>.npz --profile examples/profiles/<team>.yaml
```

`--delete` matters on the way back: it's what makes the folders you deleted
locally actually disappear on HPC.

### Alternative: draw boxes in Label Studio

Whole-track accept/reject is faster, but for a player no track cleanly isolates
— a keeper in a different kit, a child who only ever appears mid-cluster — you
can label boxes on individual frames instead:

1. Export frames from the run's proxy and set up a project with
   `rectanglelabels`, one `<Label value="..."/>` per roster name:
   ```xml
   <View><Image name="img" value="$image"/>
     <RectangleLabels name="player" toName="img">
       <Label value="Simon Weinstein"/>
     </RectangleLabels></View>
   ```
2. Tasks carry the frame number in `data.frame` (or a numeric filename stem).
3. Enrol the export: `soccer-vision enroll --run runs/<match_id>
   --from-label-studio export.json --out galleries/<team>.npz`.

Slower per player, but it's ground truth and it reaches players the folder route
can't.

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
