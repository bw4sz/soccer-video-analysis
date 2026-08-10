# U14G gold set — 40:00 to 43:00, every lane, every kit

**Block:** `u14g-2026-07-11-secondhalf` · **Gold set:** `u14g-gold-3min`
**Footage:** `wfc-rangers-vs-saints-pcu-cup-2026-07-11.mp4` (Veo), 2400–2580 s
**Size:** 51 tasks over 9 × 20 s stretches · 767 lanes to decide

## Why this one is different from the last two batches

The tracklet batches of 2026-07-29 and 2026-08-01 were **training data** — they
built the gallery. This one is the opposite: it is the **answer sheet**, and
nothing may ever learn from it.

The whole 38:00–48:00 stretch of this match is declared off-limits in
`heldout.yaml`. `soccer-vision enroll` now refuses to bank a crop from it, so
what you label here can score the pipeline without the pipeline having seen it.
That is the one thing none of our existing numbers can claim (see
[research record 0001](research/0001-heldout-validation.md)).

Two consequences for how to label:

- **Completeness matters more than volume.** For a gallery, a lane skipped is a
  lane not banked. Here, a lane skipped is a hole in the answer sheet, and a
  recall number computed over a hole flatters us exactly where we're weakest.
- **`unsure` is a real answer and a useful one.** Do not guess to clear the form.
  A guessed name here doesn't just cost one crop, it silently marks a correct
  prediction wrong (or a wrong one right).

## Setting up Label Studio

Serve the clips from the repo root, so both gold projects share one document
root:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=/orange/ewhite/b.weinstein/soccer-video-analysis
label-studio start
```

Then: **create project → Labeling Setup → Code → paste `labeling_config.xml` →
Import → `label_studio_tasks.json`**.

Import the *tasks file*, not the clips folder. Label Studio drops every field but
`video` when you point a project at a directory, and those fields carry the
slot → track-id map. (Enrolment can fall back to the clip filename, but the
on-screen guide under the video is lost, and that guide is what tells you when
each player is in shot.)

Keep `tracklets.json`. It is the only record of what each slot number meant, and
without it the export is uninterpretable.

## What you're looking at

Each clip is 20 seconds of play with tracked players ringed at the feet and
numbered. **The number on the chip is not a jersey number** — it is the "Player
N" dropdown below the video, and it is renumbered in every clip.

Players are numbered **in the order they first appear**. Playing the clip once
walks the form top to bottom. The text under the video says when each one is on
screen ("Player 3: 4–19 s"), and names the slots this clip doesn't use so a form
with 16 dropdowns and 9 players doesn't look broken.

**The same 20 seconds comes round up to 7 times**, ringing different players each
pass — the header says `PASS 3 of 7`. That is expected, not a duplicate task.

**Passes are ordered so that ours come first.** Pass 1 of every stretch is our
black kit, all sixteen of them; pass 2 is mostly ours; from pass 4 on it is the
opposition and the neighbouring pitch almost entirely. Within each group the
busiest lanes come first — how far someone moved once the camera's own pan is
subtracted, which is what separates a player from somebody standing on the
touchline.

Both are *orderings*. Every lane still gets a pass, including the ones the kit
classifier got wrong, because it does get them wrong — it reads the yellow
referee as black, and a shaded white shirt as black too.

## The decision, for each ringed player

| Choose | When |
|---|---|
| A name (Morgan, Catherine, Gia, Izzy, Joelle, Leire, Evie, Ila, Mo, Lainey, Riley, Iris, Quinn) | It's one of ours and you can tell who |
| `not ours` | Opposition, referee, spectator, or a player on the neighbouring pitch |
| `unsure` | It's one of ours but you can't say which — or you can't tell whose team it is |

**`not ours` is the majority answer and it is not a throwaway.** Roughly 60% of
the lanes in these clips are the light-blue opposition, the yellow-shirted
officials, the crowd sitting along the far touchline, or the match being played
on the pitch behind ours. Naming those is the single largest measured error in
this pipeline — before the kit gate, 878 of 1595 names landed on the white kit,
Gia Olson alone taking 511 of them. A gold set holding only our own squad cannot
see that error, which is why every lane gets a decision.

Two cases worth knowing:

- **A ring that follows one player and then jumps to another.** Real, and worth
  catching: ByteTrack lanes sometimes contain two people. Label it `unsure` and,
  if you can, note the clip and slot — that's the case a track *splitter* exists
  for, and we have no other way to find examples.
- **A ring on empty grass, or on a duplicate box over a player already ringed.**
  `not ours`.

Guest players and the keeper are both real here: someone in our kit who isn't on
the list is `unsure` (not `not ours`), and Izzy in goal wears a different kit
from the outfield black.

## Working through it

Pages are ordered, so **stopping early leaves a gap of known size** rather than a
biased sample — which is the whole reason the clips are paged. If you only get
through part of it, finish whole tasks and stop; a half-finished task is worse
than an unstarted one.

If 51 tasks is too many, do **pass 1 and 2 of each of the 9 stretches** first —
18 tasks, and they hold 262 of the 313 lanes wearing our kit. That is a coverage
figure the analysis can state honestly: near-complete on our own squad, thin on
the opposition.

Be aware of what stopping there costs, though. The lanes on passes 4 to 7 are
where a *wrong name* shows up — our biggest measured error is naming an opponent
Gia — so a gold set that stops at pass 2 can measure whether we found our
players but not whether we named somebody else's.

## When you're done

Export as **JSON** and drop it here as
`runs/saints-u14g-full-30fps/heldout_gold/annotations.json`.

Then, and only then:

```bash
# Score identity against the gold set (never enrol from it)
python slurm/eval_link_ground_truth.py --run runs/saints-u14g-full-30fps \
    --tracklets runs/saints-u14g-full-30fps/heldout_gold

# Confirm nothing learned from it
python scripts/audit_heldout.py --all
```
