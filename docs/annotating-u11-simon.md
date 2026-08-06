# U11 gold set — Simon, completely, on a camera we've never tested

**Block:** `u11-xbotgo-2026-07-19-secondhalf` · **Gold set:** `u11-simon-gold`
**Footage:** `SaintsU11_OVF_Jul192026.MP4` (XbotGo Falcon), 945–1845 s
**Roster:** one name — Simon (`Si`), #6

## What this measures, and what it doesn't

Everything in this pipeline was tuned on one match: 1080p Veo footage of the
U14G girls, filmed from a distance, at a complex with a crowd on the far
touchline and other matches in shot. `min_margin` 0.05, `min_lane_seconds` 1.0,
`on_ball_dist` 90 px, the field top-cut at 0.15 — all of it was chosen against
that one venue.

This footage is different in every way that those settings care about. The
XbotGo camera sits close to the touchline, so a player renders several times
larger; the squad wears white rather than black; the pitch is enclosed and the
crowd is a handful of people rather than a hundred; and they're eleven-year-old
boys rather than U14 girls. If the settings are really settings and not just
numbers fitted to one afternoon, they survive that. If they don't, we find out
here rather than on the next family's match.

**Read the result narrowly.** This is a test of *settings and workflow*
transferring to new footage. It is **not** yet a test of a gallery transferring
between matches: we have one XbotGo match, so Simon's exemplars come from the
first half of this same video. A second U11 match would upgrade it, and is the
next annotation worth buying.

The two halves must not touch, and they don't: `heldout.yaml` protects 945–1845 s,
`soccer-vision enroll` refuses to bank a crop from it, and there is a 45 s guard
between the last enrolment window and the first gold stretch.

## The two projects

| Folder | Footage | What it's for | Effort |
|---|---|---|---|
| `enrol_tracklets/` | first half, ~12 windows spread over 0–900 s | builds Simon's gallery | small — most windows have no Simon in them |
| `heldout_gold/` | second half, back-to-back 20 s stretches | the answer sheet | the real job |

**Do `enrol_tracklets/` first.** Nothing can be scored until Simon is in a
gallery, and if that batch is thin (he's substituted, or he plays a half) we want
to know before the gold set is annotated rather than after.

## Setting up Label Studio

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=/orange/ewhite/b.weinstein/soccer-video-analysis
label-studio start
```

**Create project → Labeling Setup → Code → paste `labeling_config.xml` →
Import → `label_studio_tasks.json`.** One project per folder — they have
different configs and different manifests.

Import the *tasks file*, not the clips folder: pointing at a directory drops
every field but `video`, and those fields carry the slot → track-id map and the
on-screen guide. Keep `tracklets.json`; without it the export is uninterpretable.

## The decision

Every ringed player gets one of three answers:

| Choose | When |
|---|---|
| `Si` | That's Simon |
| `not ours` | Anyone else — teammates, opposition, referee, spectators |
| `unsure` | You think it might be Simon but can't tell |

**Note the asymmetry, it's deliberate.** Simon's *teammates* go under
`not ours` here, which is not what that label means in the U14G project. With
one name on the roster the question is binary — is this Simon or isn't he — and
a teammate marked `not ours` is a correct negative. `unsure` is reserved for
genuine doubt about Simon specifically, and it is a real answer: a guess in
either direction corrupts the answer sheet, and this one has no second opinion
to average against.

The number on the chip is **not a jersey number** — it's the "Player N" dropdown
below the video, renumbered in every clip. Players are numbered in the order they
first appear, and the text under the video says when each is on screen. Under
`--all-lanes` the same 20 seconds comes round several times ringing different
people (`PASS 3 of 6`); that's expected, not a duplicate.

## Completeness is the point here

"Simon, completely" means every lane he appears in, across the whole held-out
half — not a sample. Recall is what this gold set exists to measure, and recall
computed over a partial answer sheet is flattering in exactly the places the
pipeline is weakest.

Stretches run **back to back** from 945 s, and pages within a stretch are
longest-lane-first. So if you stop early, stop at a stretch boundary: that
leaves a clean contiguous prefix ("we labelled 945–1305 s completely") which is
a usable gold set, where a scattered partial one is not.

If a ring follows Simon and then jumps to another player mid-clip, mark it
`unsure` and note the clip and slot. That's a ByteTrack lane containing two
people, and we have no automated way to find examples.

## When you're done

Export as **JSON** into each project folder as `annotations.json`:

```
runs/saints-u11-xbotgo-30fps/enrol_tracklets/annotations.json
runs/saints-u11-xbotgo-30fps/heldout_gold/annotations.json
```

Then enrol from the first half only — the gate makes that automatic, but check
the printed line says so:

```bash
soccer-vision enroll --run runs/saints-u11-xbotgo-30fps \
    --from-tracklets runs/saints-u11-xbotgo-30fps/enrol_tracklets/annotations.json \
    --profile examples/profiles/saints-u11.yaml \
    --out galleries/saints-u11.npz

python scripts/audit_heldout.py --gallery galleries/saints-u11.npz   # expect CLEAN
```

`CLEAN` means every exemplar is stamped with the frame it came from and none of
them falls inside the protected half. `UNAUDITABLE` is not a pass — it means the
file can't answer the question.
