# 0001 — Nothing in this repo has been measured on held-out footage

- **Status:** adopted
- **Opened:** 2026-08-05 · **Closed:** 2026-08-05
- **Assessed on:** n/a — this record *establishes* the held-out blocks everything
  after it is assessed on
- **Audit:** every existing gallery reports `UNAUDITABLE`
  (`scripts/audit_heldout.py --all`, 2026-08-05)
- **Touches:** `heldout.yaml`, `src/soccer_vision/heldout.py`,
  `src/soccer_vision/cli/enroll.py`, `src/soccer_vision/identify/gallery.py`,
  `scripts/audit_heldout.py`
- **Jobs:** 38794172 (U11 30 fps `process`)

## The problem

Every accuracy figure this project has collected was measured against exemplars
drawn from the same match — and usually the same few minutes — as the crops
being scored. The leave-one-frame-out protocol in `slurm/validate_reid_frames.py`
holds out one *frame*, then builds the gallery from the rest of the same match.
Its headline, 42% rank-1 at zero margin, is recorded in `CLAUDE.md` as "the
ceiling", and the A/B tables that followed (47 → 927 exemplars, six scoring
rules, hubness correction, k-reciprocal re-ranking) were all read off the same
45-odd crops under the same protocol.

The enrolment sources make the overlap concrete. The 24 labelled frames behind
`galleries/saints-u14g.npz` are spread across the match at roughly 150 s
intervals; the 620-crop tracklet batch behind `galleries/saints-u14g.fullmatch.npz`
comes from 20 s windows every 329 s. So for **any** crop in that match, an
enrolled crop of the same player exists within about a minute of it — same sun
angle, same patch of pitch, same hair tie, often the same stride. Re-id scoring
is nearest-neighbour over cosine similarity; a query that close to a gallery
exemplar is being asked to recognise a near-duplicate of itself.

**What this is not.** It is not a claim that 42% is wrong, or that the ceiling
isn't real. If anything the bias runs the other way: a number inflated by
near-duplicate exemplars and *still* only 42% is worse news, not better. Nor is
it a claim that anyone cheated — leave-one-frame-out is a reasonable protocol,
and the overlap is a property of how the annotation was sampled, which was
chosen for gallery diversity rather than for evaluation. The problem is that we
cannot **tell**: no gallery on disk records which frames it was built from, so
no number in `CLAUDE.md` can say what it was measured against.

## Why the obvious fix is not the fix

The obvious fix is to keep sampling annotation for diversity and to hold out a
frame at scoring time, which is what we do now. It fails for a structural
reason: the held-out unit is smaller than the unit of correlation. A frame is
held out; the *appearance* it carries is not, because the neighbouring second of
video carries the same appearance and is in the gallery.

The second obvious fix — a bigger, more varied gallery — has already been tried
twice and is documented in `CLAUDE.md` as buying nothing: 47 → 307 → 667 → 927
exemplars moved rank-1 across 42%, 38%, 37%, 43%, every row inside noise at
n≈45. That finding is *itself* subject to this record's complaint, since it was
measured under the leaky protocol. It probably survives — the effect being
looked for was large and absent — but it cannot be quoted as settled until it is
re-run against a clean block.

## What the literature says, and what transfers

Person re-identification benchmarks have treated exactly this hazard as
fundamental since the field's standard datasets were built. Market-1501 (Zheng
et al., *Scalable Person Re-identification: A Benchmark*, ICCV 2015) evaluates
**cross-camera**: when a gallery image of the query identity comes from the same
camera as the query, the protocol discards it as junk rather than counting it,
because same-camera matches share pose, lighting and background and are trivially
easy. DukeMTMC-reID and MSMT17 keep that convention. The whole discipline exists
because the easy matches inflate the number without predicting the deployment
case.

Our situation is that convention inverted. We have one camera, so
*every* match is same-camera, and we have additionally been scoring against
gallery crops seconds away in time — tighter coupling than the same-camera
matches Market-1501 throws away. The sportsreid `OSNet_x1_0` weights we use are
the SoccerNet-ReID 2022 challenge entry (83.4 mAP / 78.0 rank-1, *reported* on
SoccerNet-ReID), and that benchmark is broadcast footage with camera cuts, which
is closer to the cross-camera setting than to ours.

What transfers, then, is the **protocol**, not the number: hold out along the
axis that carries the nuisance correlation. In Market-1501 that axis is the
camera. For a single fixed camera watching one match it is **time**, because
time is what varies sun angle, position on the pitch, fatigue and pose. Hence a
contiguous protected block rather than scattered held-out frames.

The second question — whether any of this survives a different camera and squad
— is the *domain* question that generalizable-re-id work (and the SoccerNet Game
State Reconstruction task, [arXiv 2404.11335](https://arxiv.org/abs/2404.11335))
poses across datasets. We can pose it directly, because we own footage from two
cameras: Veo on the U14G girls and an XbotGo Falcon on the U11 boys. *Expected*,
and stated here so it can be checked rather than assumed: thresholds tuned on
the Veo footage (`min_margin` 0.05, `min_lane_seconds` 1.0, `on_ball_dist` 90 px)
are calibrated to a pixel scale and a crowd density that the XbotGo footage does
not share — its camera sits far closer to the pitch, so a player renders several
times larger.

## Hypothesis

Two, and they are separable on purpose:

1. **Within-video.** Re-id accuracy measured on a contiguous protected block,
   with no exemplar from that block or its guard band, is *lower* than the 42%
   measured under leave-one-frame-out. If it is not, the near-duplicate concern
   was unfounded and the 42% ceiling stands as reported.
2. **Between-video.** Settings tuned on the U14G Veo footage degrade on XbotGo
   U11 footage, and the degradation is in *coverage* (lanes gated out, spans
   dropped) rather than in naming precision — because the gates are calibrated in
   pixels and seconds, and only the pixel scale changes.

## How it will be assessed

Two blocks, declared in `heldout.yaml` before any of the annotation exists.

**`u14g-2026-07-11-secondhalf` — within-video.** 2280–2880 s (38:00–48:00) of
`wfc-rangers-vs-saints-pcu-cup-2026-07-11.mp4`. Chosen as the widest gap between
existing enrolment sources: protecting it costs the gallery 4 of its 24 labelled
frames (2290, 2539, 2688, 2788 s) and 2 of its 12 tracklet windows (2301, 2630 s),
which is the cheapest such block in the match. Ordinary passage of play — ~12
tracked black-kit players per frame, ball visible in 75% of samples.

Gold set `u14g-gold-3min`, 2400–2580 s: **every lane of every kit lasting at
least 1.0 s**, which is the same floor `identify --min-lane-seconds` applies, so
a lane out of scope here is one production never names either. 51 Label Studio
tasks, 767 lane decisions, 313 of them black-kit. Opponents, officials,
spectators and the neighbouring pitch are labelled `not ours` rather than
skipped — naming those is the largest measured error in the pipeline (878 of
1595 names landed on the white kit before the kit gate) and a ground truth
holding only our own squad cannot see it.

The 600 s block against a 180 s gold set is deliberate: the surplus is guard band
(appearance a minute either side of a query is near-duplicate) and reserve for
the next gold set, which can then be cut without renegotiating what is already
protected.

**`u11-xbotgo-2026-07-19-secondhalf` — between-video.** 945–1845 s of
`SaintsU11_OVF_Jul192026.MP4`, XbotGo Falcon, Saints U11 boys in white. Simon
Weinstein is enrolled from the first half (0–900 s) with a 45 s guard, and
labelled **completely** across the held-out half — every lane he appears in, with
everyone else `not ours`. One player rather than a squad because the question is
different: not "can we tell teammates apart" (issue #25) but "does any of this
survive a new camera, venue, kit and age group".

Read that block narrowly. It tests transfer of *settings and workflow*, not of a
*gallery*: with one XbotGo match, Simon's exemplars necessarily come from the
same video. A second U11 match would upgrade it to genuine cross-match transfer,
and is the next annotation worth buying.

**Falsifier for the whole record:** if the within-video block scores at or above
leave-one-frame-out's 42%, the leakage concern is empirically unfounded and this
machinery is overhead. That is a real possible outcome and it should be reported
as such.

**Audit.** Every gallery currently on disk returns `UNAUDITABLE` — none carries
provenance, so none can support a generalization claim. They are not condemned as
leaking; they are condemned as unable to answer:

```
saints-u14g.npz                  UNAUDITABLE     47   (47 unstamped)
saints-u14g.fullmatch.npz        UNAUDITABLE    620   (620 unstamped)
saints-u14g.fullmatch-neg64.npz  UNAUDITABLE    684   (684 unstamped)
```

The U14G block withholds 17.5% of track samples and 3,986 of 26,211 lanes on
`runs/saints-u14g-full-30fps`. That is the price, and it is paid in gallery
depth, which the A/B tables above suggest we have to spare.

## Steps taken

1. `heldout.yaml` — the registry, with both blocks and their gold sets, written
   before the annotation. Prose in the file explains each choice, because a
   block whose rationale is lost is a block someone will quietly move.
2. `src/soccer_vision/heldout.py` — parses it and answers the three questions
   callers have: is this frame protected, which spans cover this run, and split
   these items into inside/outside. Blocks list both a source video and the run
   directories derived from it, since every run renames its footage to
   `broadcast_proxy.mp4`. An **unregistered** run reports itself as unprotected
   rather than returning an empty list quietly — that is the direction that
   loses data silently.
3. `enroll` gained `--heldout-mode {exclude,only,off}`, default `exclude`, wired
   into all five routes into a gallery (tracklet export, Label Studio frames, OCR
   bootstrap, and both dump paths). `only` inverts the gate and is how the gold
   projects are staged; enrolment refuses that mode outright. The gate prints on
   every run, including when it finds nothing — a silent gate is one nobody
   notices has stopped working.
4. Galleries now carry **provenance** (`source` = `run@frame`, one per exemplar),
   surviving the per-player cap and the `.npz` round trip. Merging a stamped
   gallery with an unstamped one drops provenance entirely rather than keeping a
   half-truth that would audit clean on the half it recorded.
5. `scripts/audit_heldout.py` — three verdicts, and the middle one is the point:
   `CLEAN`, `UNAUDITABLE` (cannot answer — **not** a pass), `LEAK`.
6. Both gold projects staged with `--heldout-mode only`, plus written
   instructions for the annotator.
7. `tests/test_heldout.py` — 12 tests pinning the noisy-failure behaviours.

## Result

No accuracy result yet: the gold sets are staged, not annotated. What is settled:

| | |
|---|---|
| galleries auditable before this record | 0 of 5 |
| U14G block cost, `saints-u14g-full-30fps` | 17.5% of track samples, 3,986 of 26,211 lanes |
| U14G block cost, gallery | 4 of 24 labelled frames, 2 of 12 tracklet windows |
| U14G gold set | 51 tasks, 767 lane decisions (313 black-kit), 3 min of play |
| U11 gold set | Simon, complete, 15 min of play |

## Verdict and what changes

Adopted as process. From here: **a number quoted without naming the block it was
measured on is not a result.** The figures already in `CLAUDE.md` stay, because
they are the best we have and most of them are about ruling things out — but
they are now understood as leave-one-frame-out numbers on overlapping data, and
the ones that gate a decision should be re-run against `u14g-gold-3min` before
anyone builds on them. First candidates: the 42% ceiling itself, and the
negative-class margin work of 2026-08-04, whose `p90` gate was chosen on the same
match it was scored on.

The sentence worth keeping: **the held-out unit has to be at least as large as
the unit of correlation.** Holding out a frame from a 30 fps video holds out
nothing, because the frame beside it is the same picture.

## What this opens up

- Re-run leave-one-frame-out's conclusions against the clean block, starting with
  whether gallery size really is inert (records to follow).
- A second U11 match, which turns `u11-simon-gold` from a settings-transfer test
  into a genuine cross-match gallery-transfer test.
- The gold set is per-lane identity over a contiguous stretch, so it also scores
  things this record didn't set out to measure: link precision and recall
  (two lanes with one name are the same player), the kit gate, and spectator
  rejection. `slurm/eval_link_ground_truth.py` already expects exactly this
  format and has been waiting on data since 2026-08-02.
