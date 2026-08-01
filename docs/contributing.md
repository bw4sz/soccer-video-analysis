# Contributing

Early-stage project, small surface area, real footage to test against. Issues
and PRs both welcome.

## Dev loop

```bash
git clone https://github.com/bw4sz/soccer-video-analysis.git
cd soccer-video-analysis
pip install -e ".[dev]"

ruff check src/ tests/
pytest -q
```

214 tests, all passing; CI runs lint and tests on every push and PR.

## Where help is most useful

Roughly in order of impact:

**Naming same-kit teammates.** The hardest open problem. Appearance re-ID gets
42% rank-1 on our squad — chance is 9%, so there's signal, but not enough.
Re-ID normally separates people by clothing and teammates wear the same kit; at
~53×76 px all that's left is build, hair and gait.

Two leads that survived testing, both worth chasing:

- The correct player is in the **top 3 exemplars 62%** of the time even though
  rank-1 is 42% — the scoring isn't extracting signal that's already there.
- Accuracy tracks **contrast and box size**, not brightness. The low-contrast
  half scores 7/22 against 12/23 for the high-contrast half.

Already ruled out, so nobody re-treads it: capping exemplars per player (rank-1
unchanged, precision drops); tuning `min_similarity` (nearly inert — the
*margin* decides); `top_k` (trades recall for precision, no free accuracy).
Code is in `src/soccer_vision/identify/`.

**An action detector.** See [Training new detectors](training-detectors.md).
The plug-in seam already exists; a checkpoint is what's missing.

**Wiring the Kalman smoother into `process`.** It exists
(`soccer_vision.tracking.ball_kalman`) and is already used by `trim-empty`, but
`process` writes `ball_track.json` raw, so on-ball spans inherit the ball's
p95 905 px jitter. Contained and clearly useful.

**A turf mask for on-field/off-field.** HSV green + morphology + largest
connected component → boundary polygon, anchored on keyframes and propagated by
optical flow. No lines, no homography, no pretrained model, no domain shift.
Would replace `detection/field_filter.py`, which is currently a single
horizontal cut and cannot separate the far-touchline crowd from our own
far-side players.

**Cameras that aren't overhead**, more test footage, and real-footage bug
reports. Today's defaults are tuned for youth 7v7 on a 55×36 m pitch.

## Filing an issue

Short and self-contained beats thorough:

- **What's broken**, with concrete evidence — a number, a frame, a small chart —
  rather than a description.
- **What you already ruled out**, so nobody re-treads it.
- **The direction you think is promising**, as a lead not a mandate.
- **Where to look**: exact file paths.

One illustrative image when it makes the problem obvious at a glance. A real
chart from real data beats a mockup.

Look for `help wanted` and `good first issue` labels.

## House style

**Measure before you claim.** Every performance number in these docs came from a
job we can point at. If you're replacing a component, A/B it on a fixed held-out
set — we've twice found that an obvious improvement made no measurable
difference, which is worth knowing before it ships.

**Abstain rather than guess.** When a component isn't confident, the answer is
`unknown`. A mislabelled clip is worse than an unlabelled one.

**Delete what doesn't work.** Don't leave a broken component importable behind a
flag — someone will wire it back up. Field registration was removed along with
every metric that depended on it, and the reasoning was written down so it
doesn't get rebuilt by accident.

**Explain the why in the code.** The comments that have earned their keep here
are the ones saying why a default is what it is — why 15 fps and not 5, why the
top of the frame and not a centred box.

## License

AGPL-3.0-or-later.
