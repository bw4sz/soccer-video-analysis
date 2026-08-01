# About

## The problem

Youth clubs film every match and almost nobody watches the footage. An hour of
wide-angle video is not a highlight reel, and the tools that would turn it into
one — Trace, Veo Editor, LongoMatch — are subscriptions that keep your footage
and your analysis on someone else's server.

soccer-vision is the open version: your video stays on your disk, the pipeline
runs on a laptop CPU, and every intermediate file is plain JSON you can read.

## What it is built around

**A clip is the unit of value.** Not a stats dashboard, not a heatmap. A parent
wants ninety seconds of their kid; a coach wants every time the back line played
out under pressure. Everything upstream — detection, tracking, team assignment —
exists to make a clip selectable.

**Pixel space, not metres.** There is no pitch registration. Line-based
homography was tried and removed: on overhead footage it locked onto rooftops
and stadium walls instead of pitch lines, and our home venue paints blue, red
and white lines from several overlapping pitches, so even a perfect line
detector cannot say which touchline is *the* touchline. Every spatial question
is answered in pixels, which is enough for "was this player near the ball".

**Abstain rather than guess.** When jersey OCR can't read a number, or the
appearance gallery isn't confident, the track comes back `unknown` instead of
getting a name. A mislabelled clip is worse than an unlabelled one — you stop
trusting the whole reel. The same rule runs through the annotation tools, where
*not ours* and *unsure* are first-class answers.

**Delete what doesn't work.** Field registration failed on all six test frames,
so it was removed along with every metric that depended on it — not left
importable behind a flag. It had been returning `ok=True` with a garbage matrix,
producing 236 confidently-labelled throw-ins, all of them false. Code that fails
invisibly is worse than no code.

**Compute where the footage lives, label where you are.** A run directory is
~10 GB; the file you annotate is ~1 MB. Every annotation workflow is *generate
on the cluster → sync down → label on a laptop → sync back*.

## What honestly works today

| | Status |
|---|---|
| Finding players, keeper, referee, ball | Works — RF-DETR, F1 0.902 on broadcast benchmark |
| Following them frame to frame | Works — ByteTrack, but ids fragment (see below) |
| Splitting them onto two teams by kit | Works |
| Cutting one player's on-ball moments | Works — this is the main pathway |
| Naming a player by jersey number | Partial — ~34% of crops legible on overhead footage |
| Naming a player by appearance | Partial — 42% rank-1 on same-kit teammates |
| Naming *what* they did (pass, shot, tackle) | Not yet — no detector ships today |
| Where the pitch is | No — removed, see above |

Two known rough edges on the detector path, both measured and both open:

- **The ball track flickers.** 84.5% of frames get a ball, but the p95
  frame-to-frame jump is 905 px on a 1920 px-wide frame — it latches onto a
  jersey number for a frame and snaps back. A Kalman smoother exists and is
  wired into `trim-empty` but not into `process`.
- **Track ids are ephemeral.** 856 lanes in three minutes, median lane 1.4
  seconds. One player becomes many lanes. `enroll` and `identify` exist to merge
  them back into a person.

## Where it is going

Roughly in order:

1. **Action recognition** — passes, shots, tackles, attributed to the player who
   did them. This is what makes *"#6's passes"* real. See
   [Training new detectors](training-detectors.md).
2. **Sturdier identity** — following one player through a whole match as a
   single thread.
3. **A turf mask** for on-field/off-field — segment the green, take the largest
   region, use its boundary. No lines, no homography, no pretrained model.
4. **A desktop app** for reviewing clips without a terminal.
