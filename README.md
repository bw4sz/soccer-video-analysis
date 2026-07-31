# soccer-vision

**Youth sports video review, from raw match footage to a highlight reel — no subscription.**

Point it at a single-camera match (Veo, overhead, or any fixed wide-angle source)
and ask for the clips you want. Two kinds of request drive everything:

- **Individual actions** — *"every touch by number 6 on the black team"* → one combined reel.
- **Team actions** — *"everything the blue team did on the ball"* → one combined reel.

Same pipeline, same filter — pick a player, a team, an action label, or any
combination. An open alternative to Trace / Veo Editor / LongoMatch with
programmatic control over your own footage. Python 3.12+, CPU-viable, no cloud.

```bash
soccer-vision process match.mp4          # detect → track → attribute → store
soccer-vision reel --run runs/match_001 --number 6 --halo --out number6.mp4
soccer-vision extract --run runs/match_001 --number 6 --team blue
```

> Selecting by **action label** (`--events pass`) needs an action detector, which
> is the piece still in progress — see [What works now](#what-works-now).
> Selecting by **player** or **team** works today.

---

## Progress on real youth footage

The pipeline runs end-to-end on our own Veo match footage, on an ordinary
laptop CPU — no GPU required.

**Finding and following everyone on the pitch — every player, the keeper, the
referee, and the ball, each kept track of frame to frame
([full-res clip](docs/images/perception_clip.mp4)):**

[![Detection and tracking on a youth Veo shot on goal](docs/images/perception_clip.gif)](docs/images/perception_clip.mp4)

Finding and following players is solid; figuring out *who did what* is the
current focus (see [What's left](#whats-left)).

---

## What works now

`soccer-vision process match.mp4` turns a raw match into an event log, team
stats, and cut clips:

1. Optionally steady the shot — crop the wide single-camera view down to a
   followed, 16:9 view of the action (`--broadcast`; off by default)
2. Find the ball and every player in each frame
3. Keep track of who's who across the match
4. Sort players onto their two teams by kit colour
5. Work out **who was on the ball, and when** — the stretches where a chosen
   player was close enough to the ball to be touching or challenging for it
6. Tally the numbers: possession, team assignments, event counts
7. Save everything: clips, an event log, and contact sheets for review

**On action labels.** Naming *what* a player did — pass, shot, tackle, throw-in
— is the open problem, not finding and following them. The set-piece detector
that used to fill that gap was removed: it decided "this is a throw-in" from the
ball's position in field metres, and field registration does not work on this
footage (both estimators failed on all 6 test frames), so it produced confident
labels that were simply false. What ships today is honest and useful: pick a
player and get their on-ball moments, cut and haloed. Learned action spotting is
the next piece — see [What's left](#whats-left).

Every match writes a self-contained run directory:

```
runs/{match_id}/
├── broadcast_proxy.mp4    # the video every step reads (the source itself,
│                          #   unless --broadcast cropped a followed view)
├── annotations.json       # events: label, frame, team, player
├── tracks.json            # where every player was, and which team
├── ball_track.json        # where the ball was
├── stats.json             # team metrics
├── clips/ · sheets/       # extracted clips + review contact sheets
runs/soccer_vision.db      # match records across all your videos
```

**171 unit tests pass**; CI checks every change automatically.

---

## The two clip workflows

Both share the same `process` run and diverge only at **selection** — actions
are already tagged with a player and team, and get filtered before cutting:

```bash
# Individual player — every on-ball moment for player #6
soccer-vision reel --run runs/match_001 --number 6 --halo --out number6.mp4

# Team filter — #6's moments, restricted to when they played for blue
soccer-vision extract --run runs/match_001 --number 6 --team blue

# Once an action detector ships, add a label to either of the above
soccer-vision reel --run runs/match_001 --number 6 --event pass --out number6_passes.mp4
```

> **Two ways to pick a player.** *Team-level* filtering works off jersey
> **colour** (`--team black`) — no extra setup. *Individual-player* filtering
> (`--player Simon` / `--number 6`) needs one extra step,
> `soccer-vision identify`, which puts a name on each player by one of two
> routes. It can **recognise them by appearance**, looking each player up in a
> gallery of your squad built once by `soccer-vision enroll` — the better route
> for a team you film every week, since it works even when the number can't be
> seen. Or it can **read the jersey number** off the footage, which needs no
> setup but, on overhead footage, misses players who are too far away or turned
> away. Anyone neither route can name falls back to `--team` / `--track`.

> **Asking for one player always gives you something.** Only set pieces are
> detected as named actions today, so "every clip of number 6" would usually
> match nothing. When it does, the clips fall back to the moments that player was
> **on the ball** — close enough to it to be their touch or their challenge —
> which is a far denser signal than the event stream. That's proximity, not
> action recognition: it finds when #6 had the ball, not that #6 *passed*. Pass
> `--no-on-ball` to turn it off.

---

## Teaching it your own squad

Two short annotation passes make the clips yours, and
[`label_studio/README.md`](label_studio/README.md) walks through both:

- **Who's who** — label a squad once and every match after that names players by
  appearance instead of squinting at jersey numbers (`soccer-vision enroll` →
  `identify --method reid`). Either watch a few 20s clips with each player ringed
  and numbered and name the numbers, or name pre-drawn boxes on a couple of dozen
  still frames — the clips are higher yield, the frames need no processed run.
- **What happened** — confirm or correct the pipeline's event label on each
  clip in Label Studio; the corrections are the training set that teaches it
  youth footage.

Both are built to run where the footage lives (a cluster, a workstation) and
labelled on your laptop — the files you annotate are small, the ones they come
from aren't.

---

## Install & run

```bash
pip install -e .            # core (CPU) — also [gpu] / [gui] / [dev] extras
```

Requires `ffmpeg` on PATH.

```bash
soccer-vision process match.mp4 [--config examples/process_match.yaml] [--profile examples/profiles/saints-u10.yaml]
soccer-vision extract --run runs/match_001/ --events goal_kick corner_kick
soccer-vision reel    --run runs/match_001/ --event goal_kick --out goal_kicks.mp4
soccer-vision verify  --run runs/match_001/ --profile examples/profiles/saints-u10.yaml   # needs ANTHROPIC_API_KEY
soccer-vision ask "which team had more corners?" --run runs/match_001/
```

---

## Project structure

```
src/soccer_vision/     the pipeline itself — capture, track, label, and cut clips
training/              scripts for improving the action-recognition models
docs/                  full documentation
tests/                 172 tests + video fixtures
```

The full technical architecture — which models, which libraries, what's done
vs. in progress — lives in `SOCCER_VISION_SPEC.md`, kept separate from this
README so newcomers aren't met with implementation detail up front.

---

## What's left

Roughly in priority order:

- **Recognizing more actions automatically** — passes, shots, tackles, crosses,
  headers, and more, attributed to the player who did them. This is the whole
  point of the current phase, and what makes *"#6's passes"* real. A model is
  training on our own footage now.
- **Sturdier player identity** — `soccer-vision enroll` now carries a squad's
  appearances between matches, so players can be named where their number can't
  be read. Still to do: following one player through the whole match as a single
  thread, rather than naming each stretch of tracking on its own.
- **A usable sense of where the pitch is.** Line-based field registration was
  removed rather than kept limping: on overhead footage it locked onto rooftops
  and stadium walls instead of pitch lines, and our home venue paints blue, red
  and white lines from several overlapping pitches, so even a perfect line
  detector can't say which touchline is *the* touchline. The promising direction
  is a **turf mask** — segment the green, take the largest region, use its
  boundary polygon — which gives on-field/off-field and rough zones without
  needing lines, metres, or a pretrained model.
- **A desktop app** for reviewing clips without the command line.
- **More action labels** — free kicks, kickoffs, substitutions.
- **Easier install** — hosted docs, a PyPI release, example notebooks.

---

## Contributing

Early-stage — high-leverage right now: new ways to recognize actions, support
for cameras that aren't overhead, more test footage, and real-footage bug
reports (today's defaults are tuned for youth 7v7, 55×36 m). See
`SOCCER_VISION_SPEC.md` for the architecture contributors plug into.

Dev loop: `pip install -e ".[dev]"` → `ruff check src/ tests/` → `pytest -q`.

## License

AGPL-3.0-or-later.
