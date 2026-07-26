# soccer-vision

**Youth sports video review, from raw match footage to a highlight reel — no subscription.**

Point it at a single-camera match (Veo, overhead, or any fixed wide-angle source)
and ask for the clips you want. Two kinds of request drive everything:

- **Individual actions** — *"every passing action by number 6 on the black team"* → one combined reel.
- **Team actions** — *"all the throw-ins from the blue team"* → one combined reel.

Same pipeline, same filter — pick a player, a team, an action label, or any
combination. An open alternative to Trace / Veo Editor / LongoMatch with
programmatic control over your own footage. Python 3.12+, CPU-viable, no cloud.

```bash
soccer-vision process match.mp4          # detect → track → attribute → store
soccer-vision reel --run runs/match_001 --track 6 --event pass --out number6_passes.mp4
soccer-vision extract --run runs/match_001 --events throw_in --team blue
```

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
5. Map pixel positions onto the real field
6. Spot the actions — today that's set pieces from ball position; passes,
   shots, and tackles are on the way — and tie each one to the player and team
   who did it
7. Tally the numbers: distance covered, possession, shots, event counts
8. Save everything: clips, an event log, and contact sheets for review

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

**172 unit tests pass**; CI checks every change automatically.

---

## The two clip workflows

Both share the same `process` run and diverge only at **selection** — actions
are already tagged with a player and team, and get filtered before cutting:

```bash
# Individual player — every action by player #6
soccer-vision reel --run runs/match_001 --track 6 --out number6.mp4

# Team action — all throw-ins by the blue team
soccer-vision extract --run runs/match_001 --events throw_in --team blue

# Combine — only #6's passes, one reel
soccer-vision reel --run runs/match_001 --track 6 --event pass --out number6_passes.mp4
```

> **Two ways to pick a player.** *Team-level* filtering works off jersey
> **colour** (`--team black`) — no extra setup. *Individual-player* filtering
> (`--player Simon` / `--number 6`) needs one extra step,
> `soccer-vision identify`, which reads each player's jersey number off the
> footage. On overhead footage some players are too far or too turned away to
> read — fall back to `--team` / `--track` there.

> **Asking for one player always gives you something.** Only set pieces are
> detected as named actions today, so "every clip of number 6" would usually
> match nothing. When it does, the clips fall back to the moments that player was
> **on the ball** — nearest to it and close enough for it to be their touch —
> which is a far denser signal than the event stream. That's proximity, not
> action recognition: it finds when #6 had the ball, not that #6 *passed*. Pass
> `--no-on-ball` to turn it off.

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
- **Sturdier player identity** — carrying a player's identity through
  stretches where their jersey number can't be read, so overhead-camera
  footage resolves as reliably as broadcast footage.
- **A better way to map the field** when the lines on the pitch are faint or
  partly hidden.
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
