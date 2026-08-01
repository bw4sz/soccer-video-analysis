# Team profiles

A profile is one YAML file describing your squad. It's optional for team-colour
clips and required for naming players.

```yaml
team_name: "Northgate U14"
season: "2026"
profile_id: "northgate-u14"

kits:
  - black
  - white

roster:
  - name: Alex Rivera
    jersey: 4
  - name: Priya Raghunathan
    nickname: Pri
    jersey: 8
  - name: Margaret Okonkwo
    nickname: Maggie
    jersey: 21

reid:
  gallery: galleries/northgate-u14.npz
  min_similarity: 0.5
  min_margin: 0.05
```

Use it with any command:

```bash
soccer-vision process match.mp4 --profile examples/profiles/northgate-u14.yaml
soccer-vision identify --run runs/match_001 --profile examples/profiles/northgate-u14.yaml
soccer-vision reel --run runs/match_001 --player Maggie --profile examples/profiles/northgate-u14.yaml
```

## The blocks

`kits`
: The two kit colours in the match. Declaring them lets the team classifier
  match clusters to a real kit rather than guessing from HSV — a black kit
  filmed into a low sun renders navy. It also decides which
  [team-splitting route](team-highlights.md#why-kit-colour-is-harder-than-it-sounds)
  is used.

`roster`
: Maps jersey number to name, so `--player Alex` resolves to `--number 4`.
  Also drives the label list an annotator sees in Label Studio.

`nickname`
: What the squad actually calls someone. It replaces the first name in the
  annotator's label list — clicking "Maggie" twenty times a frame beats
  translating "Margaret" each time. Enrolment maps it back to the full name, so
  the gallery, `--player Maggie`, `--player "Margaret Okonkwo"` and `--number 21`
  all reach the same person.

`reid`
: Where the appearance gallery lives and how confident a match must be. Settle
  these once here and drop the flags. `--method` on the command line still
  overrides. Omit the block entirely to stay on jersey OCR.

:::{tip}
Tune `min_margin`, not `min_similarity`. Re-ID cosine similarities bunch high
even between different people, so `min_similarity` is nearly inert — 0.0 and 0.7
give the same answer. The *margin* between first and second place is what
actually decides. 0.05 is the working default; raising it costs recall fast
without buying precision.
:::

Working examples live in
[`examples/profiles/`](https://github.com/bw4sz/soccer-video-analysis/tree/master/examples/profiles).

:::{note}
The gallery is **kit- and season-specific**. A team with a home and an away
strip needs both enrolled, or a home gallery will abstain on every away track.
Enrol from one match of each and `--append`.
:::
