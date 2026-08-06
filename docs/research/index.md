# Research records

What this project tried, why, how it was measured, and what came back — one
numbered file per investigation, kept whether the answer was yes or no.

The negative results are the point. This repo has adopted a detector on a count
that never reproduced, shipped a jersey-OCR veto that was 0 for 2 against
hand-verified truth, and spent two annotation batches on a gallery ceiling that
turned out to be a property of the embedding. Each cost a cycle because the
reasoning behind the last decision lived in a commit message. These records are
where it lives now.

`TEMPLATE.md` is the shape of one. Write them with the `research-record` skill,
which carries the standards — ground every claim in a measured number from our
own footage, cite the literature *and* say what transfers across the domain gap,
and name the held-out block **before** the experiment runs.

## Held-out data

Every record names the footage it was assessed on. `heldout.yaml` at the repo
root declares the spans nothing may learn from, `enroll` enforces it, and
`scripts/audit_heldout.py` checks a finished gallery.

| Block | Flavour | Footage | Gold set |
|---|---|---|---|
| `u14g-2026-07-11-secondhalf` | within-video | Saints U14G, Veo, 38:00–48:00 | `u14g-gold-3min` — every lane ≥1 s, 40:00–43:00 |
| `u11-xbotgo-2026-07-19-secondhalf` | between-video | Saints U11 boys, XbotGo, 15:45–30:45 | `u11-simon-gold` — Simon, completely |

## Records

| # | Record | Status | Assessed on |
|---|---|---|---|
| 0001 | [Nothing has been measured on held-out footage](0001-heldout-validation.md) | adopted | — (establishes the blocks) |

```{toctree}
:maxdepth: 1
:hidden:

0001-heldout-validation
TEMPLATE
```
