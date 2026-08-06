---
name: research-record
description: Write or update a numbered research record in docs/research/ — the standing log of what this repo tried, why, how it was measured against held-out data, and what came back. Use whenever proposing a non-trivial change to detection, tracking, re-id, linking, identity or event selection; whenever an experiment finishes; or when the user asks to write up, record, justify or revisit an improvement. Also use before starting such work, to state the hypothesis and the assessment plan before the numbers exist.
---

# Research record

This repo advances by measurement, and its hardest-won knowledge is negative:
SAM3 was adopted on a count that never reproduced, field registration returned
`ok=True` with a garbage matrix, drop-on-conflict shipped and was withdrawn the
same day, a 20x bigger gallery bought nothing, hubness correction and
re-ranking — the two textbook fixes for the exact symptom — bought nothing.
Every one of those cost a cycle because the reasoning behind the previous
decision lived in a commit message or somebody's head.

A research record is where that reasoning goes instead. One file per
investigation, numbered, in `docs/research/`. It is a **lab notebook entry, not
a changelog**: it is written to be read by someone deciding whether to re-tread
the ground, and it is worth writing even — especially — when the answer is "this
did not work".

## When to write one

- **Before** starting non-trivial work on detection, tracking, re-id, linking,
  identity, or selection. State the problem, the hypothesis and the assessment
  plan while they can still be falsified. A record opened at status
  `proposed` is the deliverable.
- **After** an experiment lands, whatever the result. Update the same file to
  `adopted`, `rejected` or `inconclusive`. Never open a second record for the
  same question.
- When the user asks to write up, justify, revisit or explain a change.

Do **not** write one for routine work: bug fixes, refactors, docs, plumbing, or
anything whose correctness is settled by a unit test.

## How to write one

Read `docs/research/TEMPLATE.md` and follow it section by section. Then:

1. **Number it.** `ls docs/research/` and take the next free integer.
   `docs/research/0007-short-slug.md`.
2. **Add one line to `docs/research/index.md`**, in the table, newest last.
3. **Cross-link.** If the record supersedes or depends on another, say so by
   number in both files.

### Ground every claim in this repo's own numbers

A record that argues from the literature alone is not usable here. The measured
facts in `CLAUDE.md` and `slurm/job_ledger.md` are the shared ground: cite them
by figure and by job number (`job 38526638`, `runs/saints-u14g-full-30fps`).
When you assert a problem exists, show the number that shows it. When you can
cheaply produce a fresh number from saved artefacts — most of `slurm/*.py` needs
no GPU — produce it rather than estimating.

Distinguish, always and explicitly:

- **measured** on our footage (say which run, which job, which n);
- **reported** in a paper (say which dataset — SoccerNet-Tracking is broadcast,
  not a fixed wide-angle youth camera, and the gap is usually the whole story);
- **expected**, i.e. your reasoning. Label it as such.

### Cite the literature, and say what transfers

Name the paper, its venue/year, and its arXiv id or repo. Then do the part that
matters: **say what about our setting differs from theirs**, and what that does
to the reported number. The stack this repo lives in is
[SoccerNet Game State Reconstruction](https://arxiv.org/abs/2404.11335), whose
GS-HOTA went 29.0 → 63.9 in a year almost entirely on identity; the components
already scouted are listed under *Identity coverage* in `CLAUDE.md`
(gta-link, Deep-EIoU, jersey-number-pipeline, SportsSUSHI, PRTreID). Prefer
importing a method with a public implementation over re-deriving one, and say in
the record why the method should survive the domain shift — small players, one
kit per squad, a panning consumer camera, and a crowd on the far touchline.

If a search is needed to get the citation right, do it. A record with a
hallucinated reference is worse than one with none, because the next person
spends an hour looking for the paper.

### Name the held-out data before you measure anything

This is the section that makes the record worth keeping. `heldout.yaml` declares
the footage nothing may learn from, and every record must say **which block or
gold set it will be assessed on**, chosen before the experiment runs.

- `u14g-2026-07-11-secondhalf` / gold `u14g-gold-3min` — **within-video**: can we
  name a player in a stretch of a match we enrolled her from but not in.
- `u11-xbotgo-2026-07-19-secondhalf` / gold `u11-simon-gold` — **between-video**:
  do settings tuned on Veo U14G footage survive a different camera, venue, kit,
  and age group.

Rules that hold without exception:

- **Never tune on a gold set.** If a threshold is swept, it is swept outside the
  protected block and reported once on the gold set. A record that reports a
  swept-then-scored number on the same data is reporting nothing.
- **Run `python scripts/audit_heldout.py --all` and paste the verdict** into the
  record's assessment section. `UNAUDITABLE` is not a pass.
- **State the falsifier.** What result would make you drop this? A record whose
  hypothesis cannot fail is an advertisement.
- **State n, and whether the effect clears it.** This project has repeatedly
  been fooled by 45-crop samples: 42% vs 38% there is noise, and the record
  should say so rather than reporting a winner.
- If a claim can only be judged by eye — a halo, a chain, a reel — say that, and
  reference the rendered artefact (`scripts/follow_player.py`, a contact sheet).
  "Verified by eye, crop by crop" is a legitimate and sometimes the only method
  here; hiding it behind a metric is not.

### Write the way the repo writes

Prose that states the mechanism, not bullet-point summaries of it. Say what was
ruled out and why, so nobody re-treads it. Keep the honest caveat in the text
rather than in a footnote — most of the value in `CLAUDE.md` is in sentences of
the form "read that number narrowly, because…".

## After writing

- If the record identifies a scoped problem someone outside could take on, file
  it as a GitHub issue too (see *Community* in `CLAUDE.md`) and put the issue
  number in the record.
- If it changes how the pipeline should be run, update `CLAUDE.md` — the record
  is the argument, `CLAUDE.md` is the standing instruction, and they are not
  interchangeable.
- If it ran a SLURM job, add the job to `slurm/job_ledger.md` with the record
  number.
- Commit the record with the work it describes.
