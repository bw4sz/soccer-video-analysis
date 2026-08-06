# NNNN — <one line naming the problem, not the solution>

- **Status:** proposed | running | adopted | rejected | inconclusive | superseded by NNNN
- **Opened:** YYYY-MM-DD  ·  **Closed:** YYYY-MM-DD
- **Assessed on:** `<heldout.yaml block or gold id>`  ·  **Audit:** `CLEAN` (scripts/audit_heldout.py, YYYY-MM-DD)
- **Touches:** `src/soccer_vision/<module>.py`, …
- **Jobs:** 38xxxxxx  ·  **Issue:** #NN

## The problem

What is broken or missing, in one paragraph, with the number that shows it.
Name the run and the job. If the problem is a *loss* rather than an *error*, say
how much football it costs — this repo's dominant failure has been discarding
correctly-tracked play for want of a label, not mislabelling it.

State what the problem is **not**, if a plausible reading of the evidence points
elsewhere. Most wrong turns here began by fixing the visible symptom.

## Why the obvious fix is not the fix

What has already been ruled out, by whom, with what evidence. Pull from
`CLAUDE.md` and previous records by number. If nothing has been ruled out yet,
say that explicitly rather than leaving the section out.

## What the literature says, and what transfers

The relevant work: paper, venue/year, arXiv id, public implementation. Then the
part that earns its place — **what differs between their setting and ours**, and
what that does to the number they report. Broadcast footage, professional
adults, a moving director-cut camera and full-frame players are all different
from a fixed wide-angle consumer camera watching children in one kit, and the
difference is usually larger than the method's reported gain.

Label every figure: *measured* (ours, with run and n), *reported* (theirs, with
dataset), or *expected* (your reasoning).

## Hypothesis

One falsifiable sentence. "X is the binding constraint, so relieving it moves Y
by at least Z."

## How it will be assessed

- **Held-out block / gold set:** which one, and why that flavour of
  generalization is the right question for this change.
- **Metric:** what is counted, over what population, at what n.
- **Baseline:** the current number on the same data, measured the same way.
- **Falsifier:** the result that would make us drop this.
- **What tuning happens where:** thresholds swept outside the protected block,
  reported once on the gold set. Never both on the same data.
- **Audit:** paste the `scripts/audit_heldout.py` verdict for the galleries and
  runs involved.

## Steps taken

What was actually run, in order, with commands and job numbers, so it can be
repeated. Include the things that failed on the way if they cost time — a dead
end recorded is a dead end nobody walks twice.

## Result

The numbers, in a table, against the baseline. Then, in prose: does the effect
clear the noise at this n? Read every headline number narrowly and say what it
does *not* show.

## Verdict and what changes

Adopted, rejected, or inconclusive, and what happens to the code, the defaults
and `CLAUDE.md` as a result. If rejected, the most valuable sentence in the whole
record is the one that stops the next person retrying it — write that sentence.

## What this opens up

The next question, stated so someone else could pick it up.
