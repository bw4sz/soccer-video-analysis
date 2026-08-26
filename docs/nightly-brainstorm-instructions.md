# Nightly brainstorm — instructions for the cloud routine

You are running unattended, overnight, in the cloud. **You have no access to
`/orange`, no video, no SLURM, no GPU, and no `/blue` — only this git repo.**
Your job is to propose the next concrete step, not to take it. A human reads
your proposal the next morning on the HPC machine and decides whether to run
it.

## Read first, in order

1. `CLAUDE.md` — the whole thing. It is long on purpose: most obvious-looking
   ideas ("loosen the link gate," "cap exemplars per player," "cluster on
   lightness") have already been tried and measured, and it says why they
   failed. **Do not propose something CLAUDE.md already reports as ruled
   out.** If you're about to propose it, grep CLAUDE.md for the relevant
   keywords first.
2. `docs/research/index.md` and any record at `status: proposed` — those are
   open hypotheses someone already framed; don't duplicate one, extend it.
3. `heldout.yaml` — the spans nothing may learn from. Never propose enrolling
   a gallery from a block listed there.
4. `slurm/job_ledger.md` (read the last ~150 lines) — what actually ran
   recently and what it found.
5. Open GitHub issues: `gh issue list --state open --limit 50`. These are the
   named, scoped open problems (#19–#29 as of 2026-08). A good proposal often
   just picks one and states the smallest experiment that would move it.
6. The existing queue: `gh issue list --label nightly-proposal --state all --limit 20`.
   Read it before writing anything.

## The one rule that overrides everything else below

Check whether `runs/saints-u14g-full-30fps/heldout_gold/annotations.json`
exists in the repo (it won't show as a tracked file — check via
`gh api repos/bw4sz/soccer-video-analysis/contents/...` or note that `runs/`
is gitignored and you can't see it directly; instead check
`slurm/job_ledger.md` and recent commits for whether the gold set has been
scored, e.g. a run of `eval_link_ground_truth.py` or `audit_heldout.py`
against it). **If there's no sign the U14G gold set has been annotated and
scored yet, your only proposal is: finish annotating it.** Point at
`runs/saints-u14g-full-30fps/heldout_gold/ANNOTATING.md`. Everything else in
this file (issues #25, #28, #29, hubness correction, PRTreID, camera-pan
compensation) is downstream of that gold set existing — every accuracy number
in CLAUDE.md before it predates real held-out ground truth, so proposing
tuning work ahead of it just adds more numbers nobody should trust yet.

Once the gold set has clearly been scored (a commit or ledger entry
references its results), move on to proposing from the open issues.

## What makes a good proposal

- **One, at most two** per night. Don't flood the queue.
- **Actionable by a human with data access, tomorrow morning** — a concrete
  `sbatch` command or CLI invocation, not a vague direction. If you can't
  write the exact command, it isn't ready to propose.
- **Cites the specific number or finding that motivates it.** "Try X" is not
  a proposal; "CLAUDE.md measures Y at Z, issue #N asks for W, here's the
  command" is.
- **States what result would confirm or refute it**, in one sentence, so the
  human isn't left guessing what "worked" means.
- Respects every hard rule already in CLAUDE.md: never enroll from a
  `heldout.yaml` block, never loosen a gate on a coverage/count metric alone
  without a plan to verify by rendering actual frames, never treat an
  `UNAUDITABLE` gallery figure as a result.

## What to do with a proposal

Open one GitHub issue per proposal:

```
gh issue create \
  --title "$(date +%Y-%m-%d): <short description>" \
  --label nightly-proposal \
  --body "<rationale, 2-4 sentences><blank line><exact command><blank line>Confirms/refutes: <one sentence>"
```

If nothing new is worth proposing (the queue already has an open, unactioned
`nightly-proposal` issue, or everything actionable is already running),
**do nothing**. An empty night is a correct outcome, not a failure — don't
open an issue to have opened one.

Do not close or comment on existing issues — that's the next night's local
check step, after a human has actually run something. Do not modify any file
in the repo. Do not attempt to run, fetch, or download anything outside this
git checkout.
