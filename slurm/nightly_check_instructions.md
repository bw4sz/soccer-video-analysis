# Nightly check — instructions for the local cron job

You are running unattended, overnight, on the HPC login node, with real
access to `/orange`, `/home/b.weinstein/logs/`, and SLURM (`sacct`,
`squeue`). Your job is to close the loop the nightly cloud brainstorm opened:
find out what actually happened to jobs launched since yesterday, record it,
and report back to the GitHub issues that proposed them. You do not launch
new jobs — that stays a human decision.

## Steps

1. `git pull --ff-only` on the current branch first, in case anything changed
   upstream. If it isn't a fast-forward, stop and log why instead of forcing
   anything.
2. Find jobs relevant to this project that finished recently:
   `sacct -u b.weinstein -S now-2days -o JobID,JobName,State,Elapsed,ExitCode`
   (or check `/home/b.weinstein/logs/` for recent files). Cross-reference
   against `slurm/job_ledger.md` — look for job IDs already logged there
   (a `## <jobid> — ...` heading) that have no `**Result.**` paragraph yet,
   or jobs visible in `sacct`/`squeue` history that never got a ledger entry
   at all (a job launched from a `nightly-proposal` issue's suggested command
   might not have been logged by hand).
3. For each job needing a result, read its actual output
   (`/home/b.weinstein/logs/<name>_<id>.out` or
   `slurm/logs/<name>_<timestamp>/`) and write a `**Result.**` paragraph in
   `slurm/job_ledger.md`, in the style of the existing entries: concrete
   numbers, not a vibe. If a job failed, say how it failed — that's a result
   too.
4. Check the open proposal queue: `gh issue list --label nightly-proposal --state open`.
   For each one, see whether `slurm/job_ledger.md` now shows a completed job
   matching what it proposed:
   - **Ran and you wrote a Result above** → `gh issue comment <n> --body "..."`
     summarizing the outcome (2-3 sentences, point at the ledger entry) and
     `gh issue close <n>`.
   - **Still running or not yet launched** → leave it open, no comment needed
     every night (don't spam a running job).
   - **Failed** → comment with what failed, leave open so a human decides
     whether to retry or abandon it.
5. If `slurm/job_ledger.md` changed, commit and push just that file:
   `git add slurm/job_ledger.md && git commit -m "<summary>" && git push`.
   If nothing changed, do nothing — an empty night is fine, don't force a
   commit.
6. Print a one-line summary of what you did (or "nothing to report") — this
   goes to the cron log file, which is the audit trail for this job.

## Hard rules

- Touch only `slurm/job_ledger.md`. Never edit, rewrite, or delete an
  existing ledger entry — only fill in a blank `Result` or append a new
  entry.
- Never `git push --force`, never touch branches other than the current one,
  never run any command that submits a new SLURM job.
- Never claim a result you didn't verify against the actual log output.
