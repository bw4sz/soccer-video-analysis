# soccer-video-analysis Job Ledger

Append-only history of SLURM jobs submitted for *this* project. Scoped copy of
the cross-project ledger at `/home/b.weinstein/logs/job_ledger.md` — that file
mixes in MillionTrees/other projects, so this one exists to keep a project-
local view. Raw `.out`/`.err` stay in `/home/b.weinstein/logs/` (and detailed
per-run artifacts under `slurm/logs/<name>_<timestamp>/`); add an entry here
whenever a job is submitted for this repo.

Format:

```
## <JOBID> — <YYYY-MM-DD HH:MM> — <script>
Why: <goal/hypothesis behind this run>
Result: <outcome once known>
Next: <follow-up action>
```

## 36260512 — 2026-07-02 11:14 — training/slurm/train_footpass_taad.sbatch
Why: First full training run of the FOOTPASS TAAD baseline (player-centric ball-action
  spotting model) on the fetched SN-PCBAS-2026 tactical-cam data. 20 epochs, batch_size=6,
  hpg-turin GPU (22GB).
Result: FAILED after ~15h (2026-07-03 02:25) with `torch.cuda.OutOfMemoryError` during the
  epoch-3 forward pass (x3d resnet block in model_TAAD_baseline.py), right after epochs 1-2
  completed train+val cleanly. GPU had ~22GB total; crash came after 2 clean epochs, not on
  epoch 1, which points at a memory-growth/fragmentation issue (unreleased activations/cache
  across epochs) rather than a simple batch-size-too-large problem.
Next: Re-submit with either (a) smaller batch_size (try 4) to confirm it's a headroom issue,
  or (b) add `torch.cuda.empty_cache()` / check for retained graphs between epochs in
  train_TAAD_Baseline.py before assuming it's pure OOM-from-size. Nothing has been
  resubmitted since the crash — no soccer-vision job currently running or queued.

## 36258658 — 2026-07-02 10:29 — slurm/submit_trim_empty.sh
Why: Run `trim-empty` end-to-end on a real match (match-saints-16b-pre-mls-next-2026-04-26)
  using the RF-DETR-built ball track, to validate the dead-time cutting pipeline.
Result: COMPLETED in 55min. Ball visible in 92.5% of samples; cut 11 dead spans (10
  stationary, 1 mixed), removing 1.1min (2%) of a 53.4min match. Track/EDL/trimmed video
  saved under slurm/logs/trim_empty_20260702_102933/.
Next: Spot-check the trimmed output and EDL for correctness; if good, this is the
  reference invocation for running trim-empty on other matches.

## 36183406 — 2026-07-01 16:17 — training/slurm/soccerchat_describe.sbatch (smoke test)
Why: 3rd attempt at a SoccerChat GPU smoke test after switching the inference path from
  ms-swift to transformers+peft (commits be6cfcc, 59d58ce) — validate the VLM loads and
  produces taxonomy-mapped output end-to-end before building the Label Studio correct/
  fine-tune loop.
Result: COMPLETED (exit=0). Ran on 6 clips (L4 GPU); model loads and maps to taxonomy but
  is unreliable on youth/Veo footage — collapses to "Kick-off" class, gives class/caption
  contradictions, and invents broadcast-style details not in the actual video. Findings
  recorded in SOCCERCHAT_INTEGRATION.md (commit b47c2fd).
Next: Domain shift confirmed (youth/Veo vs. broadcast training data) — proceed with the
  planned Label Studio correct-then-fine-tune loop rather than trying zero-shot further.

## 36181836 — 2026-07-01 16:09 — training/slurm/soccerchat_describe.sbatch (smoke test, attempt 2)
Why: Retry after 36180850's ms-swift dependency failure; switched to loading the LoRA
  adapter directly via transformers+peft.
Result: FAILED (exit=1) — peft LoRA-injection hit an unsupported module type
  (only Linear/Embedding/Conv1d-3d/Conv1D/MultiheadAttention supported). Superseded by
  59d58ce ("Fix SoccerChat LoRA loading on transformers 5.x") and job 36183406.
Next: none — resolved by the follow-up fix and confirmed working in 36183406.

## 36180850 — 2026-07-01 16:00 — training/slurm/soccerchat_describe.sbatch (smoke test, attempt 1)
Why: First GPU smoke test of the SoccerChat integration (commit 2ab9d72).
Result: FAILED (exit=1) — ms-swift not installed / venv path mismatch on the compute node.
Next: none — abandoned ms-swift in favor of transformers+peft (commit be6cfcc); see 36181836.
