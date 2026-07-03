# Label Studio review for soccer-vision events

Annotate the youth clips the pipeline extracts using the **same event labels the
tool emits** ([`src/soccer_vision/events/labels.py`](../src/soccer_vision/events/labels.py)).
Each clip comes **pre-filled** with the pipeline's detected label — and, if you
ran `soccer-vision describe`, SoccerChat's caption + class — so you *confirm or
correct* one choice per clip instead of labelling from scratch. Your corrections
export directly to SoccerChat fine-tune data.

```
process ──► clips/ + annotations.json
              │
   describe   ▼ (optional, GPU)     annotate --run
  SoccerChat caption + class ──► label_studio_tasks.json + labeling_config.xml
              │                              │
              ▼                              ▼
        soccerchat.json            Label Studio: confirm/correct
                                             │
                             annotate --export ▼
                                  soccerchat_finetune.jsonl  (youth training set)
```

## 1. Produce a run

```bash
soccer-vision process data/match-....mp4 --out-dir runs
# → runs/<match_id>/{annotations.json, clips/, ...}
```

Optionally add SoccerChat's read on each clip (needs the `soccerchat` extra + a
GPU — on HPC use the SLURM job):

```bash
sbatch training/slurm/soccerchat_describe.sbatch runs/<match_id>      # full run
sbatch training/slurm/soccerchat_describe.sbatch runs/<match_id> 5    # 5-clip smoke test
```

## 2. Build the Label Studio project

```bash
soccer-vision annotate --run runs/<match_id>
# writes runs/<match_id>/labeling_config.xml and label_studio_tasks.json
```

`--serve-root` defaults to the `runs/` base; keep it as the directory you point
Label Studio's local file server at (below).

## 3. Start Label Studio with local file serving

Clips are served from disk (not uploaded). Point the document root at the same
directory used as `--serve-root`:

```bash
pip install label-studio            # or: uv sync --extra annotate
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=/orange/ewhite/b.weinstein/soccer-video-analysis/runs
label-studio start
```

In the UI:
1. **Create Project** → *Labeling Setup* → *Custom template* → paste
   `runs/<match_id>/labeling_config.xml`.
2. **Import** → upload `runs/<match_id>/label_studio_tasks.json`.

The predictions ride along with the tasks, so each clip opens with its label
pre-selected. Click through, fixing the wrong ones.

### One-step alternative (push via the SDK)

If Label Studio is already running and you have an API token:

```bash
soccer-vision annotate --run runs/<match_id> --push \
  --ls-url http://localhost:8080 --ls-key "$LABEL_STUDIO_TOKEN"
```

## 4. Export corrections → fine-tune data

In Label Studio: **Export → JSON** (save as e.g. `export.json`). Then:

```bash
soccer-vision annotate --export export.json \
  --clips-root runs/<match_id>/clips \
  --finetune-out youth_soccerchat.jsonl
```

`youth_soccerchat.jsonl` holds `{"query", "response", "videos"}` records — the
schema SoccerChat's training notebooks consume — ready to LoRA-fine-tune the
model toward youth/Veo footage. See [`../SOCCERCHAT_INTEGRATION.md`](../SOCCERCHAT_INTEGRATION.md).

## Notes

- **Labels** come from one source of truth; regenerating the config after the
  taxonomy changes keeps the UI in sync.
- **`goal_kick`** has no SoccerChat equivalent (SoccerNet folds it into
  Kick-off / Ball-out), so `describe` marks such clips `PLAUSIBLE`, not
  `CONFIRMED`. These are exactly the clips worth annotating first.
- **Remote HPC:** run `label-studio start --host 0.0.0.0` on the node and use an
  SSH tunnel (`ssh -L 8080:<node>:8080 ...`) to reach the UI from your laptop.
