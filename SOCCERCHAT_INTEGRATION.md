# Proposal: SoccerChat integration for youth-footage understanding

**Status:** scaffolded on branch `soccerchat-integration` (code + tests + HPC job
+ annotation UI). Model inference is GPU-gated and not run yet — this document is
the proposal plus the wiring to test it.

## What SoccerChat is

[SoccerChat](https://arxiv.org/html/2505.16630v1) (Gautam et al., 2025;
SimulaMet / Forzasys / UCF) is a soccer-specialised vision-language model:
`Qwen2-VL-7B-Instruct` with a LoRA adapter fine-tuned on **49,120 QA pairs**
derived from **SoccerNet-v2** broadcast footage, dense captions, ASR commentary
(SoccerNetEchoes), and referee foul triplets (SoccerNet-XFoul).

- **Weights:** [`SimulaMet/SoccerChat-qwen2-vl-7b`](https://huggingface.co/SimulaMet/SoccerChat-qwen2-vl-7b) (Apache-2.0)
- **Data:** [`SimulaMet/SoccerChat`](https://huggingface.co/datasets/SimulaMet/SoccerChat) (~85k train)
- **Input:** one 10-second clip sampled to **24 frames (2.4 fps)**, ≤100,352 px/frame, + a text prompt
- **Output:** free text — a description, or a class + justification
- **Runtime:** loaded as a PEFT LoRA over Qwen2-VL with transformers + peft + qwen-vl-utils (the checkpoint is a standard `peft_type: LORA` adapter; no ms-swift needed for inference)

### The hypothesis we're testing

SoccerChat is trained on **professional broadcast** video — tight tele lens,
on-screen graphics, human commentary. Our footage is **overhead Veo / youth**
video. This is the same domain-shift caveat the KpSFR note in `CLAUDE.md` raises
for calibration. So the question is empirical: *does a pro-trained soccer VLM
make sense of youth clips?* The integration below is built to answer that cheaply
and, when it's wrong, to capture the correction as training data.

## Design

SoccerChat takes `clip + question → text`, which is exactly the shape of the
existing Claude verifier ([`verify/claude.py`](src/soccer_vision/verify/claude.py))
and fits the pluggable event-source protocol
([`events/sources.py`](src/soccer_vision/events/sources.py)). Two integration
points, primary first:

### 1. Clip verifier / captioner  (primary — `soccer-vision describe`)

A local analogue of `verify/claude.py`: run SoccerChat on the 10 s clips the
pipeline **already extracts**, get a caption + predicted class per clip, and
reconcile it with the heuristic label. No sliding window, no re-processing — it
reuses `clips/` and annotates `annotations.json` in place.

Verdicts:

| Verdict | Meaning |
|---|---|
| `CONFIRMED` | SoccerChat's class maps to the same canonical label. |
| `PLAUSIBLE` | Candidate is a `goal_kick` and SoccerChat said Kick-off / Ball-out (it has no goal-kick class) — consistent but unconfirmed. |
| `REJECTED`  | SoccerChat predicts a different, incompatible label. |
| `UNKNOWN`   | SoccerChat produced nothing mappable. |

Implementation: [`verify/soccerchat.py`](src/soccer_vision/verify/soccerchat.py).
Heavy imports (torch / transformers) are deferred, so importing the module and
running the offline tests needs no GPU.

### 2. Sliding-window spotter  (optional — `SoccerChatSource`)

Registered in `events/sources.py` as an **opt-in** source (never fires on a plain
`process` run; must be named in `config['sources']`). It slides a 10 s window
across the proxy, classifies each into the 16 SoccerNet classes, and emits
events. Heavier and weak on youth footage until fine-tuned — kept as a stub-grade
path behind the same interface as `TackleSource`.

### Label mapping

SoccerChat's 16 SoccerNet classes map onto our canonical taxonomy
([`events/labels.py`](src/soccer_vision/events/labels.py)) via
`SOCCERCHAT_TO_LABEL`. Notable point:

> **There is no "goal kick" class.** SoccerNet folds goal kicks into
> *Kick-off* / *Ball out of play*. Youth matches are goal-kick-heavy and it's our
> primary heuristic ([`events/set_piece.py`](src/soccer_vision/events/set_piece.py)),
> so SoccerChat can't directly confirm it — hence the `PLAUSIBLE` verdict. These
> clips are the highest-value ones to hand-annotate.

## The youth ground-truth loop

Because the pipeline's labels and SoccerChat's guesses are both imperfect on
youth video, the payoff is a fast **review-and-correct** loop that doubles as a
training-set builder:

```
process → clips + events
   → describe   (SoccerChat caption + class per clip)         [GPU]
   → annotate   (Label Studio project, predictions pre-filled)
   → you confirm/correct one choice per clip
   → annotate --export → JSONL fine-tune set → LoRA-adapt SoccerChat to youth
```

Label Studio setup and the exact commands are in
[`label_studio/README.md`](label_studio/README.md).

## How to run it

```bash
# 1. Process a match (existing pipeline)
soccer-vision process data/match-....mp4 --out-dir runs

# 2. SoccerChat over the clips (GPU; smoke-test 5 clips first)
sbatch training/slurm/soccerchat_describe.sbatch runs/<match_id> 5

# 3. Build the pre-filled Label Studio review project
soccer-vision annotate --run runs/<match_id>

# 4. After annotating, export youth fine-tune data
soccer-vision annotate --export export.json \
  --clips-root runs/<match_id>/clips --finetune-out youth_soccerchat.jsonl
```

Install extras: `uv sync --extra soccerchat` (transformers + peft + qwen-vl-utils;
~16 GB weights on first run) and `uv sync --extra annotate` (Label Studio).

## What landed on this branch

| Area | File |
|---|---|
| Canonical label taxonomy (single source of truth) | `src/soccer_vision/events/labels.py` |
| SoccerChat verifier/captioner | `src/soccer_vision/verify/soccerchat.py` |
| Optional spotter source | `SoccerChatSource` in `src/soccer_vision/events/sources.py` |
| `describe` CLI | `src/soccer_vision/cli/describe.py` |
| Label Studio logic (config, tasks, export) | `src/soccer_vision/annotate/label_studio.py` |
| `annotate` CLI | `src/soccer_vision/cli/annotate.py` |
| Clip ↔ event pairing | `parse_clip_name` / `pair_events_with_clips` in `src/soccer_vision/clips/extract.py` |
| HPC job | `training/slurm/soccerchat_describe.sbatch` |
| Annotation walkthrough | `label_studio/README.md` |
| Optional deps | `soccerchat`, `annotate` extras in `pyproject.toml` |
| Offline tests | `tests/test_soccerchat.py`, `tests/test_label_studio.py` |

## First smoke-test findings (6 clips, this match)

Ran `slurm/submit_soccerchat_smoke.sh` on 6 evenly spaced 10s clips from
`match-saints-16b-pre-mls-next-2026-04-26.mp4` (L4 GPU, ~1 min on warm cache).

**The plumbing works:** base model + LoRA load on transformers 5.x, decord feeds
video frames, fluent soccer-specific output comes back and maps to the taxonomy.

**The model does not reliably read this youth footage** — the predicted domain
shift, concretely:

- **Class collapse:** 4 of 6 clips → "Kick-off" (defaulting).
- **Class ↔ caption contradictions:** e.g. one clip classed *Direct free-kick*
  but captioned as a corner; another classed *Kick-off* but captioned as a shot +
  save.
- **Invented broadcast tropes:** "linesman raises his flag for offside",
  "keeper's remarkable save", "shot from 30 meters" — a fixed overhead Veo camera
  supports none of these; it's applying professional-broadcast priors.
- **`confidence` is not meaningful** — it is only an exact-class-echo proxy (the
  model exposes no calibrated score through generation).

**Takeaway:** usable as a *caption / hypothesis aid for human review*, not as an
automated labeler on youth video. This is exactly what the Label Studio
correct→fine-tune loop is for. Next steps to consider: (a) LoRA-fine-tune on
corrected youth clips via the `annotate --export` JSONL; (b) collapse
classify+describe into one call to cut cost and the class/caption contradiction;
(c) keep it human-in-the-loop only.

## Caveats / open questions

- **Domain shift is the whole point** — expect misses on overhead framing. Track
  the `CONFIRMED / PLAUSIBLE / REJECTED / UNKNOWN` mix from a smoke run before
  trusting it for anything automated.
- **Inference stack:** the wrapper loads the LoRA with transformers + peft (not
  ms-swift, whose 4.x API dropped the `swift.llm` namespace the first attempt
  used). Remaining risk is Qwen2-VL behaviour under transformers 5.x — the smoke
  job is what validates end-to-end generation. Fine-tuning to youth footage still
  uses ms-swift, which consumes the `annotate --export` JSONL.
- **`SoccerChatSource` (spotter)** is untested on GPU and off by default — treat
  as experimental until the verifier path proves transfer.
- **No audio:** Qwen2-VL has no audio branch, and Veo clips rarely carry useful
  commentary anyway, so the ASR half of SoccerChat's training isn't exploited.
