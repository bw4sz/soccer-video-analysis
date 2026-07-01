# Ball Action Spotting (recokick / MultiDimStacker)

This is the **canonical ball-action-spotting model** for soccer-vision. We
standardize on it and retire the two other spotting scaffolds
(`sn_spotting/train_action_spotting.py` = OSL NetVLAD,
`sn_spotting/train_teamspotting.py` = T-DEED) because those ship **no weights**,
whereas recokick provides pretrained checkpoints and trains end-to-end from
video (matching the footage we are collecting).

Upstream: <https://github.com/recokick/ball-action-spotting> — a fork of Ruslan
Baikulov's [1st-place SoccerNet Ball Action Spotting 2023 solution](https://github.com/lRomul/ball-action-spotting),
adapted as the **2024 baseline** via transfer learning (focal loss gamma=1.0,
alpha=0.5, initialized from stage-2 weights of the 2023 winner).

Vendored (untracked, see root `.gitignore`) at:
`vendor/ball-action-spotting/`

---

## Architecture

`MultiDimStacker` (`src/models/multidim_stacker.py`) — a single-stage 2.5D + 3D
model:

- **2D backbone:** `tf_efficientnetv2_b0.in1k` via `timm`, run on a stack of
  **15 grayscale frames** (`frame_stack_size=15`, `frame_stack_step=2`), grouped
  into sub-stacks of 3 (`stack_size=3`).
- **3D head:** 4 lightweight 3D blocks fuse temporal features across the stack,
  then GeM pooling → per-class logits (sigmoid, multi-label).
- **Training framework:** `pytorch-argus` (checkpoints are `*.pth` holding both
  model params and `state_dict`; load with `argus.load_model(path)`).
- **Loss:** focal loss. **EMA** weights used at inference.

### Classes (12, SoccerNet BAS 2024)

```
PASS, DRIVE, HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT,
BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL
```

These match `TEAM_ACTION_LABELS` / `TEAM_LABEL_TO_SV` in
`training/sn_spotting/train_teamspotting.py` — reuse that dict to map into the
soccer-vision taxonomy (`src/soccer_vision/events/labels.py`).

### Input expectations

- Video: **720p mp4 @ 25 fps**, one file per game dir named `720p.mp4`
  (the code asserts `fps == 25.0`; `RESOLUTION = "720p"`). Internal model
  `image_size = (1280, 736)`.
- SoccerNet BAS 2024 is single-half (`num_halves = 1`).

---

## Pretrained weights

Author's Google Drive folder (in the upstream README):
<https://drive.google.com/drive/folders/1mIu62cIdsRn3W4o1E5vRR8V5Q1B6HHoz>

| Bundle | What it is |
|---|---|
| `sampling_weights_001` | Ball-action model, 7-fold cross-val checkpoints |
| `action_sampling_weights_002` | Auxiliary action-spotting model (transfer source) |
| `ball_tuning_001` | Final transfer-learned experiment (configs present in repo) |

Expected on-disk layout (README):

```
data/
├── action/experiments/action_sampling_weights_002
├── ball_action/experiments/sampling_weights_001   # fold_0/*.pth … fold_6/*.pth
└── soccernet/spotting-ball-2024/england_efl/...    # videos (NDA)
```

`get_best_model_path()` picks the `*.pth` in a `fold_<n>/` dir with the highest
score encoded in the filename (`...-<score>.pth`).

Our copy lives under `vendor/ball-action-spotting/data/` (orange storage;
`*.pth` are gitignored anyway).

---

## HiPerGator reproduction plan

The repo is built to run **inside its Docker image** (`osaiai/dokai:23.05-vpf`)
and uses NVIDIA **VideoProcessingFramework (VPF) v2.0.0** for GPU NvDec video
decoding. Two frictions to plan around on HPG:

1. **VPF is hard to build outside Docker.** Options, in order of preference:
   - Convert their Dockerfile to an **Apptainer/Singularity** image on HPG
     (`apptainer build bas.sif docker://osaiai/dokai:23.05-vpf`, then
     `pip install -r requirements.txt` inside), **or**
   - Skip VPF entirely: the repo ships an `OpencvFrameFetcher`
     (`src/frame_fetchers/opencv.py`) alongside the NvDec one. Swapping
     `NvDecFrameFetcher → OpencvFrameFetcher` in `scripts/ball_action/predict.py`
     lets prediction run with plain OpenCV/FFmpeg decode — no VPF. (The fetcher
     still `.to(cuda)`s the frame; keep a GPU in the job or patch to CPU.)

2. **Hardcoded `/workdir`.** `src/constants.py` sets `work_dir = Path("/workdir")`
   (the container mount). Off-Docker, either symlink `/workdir` → the repo, run
   from a bind-mounted `/workdir`, or patch `constants.py` to read
   `$BAS_WORKDIR`.

### Weights-only path (no training — do this first)

```bash
# 1. weights already downloaded under vendor/ball-action-spotting/data/
# 2. NDA videos: SoccerNet spotting-ball-2024 (needs .soccernet_token)
python download_ball_data.py           # or the SoccerNet SDK
# 3. predict + evaluate the released ball_tuning_001 on the test folds
python scripts/ball_action/predict.py  --experiment ball_tuning_001
python scripts/ball_action/evaluate.py --experiment ball_tuning_001
```

### Full retrain / fine-tune to our footage

```bash
python scripts/ball_action/train.py    --experiment sampling_weights_001
python scripts/ball_action/predict.py  --experiment sampling_weights_001
python scripts/action/train.py         --experiment action_sampling_weights_002
python scripts/ball_action/train.py    --experiment ball_tuning_001   # transfer
python scripts/ball_action/predict.py  --experiment ball_tuning_001
python scripts/ball_action/evaluate.py --experiment ball_tuning_001
```

---

## Integration into soccer-vision

The inference hook exists but is stubbed:

- `src/soccer_vision/events/sources.py :: TackleSource` — `is_available()` is
  `False`, `detect()` returns `[]`. This is where a `RecokickSource` (or a
  generalized `BallActionSource` emitting all 12 classes) plugs in behind the
  `EventSource` protocol — association (`events/associate.py`) and clip code need
  **no changes**.
- `src/soccer_vision/events/spotting.py` is a label-map-only stub
  (`# Stub — Phase 5`).

**Adapter TODO:** wrap `MultiDimStackerPredictor` → emit soccer-vision event
dicts (`{label, frame, timestamp_s, confidence, ...}`), mapping the 12 classes
through `TEAM_LABEL_TO_SV`, then flip `is_available()` to true.

---

## Domain-gap caveat

Trained on **professional broadcast** England EFL footage at a fixed 720p/25fps
broadcast framing. Our Veo/youth footage is wide-angle overhead — expect weak
recall until fine-tuned. Verify the released weights reproduce on SoccerNet test
first (below), *then* assess on our footage before investing in a retrain.
