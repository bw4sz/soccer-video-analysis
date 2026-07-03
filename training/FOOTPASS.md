# FOOTPASS / SN-PCBAS-2026 — primary ball-action-spotting target

**This is the canonical target for player-centric ball-action spotting.**
It supersedes the recokick MultiDimStacker path (whose only released weights were
2-class PASS/DRIVE — deleted). FOOTPASS predicts **who does what and when**:
every event is `(frame, team, jersey, class)`.

Upstream: <https://github.com/JeremieOchin/FOOTPASS> (a.k.a. Footovision/FOOTPASS)
— reference baselines for the **SoccerNet 2026 Player-Centric Ball-Action
Spotting Challenge**. Vendored (untracked) at `vendor/FOOTPASS/`.

Paper: <https://hal.science/hal-05373478v1> · Eval: Codabench comp 11232.

---

## Why this over recokick

| | recokick (BAS) | **FOOTPASS (SN-PCBAS-2026)** |
|---|---|---|
| Classes | 2 pretrained (PASS/DRIVE) | **8**: Pass, Drive, Cross, Shot, Header, Throw-in, Tackle, Block |
| Per-event attribution | no | **team + jersey + role (13 roles)** |
| Fit to `events/associate.py` | weak | **strong — attribution is native** |
| Data access | SoccerNet NDA password (we lack it) | HuggingFace `gated: auto` + (maybe) NDA pw to unzip |
| Pretrained weights | 2-class only | none, but X3D backbone is Kinetics-pretrained |

54 matches (2023–24, top-5 leagues + UCL). Train 48 / Val 3 / Challenge 3;
102,992 events. Metric: F1 @ τ=0.15, ±12-frame tolerance.

---

## Baselines (architectures)

- **TAAD** — Track-Aware Action Detector. `X3D-S` video backbone
  (`torch.hub facebookresearch/pytorchvideo x3d_s`, Kinetics-pretrained) +
  `roi_align` over per-player **tracklets** (bbox sequences). Clip length 50,
  4 tracklets/sample. `train_TAAD_Baseline.py`.
- **TAAD + GNN** — adds a spatio-temporal graph over players (needs
  `torch_geometric`). `train_GNN.py`.
- **TAAD + DST** — Denoising Sequence Transduction, game-level reasoning on top
  of TAAD predictions. `run_TAAD_on_matches.py → NPpreds2HDF5.py → train_DST.py`.

All output `(frame, team, jersey, class[, score])`.

---

## Data access (HuggingFace path — chosen)

Dataset: <https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026>
(`gated: auto`, not private). Contains tactical data **and** videos:

```
tactical_data_{TRAIN,VAL,CHALLENGE}.zip   tactical_data_format.txt
videos_352x640_{TRAIN,VAL,CHALLENGE}.zip  (small — start here)
videos_fullHD_{TRAIN_01..05,VAL,CHALLENGE}.zip  (large; TRAIN split in 5 parts)
```

**One-time setup:** create an HF account → accept terms on the dataset page
(auto-granted) → log in. Anonymous access 401s.

> ⚠️ The `hf` on `$PATH` is the project `.venv` one and is **broken**
> (`huggingface_hub` 1.13 CLI vs typer mismatch). Use the **blue env's** working
> CLI, or just export a token:
> ```bash
> /blue/ewhite/b.weinstein/envs/soccer-vision/bin/hf auth login   # writes ~/.cache/huggingface/token
> # or, no CLI:
> export HF_TOKEN=hf_xxxxxxxx
> ```
> Login persists in `~/.cache/huggingface/token`, shared by all envs, so the
> fetch script (run with the blue-env python) picks it up.

**Extraction (confirmed):**
- **Tactical data zips are NOT encrypted** — extract normally. `tactical_data_TRAIN.zip`
  yields `train_tactical_data.h5` (~8.8 GB), `VAL` yields `val_tactical_data.h5`.
- **Video zips ARE WinZip-AES encrypted** (compress method 99). Python's stdlib
  `zipfile` and the system `unzip` **cannot** decrypt AES — you need **`pyzipper`**
  (`uv pip install pyzipper`, already added to the blue env). The extraction
  password is the standard SoccerNet one — verified working and already stored in
  the gitignored `.soccernet_token`, so `footpass_fetch_data.py` uses it
  automatically. No separate NDA password is needed for these HF-hosted zips.
  (Override with `--zip-password` if needed.)

**Fetch (this repo)** — use the blue-env python (has `huggingface_hub` 1.21):
```bash
BLUE=/blue/ewhite/b.weinstein/envs/soccer-vision/bin
$BLUE/hf auth login                                   # once
$BLUE/python scripts/footpass_fetch_data.py           # train+val, 352x640, into
                                                      # /blue/.../soccer-vision-data/footpass
$BLUE/python scripts/footpass_fetch_data.py --resolution fullHD --splits train val challenge
$BLUE/python scripts/footpass_fetch_data.py --list-only   # preview file list (no token needed)
```

The script builds a complete `data_root`: copies the repo-shipped
`data/TAAD_sample_list.json` + `playbyplay_GT/`, and extracts tactical zips →
`data/`, video zips → `videos/game_<idx>.mp4`.

Expected layout the TAAD dataloader (`utils/TAAD_Dataset.py`) reads:
```
<data_root>/
├── data/train_tactical_data.h5   val_tactical_data.h5   TAAD_sample_list.json
├── videos/game_<n>.mp4                 # per-match (both halves in one file)
└── playbyplay_GT/playbyplay_{train,val}.json
```
> **Verified** (`/blue/.../soccer-vision-data/footpass`): filenames match; h5 keys
> are per-half `game_<n>_H<half>` (96 train / 6 val) and all resolve; videos are
> per-match and the loader derives them via `curr_key.split('_')[1]` →
> `game_<n>.mp4`. Dataset builds **6811 train / 2358 val** samples. Note the ROI
> coords are in fullHD space and scaled by 1080/352 for the 352×640 videos.

---

## Environment (differs from soccer-vision env)

FOOTPASS needs: Python 3.11, PyTorch 2.1, torchvision 0.16, **decord 0.6**,
**albumentations 2.0.8**, **h5py 3.14**, opencv; `torch_geometric 2.6.1` only for
the GNN variant. Build a dedicated env/container on HiPerGator rather than
reusing the soccer-vision env (which is torch 2.12). X3D-S weights download on
first `torch.hub.load`.

---

## Train (once data + env are ready)

```bash
cd vendor/FOOTPASS
python train_TAAD_Baseline.py --data_root /blue/ewhite/b.weinstein/soccer-vision-data/footpass \
    --run_path runs/taad_$(date +%d%m%Y) --epochs 20 --batch_size 6
# then optionally: train_GNN.py / (run_TAAD_on_matches → NPpreds2HDF5 → train_DST)
```
A SLURM wrapper (`training/slurm/train_footpass_taad.sbatch`) targets hpg-turin
(L4). NOTE: the upstream train script had Windows path separators in the
checkpoint paths (`checkpoints\best_model.pt`) — patched to `os.path.join` in our
vendored copy so checkpoints land in `<run_path>/checkpoints/`.

---

## Visualization (roboflow supervision)

`scripts/footpass_visualize.py` overlays player boxes (coloured by team), jersey
numbers, and action labels onto the broadcast video and writes an mp4.

```bash
python scripts/footpass_visualize.py --game game_18_H1 --split val \
    --start-frame 15811 --num-frames 350 --out footpass_gt.mp4
```

Ground-truth demo rendered at
`/blue/.../soccer-vision-data/footpass/viz/footpass_gt_game18_632s.mp4`. Same tool
takes model predictions once trained (swap CLS with predicted class per
tracklet). Reminder: **team + jersey come from the tracklets, not TAAD** — TAAD
only predicts the action class.

---

## Running on / fine-tuning to OUR footage

**Key architecture fact:** TAAD is *track-aware* — it consumes per-player
**tracklets** (bbox + team + jersey, the tactical h5) and predicts only the
action **class**. It does **not** run on raw video. So to use it on our Veo
footage we must first produce tactical data for our video. Two levels:

### A. Inference on our footage (no labels needed)
**Status: built** — `scripts/footpass_extract_tracklets.py` composes RF-DETR +
ByteTrack + `TeamClassifier` and writes the FOOTPASS h5 schema + a manifest.
Smoke-tested on our Saints match (frame 57000, ~1903s): 16 tracks, teams
clustered black/green; preview at
`/blue/.../footpass/our_footage/preview_smoke.mp4`. Runs on CPU (slow) — use
`--device cuda` via SLURM for full matches. **Observed quality gaps to fix:**
sideline spectators get detected as players (apply
`detection/field_filter.filter_spectators` / field-hull), team colour clustering
is imperfect, and there's no jersey OCR yet. The remaining piece is the
**TAAD inference adapter** that feeds these tracklets (rescaled per the manifest
WxH) through the trained model.

Pipeline pieces (all in `src/soccer_vision/`):
1. **Player detection** — RF-DETR (`julianzu9612/RFDETR-Soccernet`, already a
   ready lead) per frame → boxes.
2. **Tracking** — ByteTrack (`supervision`) → persistent `PLAYER_ID` tracklets.
3. **Team assignment** — `LEFT_TO_RIGHT` (0/1): jersey-colour clustering +
   which half/goal each team attacks.
4. **Jersey number** — OCR on the player crop (hard on youth/Veo; may be
   low-confidence — treat jersey as optional/best-effort).
5. **Assemble** into the FOOTPASS h5 schema (FRAME, PLAYER_ID, LEFT_TO_RIGHT,
   SHIRT_NUMBER, ROLE_ID, X/Y_POS, speeds, ROI_*, CLS=unknown) + a sample list.
6. Run the trained TAAD → predicted `(frame, team, jersey, class)` →
   `footpass_visualize.py`.

> The whole thing is bounded by tracklet quality. Our Veo camera is wide-angle
> overhead (vs broadcast); RF-DETR + ByteTrack should port, but **team/jersey are
> the fragile parts** — small, low-res numbers. Budget for this; jersey OCR may
> not be reliable and can be dropped to a per-team-only attribution.

### B. Fine-tuning on our footage (needs labels)
On top of A, you also need **ground-truth action labels** for our matches:
1. **Annotate** a handful of our matches with `(frame, team, jersey, class)` for
   the 8 actions — via Label Studio (`soccer_vision.annotate.label_studio`). This
   is the expensive human-in-the-loop step; start with the common classes
   (Pass, Drive) and set pieces.
2. **Convert** our annotations + tracklets into the FOOTPASS h5 +
   `TAAD_sample_list.json` format (a small adapter script).
3. **Resume** from the FOOTPASS-trained `checkpoints/best_model.pt` with a low LR
   — the train script already freezes X3D for the first 2 epochs
   (`set_x3d_freezing_schedule`), so the head adapts first, then the backbone
   unfreezes. Point `--data_root` at our-footage data.
4. **Evaluate** on a held-out our-footage split (F1 @ τ=0.15, ±12 frames) and
   iterate. Expect several annotate→train→inspect loops.

Recommended sequence: **(1) let the FOOTPASS baseline finish → (2) build A and
run it on one of our clips to see the raw domain gap → (3) decide how much
annotation B is worth** based on how far off the baseline is.

---

## Relationship to SoccerChat

Complementary, not competing: **TAAD is the structured event spotter**;
**SoccerChat is a natural-language describer/verifier** whose *structured* label
is unreliable on our Veo footage (it defaults to "Kick-off" and contradicts its
own captions). Full analysis with evidence from the smoke run:
[`FOOTPASS_vs_soccerchat.md`](FOOTPASS_vs_soccerchat.md).

## Integration into soccer-vision

`(frame, team, jersey, class)` maps cleanly onto the pipeline: `class` → the
soccer-vision taxonomy (`events/labels.py`), and `team`/`jersey` feed straight
into `events/associate.py` (which today infers team/track_id downstream). A
`FootpassSource` implementing the `EventSource` protocol
(`events/sources.py`) is the integration point — replaces the stubbed
`TackleSource`. 8-class output is far richer than the retired 2-class model.

Domain-gap caveat: broadcast top-5-league footage vs Veo/youth overhead — expect
to fine-tune before it performs on our footage.
