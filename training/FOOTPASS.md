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
(auto-granted) → `hf auth login` (or set `$HF_TOKEN`). Anonymous access 401s.

**Caveat:** the FOOTPASS README says a password may be needed to *extract* the
zips ("the password that allows you to extract the files"). If unzip fails, pass
the SoccerNet NDA password via `--zip-password`. Our current `.soccernet_token`
is the **public** password `s0cc3rn3t` and will **not** work for NDA-gated
extraction — a real NDA password may still be required for the videos.

**Fetch (this repo):**
```bash
hf auth login
python scripts/footpass_fetch_data.py                 # train+val, 352x640, into
                                                      # /blue/.../soccer-vision-data/footpass
python scripts/footpass_fetch_data.py --resolution fullHD --splits train val challenge
python scripts/footpass_fetch_data.py --list-only     # preview file list
```

The script builds a complete `data_root`: copies the repo-shipped
`data/TAAD_sample_list.json` + `playbyplay_GT/`, and extracts tactical zips →
`data/`, video zips → `videos/game_<idx>.mp4`.

Expected layout the TAAD dataloader (`utils/TAAD_Dataset.py`) reads:
```
<data_root>/
├── data/train_tactical_data.h5   val_tactical_data.h5   TAAD_sample_list.json
├── videos/game_<idx>.mp4
└── playbyplay_GT/playbyplay_{train,val}.json
```
> Verify the extracted `.h5` names match `train_tactical_data.h5` /
> `val_tactical_data.h5`; rename if the zip uses different names.

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
A SLURM wrapper (`training/slurm/train_footpass_taad.sbatch`) should target a GPU
partition; X3D + video decode wants a decent GPU and fast disk for the videos.

---

## Integration into soccer-vision

`(frame, team, jersey, class)` maps cleanly onto the pipeline: `class` → the
soccer-vision taxonomy (`events/labels.py`), and `team`/`jersey` feed straight
into `events/associate.py` (which today infers team/track_id downstream). A
`FootpassSource` implementing the `EventSource` protocol
(`events/sources.py`) is the integration point — replaces the stubbed
`TackleSource`. 8-class output is far richer than the retired 2-class model.

Domain-gap caveat: broadcast top-5-league footage vs Veo/youth overhead — expect
to fine-tune before it performs on our footage.
