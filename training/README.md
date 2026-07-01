# Training Scripts

Scripts for SoccerNet tasks that need training from scratch or fine-tuning.

## Ready (pretrained weights available)

| Task | Package | Status |
|---|---|---|
| Detection (ball/player/ref/GK) | `rfdetr` + HuggingFace weights | Ready — `julianzu9612/RFDETR-Soccernet` |
| Multi-object tracking | `supervision` ByteTrack | Ready — no training needed |
| Field registration (Hough) | `soccer_vision.registration.hough` | Ready — classical CV |

## Needs Training

| Task | Doc | Framework | Data |
|---|---|---|---|
| **Ball action spotting (player-centric)** | **[`FOOTPASS.md`](FOOTPASS.md) — primary** | TAAD (X3D + tracklets) | SN-PCBAS-2026 (HuggingFace, gated) |
| Ball action spotting *(deprecated)* | `sn_spotting/train_action_spotting.py` | OSL-ActionSpotting + NetVLAD | SoccerNet-v2 labels + Baidu features |
| Team ball action spotting *(deprecated)* | `sn_spotting/train_teamspotting.py` | sn-teamspotting / T-DEED | SoccerNet BAS 2024 |

> **Ball-action spotting is standardized on FOOTPASS / SN-PCBAS-2026**
> (8 player-centric classes with team+jersey, `(frame, team, jersey, class)`).
> See [`FOOTPASS.md`](FOOTPASS.md). The recokick MultiDimStacker path was
> evaluated and dropped — its only released weights were 2-class (PASS/DRIVE),
> now deleted; the vendored repo is kept for reference only. The two spotting
> scripts above (OSL NetVLAD, T-DEED) ship no weights and are superseded.

## Needs Fine-tuning

| Task | Script | Data | Notes |
|---|---|---|---|
| Field calibration (neural) | `sn_calib/train.py` | SoccerNet calibration-2023 (~400 broadcast images) | Pretrained DeepLabv3 on Google Drive; our wide-angle footage has large domain gap |

## SLURM Scripts (HiPerGator)

| Script | Task | Partition | GPUs | Time |
|---|---|---|---|---|
| `slurm/train_calibration.sbatch` | sn-calibration fine-tuning | hpg-turin | 1 | 8h |
| `slurm/train_action_spotting.sbatch` | Action spotting (NetVLAD) | hpg-turin | 1 | 24h |
| `slurm/run_pipeline.sbatch` | Full pipeline on a video | hpg-turin | 1 | 4h |

## Quick Start

```bash
# 1. Test ready leads on sample footage (local)
python scripts/test_leads.py

# 2. Run full pipeline on HPC
sbatch training/slurm/run_pipeline.sbatch /path/to/match.mp4

# 3. Train calibration on HPC
sbatch training/slurm/train_calibration.sbatch

# 4. Train action spotting on HPC
sbatch training/slurm/train_action_spotting.sbatch
```

## sn-calibration Notes

The sn-calibration repo provides:
- **Data**: Downloadable via SoccerNet SDK (`calibration-2023` task, ~400 broadcast images with line annotations)
- **Pretrained weights**: DeepLabv3 checkpoint on Google Drive (segmentation backbone only; camera parameter estimation is classical geometry)
- **No training script**: We wrote `sn_calib/train.py` wrapping their dataloader

Key considerations:
- Broadcast → wide-angle domain gap is the main challenge
- Original repo pins PyTorch 1.10; our script uses current PyTorch
- SoccerNet data requires agreeing to a research-use NDA
- The geometry step supports radial distortion (up to 6 coefficients), which helps for wide-angle
