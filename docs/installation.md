# Installation

Python 3.12+ and `ffmpeg` on your `PATH`. A GPU is optional — the whole
pipeline runs on CPU, just slower.

```bash
git clone https://github.com/bw4sz/soccer-video-analysis.git
cd soccer-video-analysis
pip install -e .
```

Check it:

```bash
soccer-vision --help
```

## Extras

Install only what you need — several of these pin conflicting versions of
`numpy` / `torch`, so **use at most one of `gpu`, `annotate`, `soccerchat` and
`broadcast` per environment**.

| Extra | Install | What it adds |
|---|---|---|
| *(core)* | `pip install -e .` | detect, track, teams, clips — everything on the highlight pages |
| `identify` | `pip install -e ".[identify]"` | jersey-number OCR (PARSeq) |
| `annotate` | `pip install -e ".[annotate]"` | Label Studio, for teaching it your squad |
| `gpu` | `pip install -e ".[gpu]"` | CUDA inference |
| `harvest` | `pip install -e ".[harvest]"` | pull CC-BY youth clips off YouTube |
| `train` | `pip install -e ".[train]"` | action-spotting training deps |
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, sphinx |

Naming players **by appearance** needs no extra — the re-ID weights are fetched
on first use into `~/.cache/soccer_vision/reid/`.

## ffmpeg

```bash
# macOS
brew install ffmpeg
# Debian / Ubuntu
sudo apt install ffmpeg
# HPC module systems
module load ffmpeg
```

## Model weights

Fetched automatically on first run, then cached:

| Model | Source | Cached to | Size |
|---|---|---|---|
| RF-DETR detector | `julianzu9612/RFDETR-Soccernet` (HuggingFace, ungated) | HF cache | ~500 MB |
| OSNet re-ID | [sportsreid](https://github.com/shallowlearn/sportsreid) Google Drive | `~/.cache/soccer_vision/reid/` | ~9 MB after stripping |
| PARSeq OCR | `torch.hub` | torch hub cache | ~100 MB |

The published re-ID checkpoint is ~1 GB because it carries a 161k-identity
classifier head. That head is discarded on first load and only the backbone is
cached — those identities aren't your players, and yours live in the gallery.

## First run

```bash
soccer-vision process data/match.mp4 --match-id match_001
```

A 60-minute match takes roughly **1.6 hours on CPU**, most of it video decoding
rather than detection. It writes a self-contained run directory:

```
runs/match_001/
├── broadcast_proxy.mp4    the video every later step reads
├── tracks.json            where every player was, and which team
├── ball_track.json        where the ball was
├── annotations.json       detected events (empty today — see below)
├── stats.json             possession and team counts
└── clips/ · sheets/       extracted clips + review contact sheets
runs/soccer_vision.db      match records across all your videos
```

:::{note}
`annotations.json` comes out **empty**, and `process` says so. No action
detector ships today, so there are no `pass` / `shot` events to write. Detection,
tracking, team assignment and the ball track are all unaffected, and the
highlight pages work off those. See [Training new detectors](training-detectors.md).
:::

## GPU / cluster

Detection is the only step that wants a GPU. On SLURM:

```bash
sbatch slurm/submit_process.sh data/<match>.mp4 <match-id> examples/profiles/<team>.yaml
```

Pass `--device cuda` or `--device mps` to `process` directly if you're not
going through the submit script.
