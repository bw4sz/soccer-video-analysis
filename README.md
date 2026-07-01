# soccer-vision

Open-source soccer video analysis toolkit. Replicates the useful parts of subscription products (Trace, Veo Editor, LongoMatch) without vendor lock-in — programmatic control over your own match footage.

**For:** coaches, analysts, and developers working with single-camera match video (any fixed/overhead/wide-angle source). Python 3.12+, CPU-viable, no cloud dependency.

Built on [supervision](https://github.com/roboflow/supervision) and the [OSL JSON](https://opensportslab.github.io/opensportslib/data/osl-json-format/) interchange format. Uses transformers-based detection (RF-DETR) — **no Ultralytics/YOLO**.

| Commercial feature | soccer-vision equivalent | Status |
|---|---|---|
| Trace — per-player highlight reels | player tracking + event→player association + reels | ✅ track-id based, ⏳ named roster |
| Veo Editor — AI events, team stats | set-piece detection + possession/distance/shot metrics | ✅ heuristics, ⏳ learned spotting |
| LongoMatch — manual tagging, playlists | OSL event store + clip DB + contact-sheet review | ✅ CLI, ⏳ desktop GUI |

---

## What's here now

The canonical pipeline runs end to end on CPU. One command turns a raw match into a proxy video, an event stream, team stats, and cut clips:

```bash
soccer-vision process match.mp4
```

Under the hood ([src/soccer_vision/cli/process.py](src/soccer_vision/cli/process.py)):

1. **Load** raw wide-angle video
2. **Virtual broadcast** — RF-DETR follow-cam crop → 16:9 `broadcast_proxy.mp4` (all later steps read the proxy, not the raw file)
3. **Ball detection** — RF-DETR fine-tuned on SoccerNet
4. **Player tracking** — ByteTrack (via supervision); foot positions + jersey-colour samples per track
5. **Field registration** — Hough-line homography → pixel-to-metres
6. **Event detection** — pluggable *sources* (today: set-piece heuristics for goal kick / corner / throw-in), then each event is **associated with the nearest player track and their team colour**
7. **Metrics** — distance covered, possession, shots, event counts (overall and per team)
8. **Persist** — SQLite match/event/clip DB + OSL JSON annotations
9. **Clips** — ffmpeg cut per event + contact sheets for review

Every match writes a self-contained run directory:

```
runs/{match_id}/
├── broadcast_proxy.mp4    # 16:9 follow-cam proxy
├── annotations.json       # OSL JSON events (label, frame, team, track_id)
├── stats.json             # team metrics
├── clips/                 # extracted clips
└── sheets/                # contact sheets for human/Claude review
runs/soccer_vision.db      # SQLite across all matches
```

**49 unit tests pass** (OSL round-trip, set-piece detection, possession, event association, team classification, sources, clip math). CI runs ruff + pytest on every PR — no GPU, no model weights.

---

## Two clip workflows: per-player vs team-action

Both share the same `process` run. They only diverge at the **selection** step, where events (already tagged with `track_id` + `team` in step 6) are filtered before cutting.

```mermaid
flowchart TD
    RAW[Raw match video<br/>wide-angle single camera] --> PROC

    subgraph PROC["soccer-vision process · shared pipeline"]
      direction TB
      S2[Virtual broadcast proxy<br/>RF-DETR follow-cam] --> S3[Ball detection<br/>RF-DETR]
      S3 --> S4[Player tracking<br/>ByteTrack]
      S4 --> S5[Field registration<br/>Hough homography]
      S4 --> TEAM[Team assignment<br/>jersey colour → blue / white]
      S5 --> EV[Event sources<br/>set-piece heuristics: goal kick, corner, throw-in]
      EV --> ASSOC[Associate each event with<br/>nearest player track + team]
      TEAM --> ASSOC
    end

    PROC --> RUN[(run dir:<br/>annotations.json · stats.json · clips.db)]
    RUN --> Q{Select and cut clips}

    Q -->|"1 · individual player"| IND["soccer-vision reel --run RUN --track 7 --event throw_in<br/>→ player #7's throw-ins only"]
    Q -->|"2 · team action"| GRP["soccer-vision extract --run RUN --events throw_in --team blue<br/>→ every blue throw-in"]
```

The filter (`label` / `team` / `track_id`) is a single shared function ([events/select.py](src/soccer_vision/events/select.py)), so any combination works — a whole team, one player, one event label, or all three at once:

```bash
# 1) Individual player — everything track-id 7 was closest to
soccer-vision reel --run runs/match_001 --track 7 --out player7.mp4

# 2) Team action — all throw-ins by the blue team
soccer-vision extract --run runs/match_001 --events throw_in --team blue

# combine: only #7's throw-ins into one reel
soccer-vision reel --run runs/match_001 --track 7 --event throw_in --out p7_throwins.mp4
```

> **Player identity is track-id based (v1).** A `track_id` is a ByteTrack lane, not yet a named jersey number. Stable per-player identity (jersey OCR / sn-gamestate / SAM3 masklets) is a later phase; `--player <name>` is stubbed until then. Teams are assigned by clustering torso colour into two groups and naming each (blue/white/…), so `--team` works today.

---

## Install

```bash
pip install -e .            # core (CPU)
pip install -e ".[gpu]"     # + SAM3 / GPU inference
pip install -e ".[gui]"     # + desktop reviewer deps (GUI not yet implemented)
pip install -e ".[dev]"     # + pytest, ruff, sphinx
```

**System requirement:** `ffmpeg` on PATH.

```bash
# full pipeline
soccer-vision process match.mp4 [--config examples/process_match.yaml] [--profile examples/profiles/saints-u10.yaml]
# proxy only
soccer-vision broadcast match.mp4 --out runs/match_001/
# select + cut clips (see workflows above)
soccer-vision extract --run runs/match_001/ --events goal_kick corner_kick
soccer-vision reel    --run runs/match_001/ --event goal_kick --out goal_kicks.mp4
# review / query with Claude (needs ANTHROPIC_API_KEY)
soccer-vision verify  --run runs/match_001/ --profile examples/profiles/saints-u10.yaml
soccer-vision ask "which team had more corners?" --run runs/match_001/
```

---

## Project structure

```
src/soccer_vision/
├── cli/          process · broadcast · extract · reel · verify · ask
├── io/           video (ffmpeg helpers) · osl (JSON 2.0) · project (run dirs)
├── broadcast/    virtual_cam — follow-cam proxy
├── detection/    rfdetr · ball · field_filter (spectator removal)
├── tracking/     bytetrack ✅ · teams ✅ · sam3 ⏳ · gamestate ⏳
├── registration/ hough ✅ · sn_calib ⏳ · kpsfr ⏳
├── events/       set_piece ✅ · phases ✅ · sources ✅ · associate ✅ · select ✅ · spotting ⏳
├── metrics/      distance · possession · shots · heatmap
├── store/        db (SQLite) + schema.sql
├── clips/        extract (ffmpeg cut) · reels (concat)
├── verify/       sheets (contact sheets) · claude (API)
├── profiles/     loader (YAML roster / IDP)
└── gui/          ⏳ empty — PySide6 reviewer planned

training/         sn_calib/ (calibration) · sn_spotting/ (T-DEED tackle model) + SLURM sbatch
docs/             Sphinx → Read the Docs
tests/            49 tests + video fixtures
```

`SOCCER_VISION_SPEC.md` is the full architecture spec and phase plan. `CLAUDE.md` documents the older two-script prototypes ([detect_actions.py](detect_actions.py), [extract_clips.py](extract_clips.py), [register.py](register.py)) still usable standalone.

---

## What's left to build

Roughly in priority order:

- **Learned event spotting** — [events/spotting.py](src/soccer_vision/events/spotting.py) and the `TackleSource` in [events/sources.py](src/soccer_vision/events/sources.py) are interface-only. Train/export a T-DEED model ([training/sn_spotting/train_teamspotting.py](training/sn_spotting/train_teamspotting.py)) and load it — association and clip code need no changes to gain a team-aware `tackle` label.
- **Stable player identity** — jersey OCR / sn-gamestate / SAM3 so `--player <name>` resolves to a real roster number instead of a track id ([tracking/sam3.py](src/soccer_vision/tracking/sam3.py), [tracking/gamestate.py](src/soccer_vision/tracking/gamestate.py)).
- **Better registration** — neural calibration ([registration/sn_calib.py](src/soccer_vision/registration/sn_calib.py), [registration/kpsfr.py](src/soccer_vision/registration/kpsfr.py)) for footage where Hough lines are weak.
- **Desktop reviewer** — PySide6 timeline / clip bin / stats tabs ([gui/](src/soccer_vision/gui/) is empty).
- **More event detectors** — free kicks, kickoff, substitutions; each is a new `EventSource`.
- **Packaging** — Read the Docs build, PyPI release, example notebooks.

---

## How to get involved

Early-stage — high-leverage contributions right now:

1. **New event detectors** — implement the `EventSource` protocol ([events/sources.py](src/soccer_vision/events/sources.py)); it plugs in behind association + clip selection with zero downstream changes.
2. **Registration for non-broadcast cameras** — improve Hough fallback or wire a calibration model.
3. **Test fixtures** — short anonymized clips with known events grow the CI suite.
4. **Try it on your footage** and file issues on detection accuracy — set-piece thresholds are tuned for youth 7v7 (55×36 m field) and need real-world data.

Dev loop: `pip install -e ".[dev]"` → `ruff check src/ tests/` → `pytest -q`. Issues and PRs welcome.

## License

AGPL-3.0-or-later.
