# FOOTPASS/TAAD vs SoccerChat — roles in the pipeline

Grounded in the SoccerChat smoke run on our own Veo youth footage
(`slurm/logs/soccerchat_smoke_20260701_161722/`, 6 clips from
`match-saints-16b-pre-mls-next-2026-04-26.mp4`).

**TL;DR:** they solve different problems and are complementary. **TAAD is the
event *spotter*** (structured, frame-precise, player-attributed) — the detection
backbone. **SoccerChat is a natural-language *describer/verifier*** — useful for
human-facing captions and clip sanity-checks, but its *structured* label is
unreliable on our footage and it cannot localize or attribute. Do **not** use
SoccerChat as a standalone classifier.

---

## What each one is

| | **FOOTPASS / TAAD** | **SoccerChat** |
|---|---|---|
| Type | Supervised action **spotter** (X3D-S + roi_align over player tracklets) | Video **LLM** (`SimulaMet/SoccerChat-qwen2-vl-7b`, Qwen2-VL-7B) |
| Output | `(frame, team, jersey, class)` per event | Free-text caption + one forced class per clip |
| Classes | 8: Pass, Drive, Cross, Shot, Header, Throw-in, Tackle, Block | SoccerNet set-piece taxonomy (prompted) |
| Temporal localization | **Frame-level** (±12-frame metric) | **None** — whole 10s clip → 1 label |
| Player/team attribution | **Native** (team + jersey + role) | Only in prose, unreliable |
| Confidence | Real per-class scores (τ=0.15 + NMS) | **Constant 0.9** (not a real signal) |
| Training | Required — data now in hand | Zero-shot (no training) |
| Inference deps | Needs **tracklets** upstream (RF-DETR + ByteTrack) | Just the clip |
| Cost | Tiny model (~3M param head on X3D-S); cheap | 7B VLM, ~16 GB weights (L4 OK), ~10s/clip |
| Fit to `events/associate.py` | **Direct** — attribution is the output | Poor — caption text only |

---

## Observed SoccerChat behaviour on our Veo youth footage

From `smoke/results.json` (clips uniformly sampled every ~10 min, **not**
event-triggered):

| clip | SoccerChat class | its own caption says… | verdict |
|---|---|---|---|
| 1 (01:00) | Kick-off | "dribble past opponent, ball out of play" | label ≠ caption |
| 2 (11:14) | Corner | "corner kick, overhit, out of play" | plausible |
| 3 (21:28) | Direct free-kick | "corner kick, intercepted by defender" | label ≠ caption |
| 4 (31:43) | Kick-off | "possession in midfield… throw-in or passing" | label ≠ caption |
| 5 (41:57) | Kick-off | "**powerful shot 30m, keeper saves**" | label ≠ caption (a shot!) |
| 6 (52:12) | Kick-off | "**throw-in, linesman flags offside**" | label ≠ caption |

**Failure modes:**
- **"Kick-off" default** — 4/6 open-play windows collapse to Kick-off. Forcing a
  single set-piece label on an arbitrary 10s window has no good answer, so it
  guesses the most generic restart.
- **Label contradicts its own caption** — the prose often contains the *real*
  event (shot+save, throw-in+offside, dribble-out) while the structured class is
  wrong. The description is the signal; the classification is noise.
- **Constant 0.9 confidence** → no ranking/thresholding possible.
- **No localization/attribution** — can't say *when* in the clip or *which player*.

**But the captions are genuinely good** — they capture semantics TAAD's 8 flat
classes can't (e.g. "keeper makes a remarkable save", "linesman raises flag for
offside"). That is SoccerChat's real value.

---

## How they fit together in soccer-vision

```
video ─▶ RF-DETR + ByteTrack ─▶ tracklets ─▶ TAAD  ─▶ (frame, team, jersey, class)   ◀── primary detection
                                                          │
                                          candidate clips │
                                                          ▼
                                                    SoccerChat  ─▶ caption + human-readable verify   ◀── description layer
```

- **TAAD = detection.** Structured, attributed, frame-precise events feed
  `events/associate.py` and clip selection directly. This is what replaces the
  stubbed `TackleSource` (`events/sources.py`).
- **SoccerChat = description/verification.** Run it *on TAAD's candidate clips*
  (event-triggered, not uniform sampling) to (a) generate human-facing captions
  and (b) flag obvious misfires — never as the classifier of record. This matches
  its existing role as the `soccer-vision describe` verifier
  (`verify/soccerchat.py`), not the `SoccerChatSource` spotter path.

## Caveats for both

- **Domain gap:** both were trained on professional broadcast; our footage is
  wide-angle Veo youth. TAAD needs fine-tuning on our footage to perform; the
  montage shows SoccerChat already degrades here.
- **TAAD needs tracklets at inference** — its accuracy is bounded by the upstream
  detector/tracker. Budget for that dependency.
