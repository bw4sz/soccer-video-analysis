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

## 37591635 — 2026-07-19 18:30 — slurm/submit_process_saints_u11.sh
Why: Full individual-player pathway on the new Saints U11 vs OVF match
  (data/SaintsU11_OVF_Jul192026.MP4, 3.1 GB). process → identify → summary →
  extract, to answer: how many action clips can we cut for the BLACK team's #6,
  and what event labels do they carry? Black is Saints' away kit (us), #6 = Simon
  Weinstein. Selected by number (--number 6 --team black); clips are correct even
  though the job ran identify without the roster profile.
  Run dir runs/saints-u11-ovf-2026-07-19.
Result: PARTIAL — process (stage 1) COMPLETED: 199 events, 182 tracks, 199 clips,
  34 sheets in ~49min wall (compute ~29.5min + clip/ffmpeg ~19min). But identify
  (stage 2) CRASHED at rc=1 on `ModuleNotFoundError: nltk` (PARSeq import chain),
  so jerseys.json was never written and stages 3-4 (#6 summary + #6/black halo
  clips) never ran. Event output is also degenerate: 198/199 labeled throw_in,
  mean conf 0.39; team colours came back blue/white (profile expects black/white)
  with 96% unknown. nltk was declared in the [identify] extra but the deployed env
  wasn't synced with it. Resumed by job 37614025.
Next: (a) fix in place — installed nltk into env, submitted resume job 37614025
  (stages 2-4 only, reuses process output). (b) Still open: degenerate event
  detector + wrong team clustering may leave #6/black selection near-empty even
  after identify succeeds; investigate separately.

## 37614025 — 2026-07-20 09:xx — slurm/submit_resume_saints_u11.sh
Why: Resume 37591635 after its stage-2 crash without recomputing process. Installed
  nltk (was missing from the deployed env though declared in [identify]); this job
  runs only identify -> summary -> extract against the existing run dir
  runs/saints-u11-ovf-2026-07-19 (tracks.json + broadcast_proxy.mp4 already on disk).
Result: COMPLETED (exit=0, 2:39). nltk fix worked — identify ran clean, wrote
  jerseys.json. OCR yield low: 15/182 tracks legible (8%), 167 unknown. Exactly one
  track voted #6 (track 165, conf 0.73, 3 obs). BUT extract produced ZERO #6/black
  clips ("No matching events found"). Root cause is upstream in process, not the
  crash we fixed: events in annotations.json aren't associated with players/teams —
  of 199 events only 9 carry a track_id and 8 carry a team (5 blue/3 white, zero
  black). So `--team black` matches 0 events and `--number 6`→track165 can't join
  (191/199 events have no track_id). The individual-player pathway can't work on
  this run's data regardless of invocation. (Also note: my edited step-3 summary
  didn't run — SLURM snapshots the script at submit time, so the old buggy summary
  ran and threw a harmless traceback; extract still ran.)
Next: The event↔track/team association gap is the real blocker for player-level
  slicing — candidate GitHub issue (with the degenerate throw_in detector). #6 IS
  identifiable (track 165); to get *something*, could cut around track 165's frames
  directly rather than via events. Team clustering also returns blue/white not
  black/white — separate process-stage bug.

## 36500443 — 2026-07-06 — slurm/submit_footpass_ours_ball.sh (branch footpass-track-continuity)
Why: Verify the new track-continuity filter end-to-end (not just logic-checked on an existing h5).
  Drops tracks present in <50% of frames within their lifespan or seen in <10 frames total —
  the flicker signature of supporters and ref<->player flip-flopping.
Result: COMPLETED (exit 0, 6m04s). In-pipeline [continuity] dropped 45/108 flickery tracks (200
  detections), matching the offline validation exactly. Everything else held: field-mask 221,
  referee 206, ball 559/600 (93%), teams black/orange, 12233 player-detections (200 fewer).
  TAAD events unchanged (36, 18 near-ball, 1 gated) — the junk tracks were short and never made
  TAAD's top-13 slots, so the win is a cleaner overlay (fewer flickering boxes in motion), not
  different predictions. Regenerated saints_live_overlay_ball.mp4. Filter committed as 4b05ecf.
Next: TAAD domain shift remains the real blocker (fine-tune/adapt). Optional: smooth referee
  removal further by voting a track's ref-vs-player class over its lifetime.

## 36468175 — 2026-07-06 — slurm/submit_footpass_ours_ball.sh (branch compare-taad-predictions)
Why: Add ball detection + tracking, referee removal, and a gentle ball-proximity gate to the
  pipeline, then re-run the live window. Extractor now does ONE RF-DETR pass/frame split into
  ball/players/referees (predict()), tracks the best on-field ball, and drops player boxes that
  overlap a referee box (IoU>0.45). Inference tags each event by ball distance and drops only
  far+weak ones (soft gate) so off-ball actions survive.
Result: COMPLETED (exit 0, 8m21s). Ball detected in 559/600 frames (93%, matches trim-empty's
  92.5%). Referee removal dropped 206 player boxes overlapping refs (RF-DETR does emit the ref
  class here). Teams black/orange, 108 tracks. TAAD: 36 events, 18 near-ball / 18 off-ball, only
  1 gated (far+weak) — gate is appropriately gentle. Class skew still block 19 / shot 8 (domain
  shift unchanged — the gate doesn't touch the classifier, by design). Ball/off-ball tags in
  predictions.json; ball drawn in annotated.mp4; new reel saved_live_overlay_ball.mp4.
Next: ref removal is intermittent (RF-DETR flips ref<->player frame to frame) — could smooth by
  voting over a track's lifetime. Real blocker remains TAAD domain shift (needs fine-tune/adapt).

## 36460253 — 2026-07-06 — slurm/submit_footpass_ours_maskoff.sh (branch compare-taad-predictions)
Why: Mask-OFF counterpart of 36450376 (same live window, --field-mask none) to compare TAAD
  predictions with vs without the turf gate on identical frames.
Result: COMPLETED (exit 0, 5m18s). Predictions BYTE-IDENTICAL to mask-on: all 33 events match
  frame/class/team (block 18, shot 8, ...). Reason: TAAD scores only the top-13 longest tracks
  per team; the 248 off-field detections are short spectator tracks that never make the cut. So
  the field mask is upstream hygiene (viz + team split in hard windows), ORTHOGONAL to the action
  head. Teams also split black/orange even without the mask here (live window has distinct kits +
  enough on-field players). Built a comparison artifact: mask on/off null result + prediction-vs-
  reality on 3 sample frames showing TAAD fires shot/block on any tight player cluster near the
  ball (false on midfield scrums, topically-right only on a real goalmouth). Work on branch
  compare-taad-predictions (off master; scripts already merged via PR #2).
Next: TAAD domain shift is the real blocker (gates can't fix it) — fine-tune/adapt on our footage,
  or gate its output on ball-proximity + goal-zone before trusting it.

## 36450376 — 2026-07-06 — slurm/submit_footpass_ours_smoke.sh (live window + field mask)
Why: Re-run the TAAD smoke on a genuine LIVE window (1150-1170s, frame 34466, inside the
  989-1269s trim keep-segment) with the new turf field-mask gate on, to fairly judge the
  model and prove the two upstream gates (spatial field mask + temporal trim) kill the
  edge cases from 36352646.
Result: COMPLETED (exit 0, 6m30s, no OOM — the decord-release fix worked). Field mask
  dropped 248 off-field detections (polygon 56% of frame). Team split FIXED as predicted:
  both-'black' -> 'black' vs 'orange' (masking sideline adults cleaned the colour
  clusters). TAAD: 33 events but a NEW collapse — block 18, shot 8, pass 3, then 1 each
  drive/cross/throw-in/header, tackle 0 (vs the dead window's throw-in collapse). So
  detection+teams are fixed but the action head still doesn't transfer to youth/Veo
  overhead footage — domain shift, not a bug (model is fine on its own val domain).
  Residual: far-touchline people still leak through the polygon's top edge; refs boxed as
  players. Built a status dashboard artifact + embedded key frames.
Next: (a) tighten field polygon top edge / per-frame mask; (b) ref/keeper as non-player;
  (c) wire trim keep-segments directly into the extractor loop; (d) the real open problem
  is TAAD domain shift — fine-tune on our footage or adapt to overhead view.

## 36352646 — 2026-07-04 — slurm/submit_footpass_ours_smoke.sh
Why: Smoke test of the completed TAAD model (36305311 best_model.pt, epoch 18) on OUR
  Veo footage (match-saints). Extract tracklets on a ~20s window (frame 38361, 600
  frames — the restart after the 14s stoppage at 1268-1283s in the trim EDL) via
  RF-DETR+ByteTrack (.venv), then run TAAD + render annotated.mp4 + key frames via the
  new scripts/footpass_infer_ours.py (footpass env). The adapter assigns each track to
  a per-team slot 1..13 (the network doesn't use ROLE semantics), working around
  run_TAAD_on_matches.py's ROLE_ID 1..13 grouping that empties our ROLE_ID=0 ROIs.
Result: RAN end-to-end (adapter works!) but OOM-killed (exit 125, host RAM 32GB) during
  the final cv2 render — predictions.json + 6 key frames + partial annotated.mp4 were
  written first, so the assessment is intact. Findings: (1) my window (1280-1300s) was
  still DEAD TIME — ball stationary in center circle, players lined up on the far
  touchline (a stoppage/subs), so no true live actions to detect. (2) Domain shift is
  real and multi-stage on Veo footage: RF-DETR tags spectators/coaches as players (t1 =
  an adult filming, people in camping chairs all boxed); team clustering FAILS (both
  kits read 'black' under harsh backlight → team_a/team_b meaningless); TAAD collapses
  to 'throw-in' (13/15 events), spuriously triggered by people clustered near the field
  boundary. Same domain-shift story as SoccerChat (job 36183406). Fixed the render OOM
  in footpass_infer_ours.py (release decord reader before cv2; bump --mem to 64GB on
  re-run).
Next: (a) re-run on a genuine LIVE-ACTION window (mid-possession, not a trim "removed"
  span) to fairly judge TAAD; (b) spectator/coach false detections + failed team split
  are upstream blockers worth fixing before trusting any action output on this footage.

## 36305311 — 2026-07-03 10:03 — training/slurm/train_footpass_taad.sbatch
Why: Resubmit after job 36260512's epoch-3 CUDA OOM. Root cause isn't a leak: the
  vendored `set_x3d_freezing_schedule` trains head-only (X3D frozen) for epochs 1-2,
  so autograd skips storing backbone activations; at epoch 3 the backbone unfreezes
  and full backprop through X3D-S needs much more activation memory on the 22GB L4
  — batch_size=6 no longer fits once that happens. Halved to batch_size=3 (AMP
  fp16 already in use, so batch size is the only lever left); run dir
  `taad_$(date)` under `/blue/.../footpass/runs/`.
Result: COMPLETED (exit 0) 2026-07-03 20:56, 9h43m, all 20 epochs. The batch_size=3
  fix held — cleared the epoch-3 backbone-unfreeze boundary with no OOM. Logs to
  TensorBoard (runs/Jul03_11-13-14_c0606a-s22.ufhpc), NOT Comet. Checkpoints in
  runs/taad_03072026_1113/checkpoints/ (best_model.pt = epoch 18, lowest val loss
  0.0259). Final val (epoch 20, thresholded): pass P0.66/R0.72, throw-in 0.58/0.93,
  drive 0.46/0.76, header 0.40/0.88, cross 0.36/0.95, shot 0.31/0.78, block 0.15/0.71,
  tackle 0.05/0.75. Precision is the weak axis (rare classes tackle/block worst);
  recall is healthy across the board.
Next: Test best_model.pt on our Veo footage (match-saints). BLOCKED on an adapter:
  run_TAAD_on_matches.py groups ROIs by ROLE_ID 1-13, but our tracklet extractor
  writes ROLE_ID=0 (no role model) -> empty ROIs. Needs a slot/pseudo-role assignment
  + pred-overlay in footpass_visualize.py (the --source pred path is unimplemented),
  and rfdetr installed in the footpass env for extraction.

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
Reviewed: Pipeline is correct but this clip has little dead time to cut. No halftime is
  present — the longest offscreen block is 4.5s (Veo likely pre-trimmed the break), so the
  "large halftime" premise doesn't hold for this file. Ball speed is median 251 px/s and the
  longest continuous near-stationary stretch is only 15s: the ball is almost always moving,
  so few spans meet the offscreen/stationary >5s rule. The 11 short cuts (4–14s) are genuine
  set-piece setups. Detection is NOT the bottleneck (92.5% visible); the dead-time definition
  is. To trim more, extend "dead" beyond ball-only (e.g. low ball speed <80 px/s ≈ 23% of
  match, or player-cluster/idle cues) rather than loosening stationary-px.
Next: Decide whether to add a speed-based / player-based dead-time criterion, and test on a
  match that actually contains an untrimmed halftime.

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

### Job 37631798 — RF-DETR threshold test (conf=0.15)
**Date:** 2026-07-20  
**Purpose:** Validate hypothesis that RF-DETR under-detection is due to conf_threshold=0.3 being too high for overhead footage. Test if lowering to 0.15 recovers all 15 visible players in frame 17376.  
**Command:** `sbatch slurm/test_threshold_0.15.sh`  
**Status:** Running  
**ETA:** ~20 min  
**Next:** Compare diagnostics frame between original (conf=0.3) and new run (conf=0.15)


### Job 37635195 — RF-DETR threshold=0.15 via config
**Date:** 2026-07-20  
**Purpose:** Test conf_threshold=0.15 on full pipeline using config file. Frame-level test showed +6 players (19→25). Expecting team classification to jump from 8/199 to near-complete.  
**Command:** `sbatch slurm/test_threshold_config.sh`  
**Config:** `examples/saints-u11-0.15-threshold.yaml` (detector.conf_threshold: 0.15)  
**Status:** Running  
**ETA:** ~25 min  


### Job 37659579 — SAM Player Detector (Full Pipeline)
**Date:** 2026-07-20  
**Purpose:** Test SAM (Segment Anything Model) for player detection. SAM uses clean segmentation masks → expect better team classification than RF-DETR.  
**Config:** `examples/saints-u11-sam.yaml` (detector.type: sam)  
**Approach:** SAM for players + RF-DETR for ball  
**Status:** Running  
**ETA:** ~45 min (model download + slower inference)  
**Expected:** Team classification 8/199 → 100+/199 if SAM works  

### Job 37721589 — SAM Player Detector (Full Pipeline, 4h limit)
**Date:** 2026-07-21 15:41
**Purpose:** Full SAM validation run; success = stats.json with >50% team classification.
**Config:** `examples/saints-u11-sam.yaml` (detector.type: sam)
**Result:** FAILED — TIMEOUT at 4h, reached frame 28500/55354 (~51%), no stats.json.
  SamAutomaticMaskGenerator @ points_per_side=32 ("segment everything") costs
  ~3.0s per SAM call; full match ≈9226 calls ≈7.7h. Not a hardware limit — the
  32x32 grid segments the whole frame (turf/lines/spectators) then discards ~98%
  of masks to keep ~20 players.
**Next:** Don't just raise wall time. Benchmark grid density first (job 37822023)
  to pick a coarse-grid config that keeps all players; then batched-encoder path
  if needed. Coarsening 32->16 + points_per_batch 64->256 alone should hit ~2h.

### Job 37822023 — SAM speed benchmark (grid density sweep)
**Date:** 2026-07-22
**Purpose:** Measure s/frame + median player-count for points_per_side {32,16,12}
  on 18 real mid-match frames, to size the full run empirically (user: "don't
  think we should need that much GPU power"). Script: slurm/bench_sam_speed.py.
**Status:** Running (30-min job).
**Next:** Pick the coarsest grid that still finds all ~20 players; apply to
  sam2.py + resubmit full run at the extrapolated wall time. If even grid12 is
  too slow, move to turf-mask + connected-components proposals (SAM prompted, not
  automatic) or drop SAM for team-color sampling entirely.


### Job 37826241 — SAM3 text-prompt validation (the real pathway)
**Date:** 2026-07-22
**Purpose:** Cheap domain-shift gate for the user's actual proposal — prompt SAM3
  with "soccer player" and let its VIDEO model detect + track players natively,
  replacing the broken SAM-v1 hack (detection/sam2.py: no weights loaded -> 0
  masks, confirmed by bench 37822023) AND ByteTrack. Findings that reframed this:
  (a) `segment-anything 1.0` = SAM **v1**, cannot do text prompts; detection/sam2.py
  was misnamed and ran with random weights. (b) The intended seam tracking/sam3.py
  is an empty Phase-5 stub. (c) The real `facebook/sam3` (arch Sam3VideoModel,
  text-prompt concept seg + masklet tracking) is ALREADY cached + past the HF gate,
  and transformers 5.12.1 supports it. Script: slurm/validate_sam3.py (48
  consecutive frames from 15000, offline load, reports players/frame + track-ID
  stability + 3 annotated frames to runs/sam3_validation/).
**Status:** Running (30-min job, loads offline so no token needed).
**Next:** If max players/frame >=~12 and IDs are stable, implement tracking/sam3.py
  properly (Sam3VideoModel), wire process.py to it, retire detection/sam2.py +
  the v1 configs. If it under-detects, retry prompt "person" / lower threshold
  before concluding domain shift.

### Jobs 37864846 / 37872201 / 37883252 — SAM3 validation + ball head-to-head
**Date:** 2026-07-23
**Result:** SAM3 (facebook/sam3, Sam3VideoModel, text-prompt concept segmentation)
  works on this footage and replaces RF-DETR + ByteTrack for players AND the ball.
  - players "soccer player": 20-22/frame, 21/22 ids stable (RF-DETR: 5-6/frame)
  - referee "referee": exactly 1 stable object -> refs are cleanly subtractable
  - ball "soccer ball": 66% detected, median jump 24px, p95 208px
    vs RF-DETR 71% detected, median 117px, **p95 1260px** (mostly false positives
    in trees/sky). Lower detection rate is a WIN. "ball" alone is worse (47%).
  Prior SAM work was broken: detection/sam2.py used SAM **v1** with NO checkpoint
  (random weights -> 0 masks, 7.2h/match); tracking/sam3.py was an empty stub.
**Gotchas found:** (1) session masklet memory grows ~0.12GB/frame and is never
  released -> OOM at ~150 frames; fixed with chunked sessions + IoU id stitching.
  Object pruning does NOT help (growth is per-frame, not per-object).
  (2) object_ids retains non-visible objects. (3) BGR->RGB ::-1 view has negative
  stride; torch.from_numpy rejects it.

### Session fixes to the pipeline (2026-07-23)
- **field_filter**: Hough homography returns ok=True with a degenerate matrix on
  this footage and rejected 100% of players from the frame it was first computed
  (job 37877533: 20 -> 0). Added a sanity guard -> falls back to the hull.
  Track-frames 52 -> 4292.
- **teams**: jersey colour was sampled over a rectangular torso patch = mostly
  turf on overhead footage, collapsing both clusters to one colour ("blue, blue").
  Now samples inside the SAM3 mask -> "black, blue" (black = correct Saints kit).
- **process**: ball track was computed then discarded; now persisted to
  run_dir.ball_track in the events.deadball schema (trim-empty can reuse it).
- **associate**: events carry garbage field_x from the bad homography, so
  field-space matching failed the 5m threshold for every event (0/8). Now falls
  back to pixel space; stamp_event_positions anchors events on the ball.
**Resolved (2026-07-27):** the ~100% throw_in flood was detect_throw_ins gating on
  near_touchline(field_x, field_y) from the same degenerate homography — and
  unlike in_goal_zone/in_corner_zone that gate had no bounds check, so any
  out-of-field fy (measured: -889 on a 36 m pitch) read as "on the touchline".
  detect_throw_ins + near_touchline are now deleted from events/set_piece.py; the
  two remaining detectors are bounded and abstain on bad coordinates, so the rules
  engine emits ~0 events on this footage instead of 236 false ones.
**Still open:** the homography itself. It is KEPT (user decision) but returns
  ok=True while mapping frame centre to e.g. (14284, -14) m on a 55×36 pitch
  (12/13 sampled frames on the saints proxy). Real set-piece spotting needs either
  a working registration or the `learned` engine.

### Job (pending submit) — slurm/submit_sam3_saints_full.sh
Why: First FULL-MATCH SAM3 run (players + ball text prompts) + identify, to
  produce real #6 data for the proximity/on-ball reel (events/on_ball.py). Chunked
  sessions only validated on ~300-frame clips; 8h wall as a hedge. Config
  saints-u11-sam3.yaml, profile saints-u11.yaml, match_id saints-u11-sam3-full.
Next: if it holds at length, build #6 on-ball reel from tracks/ball_track/jerseys;
  watch (a) does the chunked session survive 9226 sampled frames, (b) #6 OCR yield.
