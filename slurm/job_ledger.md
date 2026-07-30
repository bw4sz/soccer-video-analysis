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

## 38127264 — 2026-07-27 10:27 — slurm/downscale_clips.sh runs/saints-u11-sam3-full
Why: Prepare the event-annotation set for labelling on a laptop. The run's 236
  event clips are 1920x1080/20s/~25MB each = 6.1GB, too much to rsync down and
  far more than Label Studio needs to scrub. Re-encodes to 720p/CRF30/no-audio
  into runs/saints-u11-sam3-full/clips_720p (originals untouched), 8-way xargs.
  Companion to the `enroll --dump-crops` pass for the re-id gallery, both feeding
  the annotation walkthrough in label_studio/README.md.
Result: COMPLETED in 5:03, no errors. 236/236 clips re-encoded 1920x1080 -> 1280x720,
  6.1GB -> 351MB (17x smaller), 20s duration preserved. 8-way xargs on one node;
  the 1-core login shell would have taken ~an hour.
Next: rsync clips_720p + label_studio_tasks.json + labeling_config.xml to the
  laptop under runs/<match_id>/clips/, then Label Studio. Expect to correct
  nearly every label — all 236 events came back `throw_in` off the degenerate
  homography, so the pre-fill is a candidate window, not a prior.

### Job 38133841 — FOOTPASS in-domain detector eval (SAM3 vs RF-DETR)
**Date:** 2026-07-27
**Why:** Every SAM3 number we have (20-22 players/frame, 21/22 stable ids) was
  measured on Veo footage with NO ground truth — a count, not an accuracy. This
  is the missing baseline: P/R against real per-player boxes on the broadcast
  footage both detectors were trained for. If SAM3 scores well here and badly on
  Veo, the gap is domain; if badly on both, it's the prompt/threshold.
**Data:** FOOTPASS val `game_24_H1` — 47,805 frames with >=1 visible player,
  median 16 players/frame. GT = ROI_* columns of val_tactical_data.h5, rescaled
  fullHD -> 640x352 by x/3, y/3.068181 (matching vendor/FOOTPASS TAAD_Dataset.py).
  Alignment verified by eye on a GT-only overlay: 19 boxes sit tightly on players.
**Design:** 10 windows x 40 consecutive frames (SAM3 needs continuity; session
  reset per window). Uniform-random window placement, NOT filtered to wide shots
  — replays/close-ups are part of a broadcast, and pre-selecting easy framings
  would inflate the score; the wide-shot subset (>=8 GT players) is reported
  separately. Detections kept with raw scores and swept post-hoc over
  score {0.2,0.3,0.5} x IoU {0.3,0.5}.
**Known-pessimistic precision:** GT counts only the ~22 players + keepers, not
  referees/coaches/crowd, which both detectors return. Hence also
  `precision_in_field` (detections whose centre falls in the padded GT envelope).
**Dropped before submit:** planned upscale arms (frame x3). `Sam3VideoProcessor`
  resizes every input to 1008x1008 regardless, so naive upsampling is a no-op and
  cubic interpolation adds no information. Settling resolution-vs-domain needs the
  fullHD videos (on HF, never downloaded) or SAHI-style tiling.
**Result:** COMPLETED (exit 0, 6m42s, L4). 386 frames scored, median 15 GT players.
  IoU 0.5 / score 0.3, all frames:
    sam3    P 0.760  R 0.925  F1 0.835   P(in-field) 0.799   19 pred/fr  0.91 s/fr
    rfdetr  P 0.838  R 0.977  F1 0.902   P(in-field) 0.846   18 pred/fr  0.04 s/fr
  At IoU 0.3 both are near-ceiling on recall: sam3 0.980, rfdetr 0.994.
  Wide-shot subset is within 0.003 of "all" for both — the broadcast's replays and
  close-ups are not what's costing either model anything.
**Headline:** RF-DETR BEATS SAM3 in-domain, and is 23x faster. That reverses the
  premise of the SAM3 migration. RF-DETR was retired on a Veo *count* (5-6
  players/frame vs SAM3's 20-22) with no ground truth behind it; here it detects
  18/frame at 0.977 recall. So RF-DETR is not a weak detector — it is a strong
  broadcast detector that collapses under domain shift, and SAM3's real advantage
  is robustness to that shift, not detection quality. Keep SAM3 for Veo; do not
  assume it dominates everywhere.
**Precision is a LOWER BOUND, by two mechanisms.** (1) GT excludes referees,
  coaches and touchline staff, all of which both models return — the ref alone is
  ~1 of ~4 FP/frame for sam3. (2) GT is incomplete: on the inspected overlays both
  detectors independently box real players that carry no GT box. Two independent
  models agreeing on a box the GT lacks is evidence of a GT miss, not two
  coincident false positives. Recall is the metric to trust here.
**Side finding:** SAM3's metrics are IDENTICAL at score 0.2/0.3/0.5. `obj_id_to_score`
  holds each object's *birth* detection score, constant for the object's life
  (removed objects get -1e4), and essentially all survivors score >=0.5. So
  `SAM3PlayerTracker(min_score=...)` is a near-inert knob in the production
  pipeline — not a bug, but not the tuning lever it looks like.
**Next:** (a) The resolution question is still open and now matters more — a
  median GT player is 13x26 px here, so both scores are "good at 13px", not "good
  at broadcast". Fetch one fullHD val game or add SAHI tiling. (b) Re-check
  whether SAM3 is worth 23x the compute on Veo, or whether RF-DETR at a lower
  conf + the field filter closes enough of the gap. (c) No ball GT exists in
  FOOTPASS tactical data — ball detection remains unvalidated in any domain.

### Job 38162552 — TAAD on our footage, SAM3 tracklets (controlled re-run of 36500443)
**Date:** 2026-07-27
**Why:** The July smoke runs concluded "TAAD domain shift is the real blocker",
  but they fed TAAD tracklets built by RF-DETR + ByteTrack. Job 38133841 then
  showed RF-DETR gets 0.977 recall in-domain while finding only 5-6 players/frame
  on Veo (SAM3: 20-22). TAAD is track-aware — tracklets ARE its input and it
  scores the top-13 longest tracks per team — so it was reasoning over a pitch
  missing most of its players. "The action head doesn't transfer" and "the input
  was incomplete" are confounded in every result we have.
**Design:** Same video, same 1150-1170s window (frame 34466, 600 frames), same
  checkpoint (taad_03072026_1113 best_model.pt), same inference flags
  (--conf 0.15 --nms 15 --ball-gate soft). ONLY the front end changes:
  `--detector sam3` on footpass_extract_tracklets.py, running three SAM3 sessions
  ("soccer player" / "soccer ball" / "referee") over one shared copy of the
  weights. chunk_frames=30 because three concurrent sessions each grow masklet
  memory ~0.12GB/frame and the L4 has 23GB.
**Baseline to beat (36500443, RF-DETR tracklets), vs the in-domain prior from
  FOOTPASS val (6070 events):**
    class      in-domain   RF-DETR run   ratio
    pass          50.4%        11.1%      0.2x
    drive         40.7%         2.8%      0.1x
    block          1.3%        52.8%     41.1x
    shot           1.1%        22.2%     20.1x
  pass+drive 91.1% -> 13.9%;  block+shot 2.4% -> 75.0%. The two lowest-precision
  in-domain classes (block P0.15, shot P0.31) became three-quarters of output.
**Also changed:** jersey colour is now sampled inside the SAM3 mask rather than
  over the bbox (the fix that turned "blue, blue" into "black, blue" elsewhere);
  preview caption no longer hardcodes "RF-DETR+ByteTrack".
**Result:** COMPLETED (exit 0, 14m50s). 35 events:
  block 18, shot 7, pass 5, throw-in 3, drive 1, cross 1, header 0, tackle 0.
  block+shot 71% (was 75%); pass+drive 17% (was 14%). **The collapse is unchanged.**
  Tracklets DID change materially — 55 tracks vs 63, 8000 rows vs 12233, ball
  589/600 (98%) vs 559 (93%), teams 'gray'/'black' vs 'black'/'orange' — and TAAD
  produced the same inverted distribution anyway. That points at the action head,
  not the input.
**MY PREMISE WAS WRONG, and it weakens this experiment's rationale.** RF-DETR was
  NOT starving TAAD on this video: it produced 20.4 player-detections/frame here
  against SAM3's 13.3. The "5-6 vs 20-22 players/frame" figure comes from job
  37864846, measured on `SaintsU11_OVF_Jul192026.MP4` — the Veo overhead camera —
  whereas every TAAD smoke run uses `match-saints-16b-pre-mls-next-2026-04-26.mp4`,
  a different camera. I conflated two videos. The RF-DETR-starvation hypothesis was
  never true for this footage.
**Two uncontrolled differences I introduced:**
  (1) Field polygon came back "unreliable" and was skipped, where the July run got
      a 56%-of-frame polygon. Same code, same window — the difference is the python
      env (blue soccer-vision env for SAM3, cv2 4.13, vs the repo .venv in July).
      Known immaterial to the verdict: job 36460253 proved mask on/off gives
      byte-identical TAAD predictions on this exact window.
  (2) The "referee" concept vetoed 654 player boxes (~1.1/frame). Keyframe
      kf_02_f34895 shows the real referee correctly unboxed, but also two large
      foreground players missing — the veto is over-firing. With the field mask off,
      sideline spectators near the goal are also tracked (t38/t45/t39/t47/t48).
**Verdict:** Domain shift in the action head is now the best-supported explanation,
  but this run is one 20s window, ~35 events, with two confounds. Treat as
  corroboration, not proof.
**Next:** Stop engineering tracklets for this — the returns aren't there. The real
  route is FOOTPASS.md section B: annotate our matches with (frame, team, jersey,
  class) and fine-tune from best_model.pt. Note the official metric groups by
  (team, shirt, class) so it cannot be computed on our footage without jersey
  numbers — a class-and-time-only variant is needed. Also worth fixing regardless:
  the referee-veto over-firing and the field-polygon env sensitivity.

## 38162799 / 38162800 — 2026-07-27 18:10 — slurm/submit_sam3_process.sh (U14G)
Why: `enroll --dump-frames` (and everything else) needs a processed run per squad,
  and only the U11 match has one. Target is the U14G re-id gallery from
  data/wfc-rangers-vs-saints-pcu-cup-2026-07-11.mp4 (Veo, 60.6 min, 1920x1080).
  This footage differs from the U11 match in ways SAM3 has never been tested on:
  **Veo not XbotGo**, low sun with long shadows and lens flare, and the shared
  multi-pitch venue whose blue/red/white line clutter is the venue that killed
  field registration. So a 3-min smoke (38162799, data/u14g_smoke180.mp4, cut from
  15:00) runs first and the full match (38162800) is chained
  --dependency=afterok, so a failure costs 10 GPU-minutes instead of hours.
  Profile examples/profiles/saints-u14g.yaml declares kits black/white; its roster
  is empty pending the squad list, which the Label Studio label list needs.
  New script generalises submit_sam3_saints_full.sh, which hard-codes the U11 video.
Result: FAILED — smoke 38162799 hit the 1h wall I set (TIMEOUT at 01:00:21) and the
  chained full run was cancelled (DependencyNeverSatisfied). NOT a hang: the U11
  full match (job 38050708) took 6h21m for 9226 sampled frames = ~2.5 s/frame, so
  this 3-min clip's ~900 sampled frames needed ~40min and 1h left no margin for
  model load. My error was overriding the script's 8h default with --time=01:00:00.
  Two fixes: (a) PYTHONUNBUFFERED=1 in the script — SLURM redirects stdout to a
  file, so Python buffered it and the log showed only "[Step 1]" after an hour,
  which reads exactly like a hang; (b) don't process the full 60-min match for
  enrolment at all (see 38176317).
Next: on success, `enroll --dump-frames runs/saints-u14g-pcu-2026-07-11/label_frames`
  and pull for labelling. Watch (a) does SAM3's "soccer player" prompt hold up on
  Veo at this sun angle, (b) does the black/white kit split survive the shadows —
  this run should carry a proper `teams` block, unlike the U11 run.

## 38176317 — 2026-07-27 22:15 — slurm/submit_sam3_process.sh (U14G sampler)
Why: At ~2.5 s/sampled frame the full 60-min U14G match is a ~13h job that writes
  its JSON only at the end — a bad bet for something we only need a gallery from.
  A gallery wants diverse exemplars, not coverage, so this processes
  data/u14g_sampler600.mp4: six 100s chunks (t=300/850/1400/1950/2500/3050)
  stream-copied out of the match and concatenated = 10 min, 18143 frames, ~3024
  sampled. Spanning the match keeps the diversity that matters here, since the sun
  drops through the game and shadows change. Joins verified to decode clean.
  ~2.1h expected, 5h wall. match_id saints-u14g-sampler.
Result:
Next: `enroll --dump-frames runs/saints-u14g-sampler/label_frames --profile
  examples/profiles/saints-u14g.yaml` (roster now has all 13 players), pull ~1MB
  and label in Label Studio. Full-match processing stays deferred until clips are
  actually wanted from this game.

## 38176330 / 38176804 / 38177148 — 2026-07-27 22:10 — why is SAM3 ~26x RF-DETR?
Why: Job 38133841 measured the headline gap on FOOTPASS footage (SAM3 0.91 s/frame
  vs RF-DETR 0.04) but never said where the time goes, and job 38162799 — a
  **3-minute** smoke — hit the 1h wall without finishing. Needed the decomposition
  before deciding what (if anything) to optimise.
  Scripts: slurm/profile_detector_speed.py (stage breakdown + resolution sweep),
  slurm/profile_sam3_scaling.py (object-count and chunk-depth sweeps).
Result: COMPLETED. On an L4, 1920x1080, 120 frames of data/u14g_smoke180.mp4:

  RF-DETR (players+ball, one pass)  0.062 s/frame   flat in object count
  SAM3 "soccer player" session      1.354 s/frame
  SAM3 "soccer ball" session        0.269 s/frame
  SAM3 production total (both)      1.623 s/frame   = 26.1x RF-DETR

  Stage split of the player session: forward 93.0%, postproc 5.7%, preproc 1.0%,
  id-mapping 0.2%, session rotation 0.1%. **It is the model forward, full stop** —
  mask->box conversion and the chunked-session machinery are not the problem.

  Two scaling laws, both fit essentially perfectly:
    forward = 0.205 s + 38.7 ms x n_masklets          (R^2 = 1.000, 6 prompts)
    forward = 1.034 s + 10.7 ms x chunk_depth         (chunk_frames 10/20/30/60)
  So the fixed image+text encode is only ~0.21 s (already 3.4x RF-DETR's *whole*
  frame) and everything above that is per-tracked-object. At 27 masklets, 84% of
  the forward is per-object work. Confirmed in the transformers source: SAM3's
  tracker propagation batches objects along the **batch dimension** through memory
  attention + the mask decoder (models/sam3_video/modeling_sam3_video.py,
  run_tracker_propagation), so N objects really is ~N x the compute. RF-DETR
  decodes a fixed query set in one pass — 0.046 s whether it finds 5 or 27.

  **Resolution is not a lever.** 640x360 is only 1.2x faster than 1920x1080
  despite 9x fewer pixels (1.134 vs 1.354 s/frame) — SAM3 resizes internally to a
  fixed size. Downscaling the video buys almost nothing.

  **Prompt choice IS a lever, because it changes masklet count:**
    goalkeeper 1 -> 0.247s | referee 5 -> 0.384s | "person on a sports field"
    20.5 -> 0.994s | "soccer player" 27 -> 1.242s | "person" 34 -> 1.531s
  We pay 38.7 ms/frame for every spectator and sub SAM3 latches onto *before*
  filter_spectators ever discards them.

Separate, detector-independent finding: `VideoReader.sample_frames` seeks with
  `cap.set(CAP_PROP_POS_FRAMES, fn)` for every sampled frame
  (src/soccer_vision/io/video.py:41), which on H.264 re-decodes from the preceding
  keyframe. Measured 249 ms/frame vs 30 ms for sequential grab-and-skip — **8.3x**,
  and it hits the RF-DETR path just as hard.

Consequence: the U14G full match (108,972 frames, 18,162 detection-frames at 5 fps)
  projects to 8.19 h of SAM3 + 1.26 h of decoding = **9.4 h against the 8 h wall
  limit in submit_sam3_process.sh** — job 38162800 could never have finished even
  had the smoke passed. RF-DETR on the same match: 1.6 h, of which 1.26 h is
  decoding.
Next: cheapest wins first, in order — (1) sequential reader, ~1.1 h off every run
  regardless of detector; (2) chunk_frames 60 -> 10, ~31% off SAM3's forward, but
  re-check id stability since rotation is what fragments lanes; (3) a tighter
  player prompt to cut masklet count. Only after those, consider whether the ball
  needs its own session every frame — it re-encodes the same image for one object.

### Job 38177302 — SAM3 tracklets with per-track referee vote
**Date:** 2026-07-27
**Why:** The frame-level referee veto in 38162552 dropped 654 player boxes (~1.1/frame)
  and visibly cost two foreground players. Replaced with a per-track vote: accumulate
  the fraction of a track's frames that overlap a referee box, drop the whole track
  only above --ref-vote (0.6). A track is either the official or it isn't.
**Result:** COMPLETED (exit 0, 14m46s). Fix works — keyframe kf_02_f34895 shows the
  #7 foreground player recovered as t24, with the actual referee still correctly
  unboxed. Vote dropped 2/58 tracks (563 detections) vs the veto's 654; rows 8000 ->
  8128; continuity drops 46 -> 9.
**But the over-firing diagnosis was WRONG.** The new diagnostic shows SAM3's "referee"
  concept returns 1.13 objects/frame (min 0, median 1, max 3) — well-behaved, not
  over-matching. The problem was never a greedy concept; it was that a *correct*
  concept's occasional frame-level overlap deleted real players outright. The vote
  fixes exactly that failure mode, but the magnitude was smaller than the raw 654
  suggested: only 91 detections were recovered, the other 563 being two genuine
  referee/official tracks.
**Effect on the verdict: none.** TAAD output is byte-identical to 38162552 — 35 events,
  block 18, shot 7, pass 5, throw-in 3, drive 1, cross 1, header 0, tackle 0. The
  recovered player never entered TAAD's top-13-longest-tracks-per-team selection. So
  the class collapse is not a referee-filter artifact, and this is now the third
  distinct tracklet configuration (RF-DETR, SAM3, SAM3+vote) to produce the same
  inverted distribution.
**Behaviour change to know about:** the referee veto now runs AFTER tracking for both
  detectors so they share one code path. `--ref-filter frame` is therefore no longer
  bit-identical to the July runs (ByteTrack now sees referee boxes before they are
  filtered). The controlled comparison it existed for is finished.
**Still open (unrelated, and now looks more urgent):** the turf polygon fails in the
  blue env ("unreliable") but succeeds in the repo .venv (75% of frame) on the same
  window — an OpenCV-version sensitivity, not a data problem. With the mask off,
  sideline spectators are tracked, and being stationary they form LONG STABLE tracks —
  exactly what TAAD's top-13-longest selection favours. The July runs had the mask on.
  This is an uncontrolled variable in both SAM3 runs and worth fixing before any
  further TAAD-on-our-footage work.

## 38178685 — 2026-07-27 23:33 — slurm/submit_process.sh (RF-DETR is the default again)
Why: User's call, on two grounds the profiling above supports. (1) `facebook/sam3`
  is HF-gated, which breaks a fresh clone or a new collaborator in a way caching
  can't fix — against a "quick and dirty and easy" ethos. (2) SAM3 costs 26x per
  frame and a full match (9.4h) silently overran the 8h wall. The quality argument
  for SAM3 had already collapsed: job 38133841 has RF-DETR *ahead* in-domain
  (F1 0.902 vs 0.835, recall 0.977 vs 0.925), and the Veo count that motivated the
  migration (5-6 players/frame) never reproduced — 38162552 measured RF-DETR at
  20.4 detections/frame vs SAM3's 13.3, and the speed profiling saw 22-27/frame.
Changed: examples/process_match.yaml now carries an explicit `detector: type:
  rfdetr`; submit_sam3_process.sh renamed to submit_process.sh, taking config as
  a 4th arg (default RF-DETR) and dropping the wall 8h -> 4h; process.py comments
  reordered so RF-DETR reads as the default and SAM3 as opt-in; CLAUDE.md gained a
  "Detector" section; label_studio/README.md no longer points at the SAM3 config.
  SAM3 is untouched and still selectable via examples/saints-u11-sam3.yaml.
Result: COMPLETED (exit 0, **2m45s** on data/u14g_smoke180.mp4 — the same 3-min
  clip whose SAM3 run, 38162799, hit a 1h wall without finishing). 856 tracks,
  662 kit-stamped (77%), teams correctly named black/white from the profile,
  ball 760/899 visible (84.5%). Team colour sampling falls back from mask to bbox
  cleanly when the detector supplies no mask.
**Two real quality costs, measured, both worth fixing rather than reverting for:**
  (1) Ball jitter: median jump 54px, **p95 905px** on a 1920px frame (SAM3: p95
      208px). ball_track.json is written raw and on-ball spans inherit it.
      tracking/ball_kalman.py already exists for this and is not wired into
      `process` — but it over-rejects at the 5fps `process` samples at (~39%),
      so wiring it in probably wants the ball sampled denser at the same time.
  (2) Fragmentation: 856 lanes in 3 min, median lane 7 detection-frames, only 32
      lanes >=50 frames, vs SAM3's 55 lanes over 600 frames (38162552). enroll's
      `--dump-crops --max-tracks 60` now draws from a much shorter-lived pool.
Next: (a) decide on ball_kalman-in-process + ball sample rate; (b) re-check
  enroll crop yield per player on an RF-DETR run before trusting a new gallery;
  (c) the seek-per-frame reader (see 38176330) is now the single biggest win
  left — 1.26h of RF-DETR's projected 1.57h full-match runtime is decoding.

## 38180242 — 2026-07-28 00:16 — slurm/diag_rfdetr_u14g.py (what the preview revealed)
Why: Rendered runs/u14g-smoke-rfdetr back onto its video with the new
  scripts/preview_run.py, and it looked bad — 6-13 boxes where the eye counts
  12-20 players, the ball circle sitting on background objects, and nearly every
  box stamped "black". Needed to know whether the RF-DETR default is the cause or
  whether the pipeline around it is losing them.
Result: COMPLETED (35s). 40 frames, every 6th from 300. **The detector is not the
  problem.** Per-frame medians:
    RF-DETR raw (4 classes)      23      <- finds the players fine
    person classes (1,2,3)       22
    player+GK only (1,3)         21.5
    after filter_spectators      14      <- drops 8 of 22 (36%)
    after ByteTrack              11.5
  This also independently buries the claim that motivated the SAM3 migration:
  RF-DETR returns 22 people/frame on Veo footage, not 5-6.
**Three real problems, none of them detection quality:**
  (1) `detection/field_filter.py::filter_spectators` drops 36% of detected people.
      It is the crude central-rectangle hull CLAUDE.md already flags, and in the
      preview it keeps actual spectators along the far touchline while cutting
      real players near frame edges. This is the turf-mask work (issues #7/#20).
  (2) Team assignment has collapsed: 624 'black' vs 38 'white'. Jersey colour
      sampled from the bbox (no mask) averages kit with turf and shadow — mean
      BGR [68,70,70], std ~20, i.e. neutral grey for *both* kits, so the two
      clusters are not separable. This is the regression predicted when SAM3's
      per-player mask went away; `sample_jersey_bgr(frame, bbox, mask=None)` needs
      a mask-free way to isolate torso pixels (centre-crop + turf-hue rejection).
  (3) Ball latches onto background objects — visually confirmed at t=11s and
      t=16s, consistent with the p95 905px jump measured in 38178685.
Next: (2) is the cheapest and most valuable — team colour is what `--team` and
  every kit-aware clip query depend on, and it is currently wrong. Then (1).
  Neither argues for reverting to SAM3: it was masking these bugs, not fixing them.

## 38207124 / 38207619 — 2026-07-28 10:2x — team colour without a SAM3 mask
Why: 38180242 found team assignment had collapsed to 624 'black' / 38 'white' on
  runs/u14g-smoke-rfdetr, because dropping SAM3 dropped the per-player mask that
  `sample_jersey_bgr` used to isolate kit pixels. `--team` and every kit-aware
  clip query were wrong.
**The obvious diagnosis was wrong.** Turf contamination is real but minor. The
  actual cause is **shadow**: local turf L* ranges 50-101 under this low sun, so a
  white kit in shade is *darker* than a black kit in sun. Measured on 188 tracks —
  absolute torso L* put 174/188 in one cluster; the contact sheet
  (crops sorted by torso-minus-turf lightness) showed the kits separating cleanly
  at zero.
**And clustering cannot find that boundary.** The relative-lightness histogram is
  unimodal with a long sunlit-white tail; k-means cut at +53.2 and Otsu at +53.2,
  both isolating 12 bright shirts. Zero is the boundary for a physical reason (a
  dark kit reflects less than the grass beside it, a light kit more), not a
  statistical one — so the split is thresholded, not clustered.
Implemented in tracking/teams.py: `turf_pixels` (hue-gated grass, no value gate —
  a black kit is legitimately dark), `estimate_local_illuminant` (turf ring around
  each box), `correct_illumination` (per-channel von Kries, which also removes the
  warm-sun/blue-shade colour cast), `lightness_split_kits` (only when the declared
  kits straddle the turf — black/white and blue/white yes, red/blue no), and a
  nearest-centroid placement for tracks that never see grass. `process` now prints
  `Team split by:`. 6 new tests, incl. white-in-shade vs black-in-sun, which the
  old clustering path cannot separate at all (it names both "gray").
Result: **419 black / 243 white** (was 624/38), on 662 stamped tracks; visually
  confirmed in runs/u14g-smoke-teamfix/preview_10s.mp4 — light-kit players that
  were uniformly orange are now correctly cyan, and the large foreground dark-kit
  player stays black. Illuminant coverage is 661/662 tracks (99.8%), 99.9% of
  samples. 216 tests pass.
**Unrelated bug found:** re-running `process` with an existing --match-id crashes
  at Step 8 with `sqlite3.IntegrityError: UNIQUE constraint failed: matches.id`
  (job 38207619), *after* every artefact is written. Costs a full re-run's compute
  to discover. store/db.py should upsert rather than insert.
Next: `filter_spectators` still drops 36% of detected people and remains the
  largest quality gap; that is the turf-mask work (issues #7/#20).

## 38223174 / 38223284 — 2026-07-28 — is the SAM3-vs-RF-DETR gap video-dependent?
Why: We had four speed ratios from three scripts (38133841, 38176330, 38186278)
  that disagreed — 13x, 23x, 26x, 29x — and that spread was being read as "it
  depends on the footage". Two known methodology defects were in there: 38186278
  timed 6 frames with **no warm-up** (first-frame cuDNN autotune lands in the
  mean) and counted only SAM3's *player* session, while production runs a player
  and a ball session per frame. New script slurm/bench_detector_speed_multivideo.py
  sweeps every distinct camera we own plus 4 harvested youth clips (unfamiliar
  cameras), identical protocol for both detectors.
**First run (38223174) had a flaw I introduced and it is instructive.** Sampling
  frames *evenly* across a video is right for a stateless detector and wrong for
  SAM3, which is a stateful video tracker — frames a minute apart are scene cuts,
  masklets churn, and it re-detects from scratch. SAM3 measured 2.5-3.1 s/frame
  on the long matches but **1.26-1.48 s/frame on the 10-second youth clips**,
  where even spacing still lands frames <1 s apart. That internal contrast is the
  tell. RF-DETR scored 0.038-0.048 either way, which is what makes the artefact
  attributable to statefulness rather than to the videos.
Result (38223284, COMPLETED 7m54s, corrected to 3 bursts x 15 **consecutive**
  frames at the 5 fps process stride, session reset per burst, 5 warm-up frames
  discarded per burst per detector):

  video                    rfdetr s/fr   sam3 s/fr   ratio   rfdetr obj  sam3 obj
  u14g-veo-multipitch         0.046        1.540     33.7x      29.7      22.9
  u11-xbotgo                  0.045        1.652     36.5x      19.8      24.3
  saints-16b-sideline         0.045        1.702     38.8x      19.4      19.8
  youth-wA5HoPCvIps           0.037        1.292     34.9x      18.7      17.8
  youth-Q0xJibkjQas           0.037        1.075     28.7x      15.7      13.0
  youth-pW8Safa0khM           0.038        1.381     36.7x      25.6      23.3
  youth-zGPP8c0sPb8           0.038        1.195     31.8x      15.6      15.5

**The gap is a property of the models, not the footage.** RF-DETR is 0.045-0.046
  s/frame on every 1080p video and 0.037-0.038 on every 720p one — flat in object
  count (15.6 to 29.7 objects) exactly as 38176330's scaling study predicted. SAM3
  is 28.7-38.8x slower everywhere; the residual spread tracks resolution and
  object count, not venue. 1.540 s/frame on u14g reproduces 38176330's 1.623 on
  consecutive frames, which validates the burst protocol.
**The earlier "mixed results" were two artefacts, now both explained.** The 13x
  from 38186278 was warm-up inflating RF-DETR to 0.10 (true value 0.046) *and*
  omitting SAM3's ball session. There was never a video where SAM3 was
  competitive on speed.
**Object counts genuinely do flip, and that is the real cross-video finding** —
  RF-DETR finds more on u14g (29.7 vs 22.9), SAM3 more on u11 (24.3 vs 19.8).
  Neither is ground-truthed, so this is a count not an accuracy; the only
  ground-truthed comparison we have is still 38133841 (FOOTPASS, RF-DETR ahead:
  F1 0.902 vs 0.835, recall 0.977 vs 0.925).
**Projected 60-min match, detector only:** RF-DETR 0.19-0.23 h vs SAM3 5.4-8.5 h.
  SAM3 exceeds the 4 h wall in submit_process.sh on every single video.
Side note: the rfdetr err log advertises `model.optimize_for_inference(dtype=
  float16)` for "~8x on T4 via FP16 Tensor Cores" — every RF-DETR number here is
  *unoptimized*, so the speed lever is untouched on the default path.
Next: nothing argues for revisiting the default. If SAM3 is wanted for a domain-
  shift check, budget ~8 h/match and raise the wall.

### Job 38223804 — TAAD on U14G footage (first run on this camera)
**Date:** 2026-07-28
**Script:** `slurm/submit_footpass_ours_u14g.sh`
**Why:** Every TAAD-on-our-footage smoke so far (36450376, 36460253, 36500443,
  38162552) used the *same* video — `match-saints-16b-pre-mls-next-2026-04-26.mp4`,
  the XbotGo camera. The block+shot class collapse has therefore been measured
  four times on one camera and zero times on U14G, which is a different camera
  (Veo), venue (shared multi-pitch) and sun angle (low, long shadows, flare).
  Before committing to annotation + fine-tuning (FOOTPASS.md section B), find
  out whether we are fighting one domain gap or two.
**Design:** Protocol identical to 36500443 so the runs are comparable —
  600 frames, stride 1, RF-DETR+ByteTrack, `--conf 0.3 --ball-conf 0.2
  --ref-filter vote`, TAAD at `--conf 0.15 --nms 15 --ball-gate soft`, same
  `taad_03072026_1113/best_model.pt`. Only the video and window change.
  Window = frames 2700-3300 of `data/u14g_smoke180.mp4`, picked off
  `runs/u14g-smoke-rfdetr/ball_track.json` as live open play (92-95% ball
  visibility, 56-120 px median ball motion). 36450376 was invalidated by landing
  in a dead-ball span; this avoids that.
  RF-DETR not SAM3 because 38162552 showed the front end does not move TAAD's
  class distribution, and RF-DETR is 33x faster on this exact footage (38180242).
**Baseline to compare against (in-domain FOOTPASS val prior, 6070 events):**
  pass 50.4%, drive 40.7%, header 2.7%, cross 1.8%, throw-in 1.6%, block 1.3%,
  shot 1.1%, tackle 0.4%. XbotGo runs inverted it to block+shot 71-75%,
  pass+drive 14-17%.
**Known caveat:** `--field-mask auto` takes the largest green blob, and this is a
  multi-pitch complex, so the neighbouring match is turf too and is NOT excluded.
  The `--pitch-region` flag that would have fixed it was removed in 24557ed.
  Check the tracking preview before reading anything into the class counts.
**Result:** COMPLETED (exit 0, ~4 min). **The block+shot collapse does NOT
  reproduce on U14G.** 14 events: drive 5, throw-in 3, shot 2, pass 1, cross 1,
  header 1, tackle 1, **block 0**.
    class         in-domain   XbotGo(SAM3)   U14G(this)
    block+shot        2.4%        71%           14%   (2/14)
    pass+drive       91.1%        17%           43%   (6/14)
  block+shot 25/35 -> 2/14 is Fisher p=0.0004. pass+drive 6/35 -> 6/14 is
  p=0.076 (suggestive, underpowered). So the *specific* pathology the four
  XbotGo runs measured is camera-specific, not a universal TAAD failure — but
  U14G is still far off the in-domain prior (pass+drive 43% vs 91%, throw-in
  21% vs 1.6%).
**Read it narrowly — three real weaknesses:**
  (1) n=14 events. Small, and the event *rate* is a third of XbotGo's (14 vs
      35-36 on the same 600-frame budget), so TAAD is also much less confident
      here.
  (2) **Only 4/14 events are near-ball** (ball_dist <= 0.135); the other 10 run
      out to ball_dist 0.78 and are almost certainly false positives whatever
      their class. The 4 near-ball ones are drive / shot / tackle / pass — a
      plausible mix, which is the encouraging part of this run.
  (3) The multi-pitch confound fired exactly as predicted. Turf polygon = 69% of
      frame and **dropped 0 off-field detections**; the mask did nothing.
      Keyframe kf_05_f3236 shows `t97` is an adult coach in a light-blue polo on
      the sideline and `t101` is the bench row — both tracked, and t97 produced
      the f2884 throw-in (ball_dist 0.779). 3 of the 14 longest tracks are
      near-camera sideline figures (t1/t97/t101, box heights 142/177/213 px
      against a 57 px median), which is exactly what TAAD's top-13-longest
      selection favours.
**Also:** team split named `orange`/`gray` where the profile declares
  black/white. The red/blue box assignment looks broadly right in the keyframes
  (red=light kit, blue=dark), so the *split* works and only the colour *names*
  are wrong — `footpass_extract_tracklets.py` calls `TeamClassifier` directly
  and so misses the turf-relative lightness fix that `process` got in 59c396a.
**Artifacts:** `/blue/.../footpass/ours/taad_smoke_u14g/` (predictions.json,
  annotated.mp4, 8 keyframes), tracklets
  `/blue/.../footpass/ours/our_u14g_rfdetr_u14g.h5` (+ manifest),
  preview `/blue/.../footpass/ours/tracking_preview_u14g.mp4`.
**Next:** Do NOT size the annotation budget off the XbotGo runs — they measured
  a failure mode this camera does not have. Two cheap things before annotating:
  (a) re-run this window with the sideline/bench figures excluded, to see how
  much of the off-ball noise is non-participants (needs a spatial gate; the
  `--pitch-region` flag was removed in 24557ed, so this needs a decision, not
  just a flag); (b) the zero-annotation calibration check — `footpass_infer_ours.py`
  does softmax+argmax at conf 0.15, so reweight by the FOOTPASS val class prior
  and see whether pass/drive recover on the XbotGo window. Only then scope
  FOOTPASS.md section B.

## 38225019 — 2026-07-28 — slurm/submit_process.sh (smoke after deleting SAM3)
Why: Removing SAM3 touched the hot path in cli/process.py — the detector setup
  and the per-frame detect/track branch both lost their `if sam3_tracker` arm.
  Tests pass but none of them exercise that loop, so this re-runs the same 3-min
  U14G clip job 38178685 used, to prove the RF-DETR path is behaviourally
  unchanged rather than merely importable.
Result: COMPLETED (exit 0, 5m10s). **Byte-identical counts to 38178685**:
  856 tracks, 662 kit-stamped, ball 760/899 visible (84.5%), teams black/white,
  0 events (expected — no action engine). Run runs/u14g-smoke-postsam3.
Next: none. SAM3 removal is behaviour-preserving on this footage.

## 38313387 — 2026-07-29 15:2x — slurm/submit_process.sh (U14G full match)
Why: The re-id gallery is stuck at 42% naming teammates, and the first tracklet
  batch (260 crops, `runs/u14g_tracklets`) did not move it — 19/45 to 17/45 on a
  fixed held-out set. Cause is sampling, not the labelling workflow: every one of
  those crops came from `u14g_smoke180.mp4`, a 3-minute cut from 15:00, so one sun
  angle and only 7 of 11 players. `enroll --dump-tracklets` cannot spread windows
  across a match it has no tracks for, so the smoke clip is the binding
  constraint. This processes the full 60-min match
  (data/wfc-rangers-vs-saints-pcu-cup-2026-07-11.mp4, match_id saints-u14g-full)
  so windows can span the whole game — including the late backlit stretch where
  accuracy collapses to 3/13 — and every player appears in several.
  ~1.6h expected (mostly video decoding), 4h wall, RF-DETR.
Result: **FAILED — OOM killed (exit 137)** at frame 19500/108972, 8.5 min in, 64GB
  exhausted. Cause: TeamClassifier kept 4 image crops per track id and RF-DETR
  fragments this venue into ~285 lanes/min, so a 60-min match asks for ~68,000
  crops — for a preview PNG that renders 8 per team. Fixed by a global
  `max_crops=600` cap; wall memory also raised 64->128GB since per-frame track
  boxes are held for the whole run. Resubmitted as 38348887.
Next: `enroll --run runs/saints-u14g-full --dump-tracklets runs/saints-u14g-full/tracklets
  --profile examples/profiles/saints-u14g.yaml --team black --n-windows 12`, label,
  then A/B the new gallery with `slurm/ab_gallery_sources.py` before adopting it —
  the last batch's 260-crop yield looked like progress and wasn't.

## 38348887 — 2026-07-29 21:3x — slurm/submit_process.sh (U14G full match, retry)
Why: Retry of 38313387 after its OOM. Two changes: TeamClassifier crops capped
  globally at 600 (was 4 per track, unbounded in match length) and --mem 64->128GB.
  Both at once, so success won't attribute the fix — the goal is a full-match run
  to hang tracklet windows off, since the 3-min smoke capped the first labelling
  batch at one sun angle and 7 of 11 players (260 crops, no measurable gain).
Result:
Next: 12 tracklet windows across the full match, label, then A/B with
  slurm/ab_gallery_sources.py before adopting the gallery.
