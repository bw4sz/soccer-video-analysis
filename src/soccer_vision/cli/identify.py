"""CLI: soccer-vision identify — name each track's player.

Two ways to put a name on a ByteTrack lane, and this drives both:

- **re-id** (``--method reid``) — embed the track's crops and look them up in the
  team's appearance gallery from `enroll`. No jersey needs to be readable, so it
  works on the backs and blurs OCR gives up on. Preferred for a team you see
  every week.
- **OCR** (``--method ocr``) — read the digits and vote per track. Needs no prior
  enrolment, so it's the cold-start path and the fallback.

``--method reid+ocr`` (what ``auto`` picks when a gallery exists) runs re-id
first and sends the tracks it abstained on to OCR — the gallery can't name a
player it never enrolled, and OCR occasionally can. It also sends the tracks
re-id *did* name, not to rename them but to **cross-check** them: several
high-confidence reads agreeing on a number that isn't the named player's is
proof the re-id match is wrong, and that track is dropped back to unknown
(:mod:`soccer_vision.identify.crosscheck`; ``--no-ocr-verify`` skips this pass
and the OCR it costs).

``--team <kit>`` restricts naming to one squad's kit colour, and on a match with
two teams on screen you almost always want it: a gallery holds one squad and
cannot answer "none of the above", so an opponent or a referee gets named after
whichever of our players is nearest. See :mod:`soccer_vision.identify.team_gate`.

Runs as its own opt-in step (not part of `process`) over an already-processed
run, and writes ``jerseys.json``: per track, the matched name and/or voted
jersey, plus the ``source`` that named it so a clip's selection can be audited.
Downstream, `extract`/`reel --player NAME` / `--number N` resolve to the matching
lanes.
"""

from __future__ import annotations

import json
from pathlib import Path


def run_identify(args):
    from soccer_vision.clips.halo import load_track_boxes
    from soccer_vision.profiles.loader import get_reid, load_profile

    run_dir = Path(args.run)
    tracks_path = run_dir / "tracks.json"
    proxy_path = run_dir / "broadcast_proxy.mp4"
    jerseys_path = run_dir / "jerseys.json"

    if not tracks_path.exists():
        print(f"No tracks.json in {run_dir} — run `soccer-vision process` first.")
        return
    if not proxy_path.exists():
        print(f"No broadcast_proxy.mp4 in {run_dir} — run `soccer-vision process` first.")
        return

    profile = load_profile(args.profile) if args.profile else None
    reid_cfg = get_reid(profile) if profile else {}

    gallery_path = _resolve_gallery(args.gallery, reid_cfg, run_dir)
    method = args.method
    if method == "auto":
        method = "reid+ocr" if gallery_path else "ocr"
    if method.startswith("reid") and not gallery_path:
        print("No gallery found — run `soccer-vision enroll` first, or use --method ocr.")
        return

    print("=== soccer-vision identify ===")
    print(f"Run:    {run_dir}")
    print(f"Method: {method}")
    track_boxes = load_track_boxes(tracks_path)
    print(f"Tracks: {len(track_boxes)}")

    results: dict[int, dict] = {tid: _blank() for tid in track_boxes}

    # The kit gate runs before any model does: a lane in the opponent's colours
    # can't be one of ours, so naming it is wrong and embedding it is wasted.
    kit = args.team or reid_cfg.get("team")
    eligible = _apply_team_gate(track_boxes, tracks_path, kit, args.team_strict, results)
    # …and so does the length gate: a lane too short to carry a confident vote
    # can never be worth a clip, and naming it seeds a whole linked chain.
    min_lane_s = _first_set(args.min_lane_seconds, reid_cfg.get("min_lane_seconds"), 1.0)
    eligible &= _apply_length_gate(track_boxes, tracks_path, min_lane_s, results)
    namable = {t: b for t, b in track_boxes.items() if t in eligible}

    if method.startswith("reid"):
        _run_reid(args, namable, proxy_path, gallery_path, reid_cfg, profile, results,
                  tracks_path=tracks_path, run_dir=run_dir)

    verify = method == "reid+ocr" and not args.no_ocr_verify

    if method.endswith("ocr"):
        # OCR sees the tracks the gallery couldn't name, plus — when verifying —
        # the ones it did, to check their numbers against the reads.
        todo = {t: b for t, b in namable.items()
                if method == "ocr" or verify or results[t]["name"] is None}
        _run_ocr(args, todo, proxy_path, profile, results,
                 verify=verify, reid_cfg=reid_cfg)

    doc = {
        "video": proxy_path.name,
        "method": method,
        "gallery": str(gallery_path) if gallery_path else None,
        "model": args.model or "parseq",
        "ocr_verify": verify,
        "team": kit,
        "team_strict": bool(kit and args.team_strict),
        "min_lane_seconds": min_lane_s,
        "tracks": {str(t): r for t, r in results.items()},
    }
    jerseys_path.write_text(json.dumps(doc, indent=2))

    named = sum(1 for r in results.values() if r["name"] or r["jersey"] is not None)
    flagged = sum(1 for r in results.values() if r["conflict"])
    by_source: dict[str, int] = {}
    for r in results.values():
        if r["source"]:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(by_source.items())) or "none"
    print(f"\nIdentified {named}/{len(results)} tracks ({breakdown}). Saved: {jerseys_path}")
    if flagged:
        verb = "dropped" if getattr(args, "drop_on_conflict", False) else "flagged"
        print(f"{flagged} re-id match(es) {verb} on a jersey conflict — see the "
              f"`conflict` records in jerseys.json")
    print(f"Next: soccer-vision reel --run {run_dir} --player <name>   (or --number <N>)")


def _blank() -> dict:
    return {"jersey": None, "name": None, "source": None, "confidence": 0.0,
            "n_obs": 0, "legible_frac": 0.0, "similarity": None, "margin": None,
            "open_margin": None, "crosscheck": None, "conflict": None,
            "kit": None, "span_s": None, "excluded": None}


def _apply_team_gate(track_boxes, tracks_path: Path, kit, strict: bool,
                     results: dict[int, dict]) -> set[int]:
    """Mark every lane with its kit and hold back the ones wearing another team's.

    Excluded lanes stay in ``jerseys.json`` carrying ``excluded: "kit"`` rather
    than being dropped, so a lane that went unnamed can always be explained.
    """
    from soccer_vision.identify.team_gate import EXCLUDED_KIT, gate_by_kit

    stamped = json.loads(tracks_path.read_text()).get("teams") or {}
    track_kits = {int(k): v for k, v in stamped.items()}
    for tid in track_boxes:
        results[tid]["kit"] = track_kits.get(tid)

    if not kit:
        if track_kits:
            print("Team gate: off — pass --team <kit> to stop the gallery naming "
                  "opponents and referees after your own players")
        return set(track_boxes)

    if not stamped:
        print(f"Team gate: --team {kit} was asked for but tracks.json has no `teams` "
              f"block — nothing to gate on, naming every lane. Re-run `process`, or "
              f"`enroll --team {kit}` to classify kits first.")
        return set(track_boxes)

    eligible, counts = gate_by_kit(track_boxes, track_kits, kit, strict=strict)
    for tid in track_boxes:
        if tid not in eligible:
            results[tid]["excluded"] = EXCLUDED_KIT

    held = "held back too" if strict else "still eligible"
    print(f"Team gate: naming the {kit} kit only — {counts['ours']} lanes ours, "
          f"{counts['other']} on another kit (excluded), "
          f"{counts['unassigned']} with no kit assigned ({held})")
    return eligible


def _apply_length_gate(track_boxes, tracks_path: Path, min_seconds: float,
                       results: dict[int, dict]) -> set[int]:
    """Hold back lanes too short to be worth naming.

    Like the kit gate this runs before any model does, so the excluded lanes cost
    no re-id forward passes and no OCR — it makes the step faster, not slower.
    Excluded lanes stay in ``jerseys.json`` carrying ``excluded: "short"`` (unless
    the kit already explained them) and their measured ``span_s``.
    """
    from soccer_vision.identify.length_gate import EXCLUDED_SHORT, gate_by_length, lane_span_s

    fps = json.loads(tracks_path.read_text()).get("fps") or 0.0
    frames = {tid: [f for f, _ in samples] for tid, samples in track_boxes.items()}
    for tid, fr in frames.items():
        results[tid]["span_s"] = round(lane_span_s(fr, fps), 3) if fps else None

    if not fps:
        print("Length gate: off — tracks.json has no `fps`, so lane length is unknown")
        return set(track_boxes)

    eligible, counts = gate_by_length(frames, fps, min_seconds)
    if not min_seconds:
        print("Length gate: off — every lane is eligible, including single-frame "
              "fragments that give re-id one crop to vote on")
        return eligible

    for tid in track_boxes:
        if tid not in eligible and not results[tid]["excluded"]:
            results[tid]["excluded"] = EXCLUDED_SHORT
    print(f"Length gate: naming lanes of at least {min_seconds:g}s — "
          f"{counts['long_enough']} long enough, {counts['too_short']} too short "
          f"(excluded)")
    return eligible


def _resolve_gallery(flag, reid_cfg: dict, run_dir: Path) -> Path | None:
    """Gallery from the flag, else the profile, else the run's own — if it exists."""
    for candidate in (flag, reid_cfg.get("gallery"), run_dir / "gallery.npz"):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _run_reid(args, track_boxes, proxy_path, gallery_path, reid_cfg, profile, results,
              *, tracks_path, run_dir):
    import numpy as np

    from soccer_vision.identify.assign import Lane, assign_identities, concurrency
    from soccer_vision.identify.gallery import (
        EXCLUDED_NEGATIVE, NEGATIVE_LABEL, load_gallery, match_track, track_scores,
    )
    from soccer_vision.identify.reid import ReIDEmbedder, embed_tracks
    from soccer_vision.io.video import VideoReader

    gallery = load_gallery(gallery_path)
    n_players = sum(1 for n in gallery["names"] if n != NEGATIVE_LABEL)
    print(f"Gallery: {n_players} players, "
          f"{len(gallery['emb'])} exemplars ({gallery_path})")
    if NEGATIVE_LABEL in gallery["names"]:
        n_neg = int((gallery["label"] == gallery["names"].index(NEGATIVE_LABEL)).sum())
        print(f"  + a '{NEGATIVE_LABEL}' class of {n_neg} exemplars — lanes matching "
              f"it are rejected rather than named")

    min_sim = _first_set(args.min_similarity, reid_cfg.get("min_similarity"), 0.5)
    min_margin = _first_set(args.min_reid_margin, reid_cfg.get("min_margin"), 0.05)

    print("Loading re-id backbone...")
    embedder = ReIDEmbedder.from_pretrained(weights=None, device=args.device)

    reader = VideoReader(proxy_path)
    try:
        per_track = embed_tracks(
            track_boxes, embedder, reader,
            max_samples_per_track=min(args.max_samples, 20),
        )
    finally:
        reader.close()

    tids = list(per_track)
    scores = np.stack([track_scores(per_track[t], gallery) for t in tids]) \
        if tids else np.zeros((0, len(gallery["names"])), dtype="float32")

    # Kept whatever the naming rule, so the assignment can be re-solved or
    # re-tuned later without a GPU — the embedding pass is the expensive part.
    np.savez_compressed(
        run_dir / "reid_scores.npz",
        track_ids=np.asarray([str(t) for t in tids]),
        scores=scores,
        names=np.asarray(gallery["names"], dtype=object),
    )

    for tid, emb in per_track.items():
        results[tid]["n_obs"] = len(np.atleast_2d(emb))

    if _use_assignment(args, reid_cfg):
        fps = json.loads(tracks_path.read_text()).get("fps") or 0.0
        lanes = [Lane(str(t), *_extent_s(track_boxes[t], fps)) for t in tids]
        out = assign_identities(lanes, scores, gallery["names"],
                                min_similarity=min_sim, min_margin=min_margin)
        for tid in tids:
            a = out[str(tid)]
            results[tid]["similarity"] = round(a.score, 3)
            results[tid]["margin"] = round(a.margin, 3)
            results[tid]["open_margin"] = round(a.open_margin, 3)
            if a.blocked_by:
                results[tid]["blocked_by"] = [list(b) for b in a.blocked_by]
            if a.rejected:
                results[tid]["excluded"] = EXCLUDED_NEGATIVE
            elif a.name is not None:
                results[tid].update(name=a.name, source="reid",
                                    confidence=round(a.score, 3),
                                    jersey=_jersey_for(a.name, profile))
        rule = (f"one lane per player at a time, margin measured against the "
                f"identities still available")
        worst = concurrency(out, lanes)
        helped = sum(1 for t in tids
                     if out[str(t)].name and out[str(t)].margin > out[str(t)].open_margin)
        extra = (f"\n  {helped} lane(s) named on a margin the constraint opened up "
                 f"— an identity that would have been the runner-up was already "
                 f"committed to an overlapping lane") if helped else ""
        print(f"Naming rule: {rule}{extra}")
        if worst and max(worst.values()) > 1:  # pragma: no cover - invariant
            print(f"  WARNING: a name still lands on {max(worst.values())} lanes at once")
    else:
        print("Naming rule: each lane decided alone (--no-assign) — a name can land "
              "on several players at once")
        for tid, emb in per_track.items():
            m = match_track(emb, gallery, min_similarity=min_sim, min_margin=min_margin)
            results[tid]["similarity"] = round(m.similarity, 3)
            results[tid]["margin"] = round(m.margin, 3)
            if m.rejected:
                results[tid]["excluded"] = EXCLUDED_NEGATIVE
            elif m.name is not None:
                results[tid].update(name=m.name, source="reid",
                                    confidence=round(m.similarity, 3),
                                    jersey=_jersey_for(m.name, profile))

    matched = sum(1 for r in results.values() if r["source"] == "reid")
    print(f"Re-id matched {matched}/{len(track_boxes)} tracks "
          f"(min_similarity={min_sim}, min_margin={min_margin})")
    if NEGATIVE_LABEL in gallery["names"]:
        rejected = sum(1 for r in results.values() if r["excluded"] == EXCLUDED_NEGATIVE)
        print(f"  {rejected} lane(s) matched '{NEGATIVE_LABEL}' and were rejected "
              f"as not our players")


def _use_assignment(args, reid_cfg: dict) -> bool:
    """Mutual exclusion is on unless explicitly turned off.

    It is the default because the alternative is not a neutral baseline: deciding
    each lane alone puts one child's name on two to four concurrent lanes, which
    is what makes a reel's halo strobe between players.
    """
    if getattr(args, "no_assign", False):
        return False
    return bool(reid_cfg.get("assign", True))


def _extent_s(samples, fps: float) -> tuple[float, float]:
    """First and last second a lane is alive, from its ``(frame, box)`` samples."""
    frames = [f for f, _ in samples]
    if not frames or not fps:
        return 0.0, 0.0
    return min(frames) / fps, max(frames) / fps


def _jersey_for(name: str, profile: dict | None) -> int | None:
    """Number behind a re-id name, so ``--number 6`` still selects a matched lane.

    Rostered players resolve through the profile; a gallery seeded from OCR names
    unrostered players ``"#7"``, which carries its own number.
    """
    from soccer_vision.profiles.loader import get_jersey_by_name

    if name.startswith("#") and name[1:].isdigit():
        return int(name[1:])
    return get_jersey_by_name(profile, name) if profile else None


def _run_ocr(args, track_boxes, proxy_path, profile, results, *,
             verify=False, reid_cfg=None):
    from soccer_vision.identify.jersey_ocr import JerseyNumberRecognizer, assign_jerseys
    from soccer_vision.io.video import VideoReader
    from soccer_vision.profiles.loader import get_player

    to_check = sum(1 for t in track_boxes if results[t]["source"] == "reid")
    detail = (f" ({len(track_boxes) - to_check} to name, "
              f"{to_check} to cross-check re-id)") if verify else ""
    print(f"OCR on {len(track_boxes)} tracks{detail}...")
    if not track_boxes:
        return

    print("Loading jersey-number recognizer...")
    recognizer = JerseyNumberRecognizer.from_pretrained(
        model_id=args.model, device=args.device
    )
    vote_kwargs = {
        "min_votes": args.min_votes,
        "min_share": args.min_share,
        "min_margin": args.min_margin,
    }
    reader = VideoReader(proxy_path)
    try:
        votes = assign_jerseys(
            track_boxes, recognizer, reader,
            max_samples_per_track=args.max_samples,
            vote_kwargs=vote_kwargs,
            progress=True,
        )
    finally:
        reader.close()

    cc_kwargs = _crosscheck_kwargs(args, reid_cfg or {})
    drop = bool(getattr(args, "drop_on_conflict", False)
                or (reid_cfg or {}).get("drop_on_conflict"))
    checked: dict[str, int] = {}

    for tid, v in votes.items():
        r = results[tid]
        if r["source"] == "reid":
            # Never rename a re-id match — only corroborate or flag it.
            verdict = _apply_crosscheck(tid, r, v, cc_kwargs, drop=drop)
            checked[verdict] = checked.get(verdict, 0) + 1
            continue
        r.update(jersey=v.jersey, confidence=round(v.confidence, 3),
                 n_obs=v.n_obs, legible_frac=round(v.legible_frac, 3))
        if v.jersey is not None:
            player = get_player(profile, v.jersey) if profile else None
            r["name"] = (player or {}).get("name") or f"#{v.jersey}"
            r["source"] = "ocr"

    if verify:
        from soccer_vision.identify.crosscheck import AGREE, CONFLICT, NO_EVIDENCE

        fate = "dropped" if drop else "flagged, name kept"
        print(f"Cross-checked {sum(checked.values())} re-id matches: "
              f"{checked.get(AGREE, 0)} corroborated, "
              f"{checked.get(CONFLICT, 0)} contradicted ({fate}), "
              f"{checked.get(NO_EVIDENCE, 0)} no legible evidence "
              f"(min_reads={cc_kwargs['min_reads']}, "
              f"min_read_conf={cc_kwargs['min_read_conf']})")


def _apply_crosscheck(tid, r, vote, cc_kwargs, *, drop: bool = False) -> str:
    """Weigh a re-id-named track's jersey reads against the name it was given.

    Returns the verdict. A contradiction is always *recorded* — ``crosscheck``
    plus a ``conflict`` block holding both sides' evidence — but the name only
    goes away under ``drop``, because on the lanes we have hand-checked the
    reader was the one that was wrong (see
    ``runs/saints-u14g-full/identity_evidence/ground_truth.md``).
    """
    from soccer_vision.identify.crosscheck import crosscheck_jersey

    cc = crosscheck_jersey(r["jersey"], vote.reads, **cc_kwargs)
    r["crosscheck"] = cc.verdict
    r["legible_frac"] = round(vote.legible_frac, 3)
    if not cc.conflicts:
        return cc.verdict

    r["conflict"] = {
        "reid_name": r["name"], "reid_jersey": r["jersey"],
        "ocr_jersey": cc.jersey, "ocr_confidence": round(cc.confidence, 3),
        "n_obs": cc.n_obs,
    }
    print(f"  track {tid}: re-id said {r['name']} (#{r['jersey']}) but "
          f"{cc.n_obs} strong reads say #{cc.jersey} "
          f"(conf {cc.confidence:.2f}) — {'dropped' if drop else 'flagged'}",
          flush=True)
    if drop:
        r.update(jersey=None, name=None, source=None, confidence=0.0)
    return cc.verdict


def _crosscheck_kwargs(args, reid_cfg: dict) -> dict:
    """Veto thresholds from the flags, else the profile's ``reid:`` block, else defaults."""
    return {
        "min_reads": _first_set(args.conflict_min_reads,
                                reid_cfg.get("conflict_min_reads"), 4),
        "min_read_conf": _first_set(args.conflict_min_read_conf,
                                    reid_cfg.get("conflict_min_read_conf"), 0.7),
        "min_share": _first_set(args.conflict_min_share,
                                reid_cfg.get("conflict_min_share"), 0.75),
        "exclude": args.conflict_exclude_jersey
                   or reid_cfg.get("conflict_exclude_jersey") or (),
    }


def _first_set(*values):
    """First non-``None`` of flag, profile setting, built-in default."""
    return next(v for v in values if v is not None)
