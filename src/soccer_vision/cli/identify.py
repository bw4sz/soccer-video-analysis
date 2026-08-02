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

    if method.startswith("reid"):
        _run_reid(args, track_boxes, proxy_path, gallery_path, reid_cfg, profile, results)

    verify = method == "reid+ocr" and not args.no_ocr_verify

    if method.endswith("ocr"):
        # OCR sees the tracks the gallery couldn't name, plus — when verifying —
        # the ones it did, to check their numbers against the reads.
        todo = {t: b for t, b in track_boxes.items()
                if method == "ocr" or verify or results[t]["name"] is None}
        _run_ocr(args, todo, proxy_path, profile, results,
                 verify=verify, reid_cfg=reid_cfg)

    doc = {
        "video": proxy_path.name,
        "method": method,
        "gallery": str(gallery_path) if gallery_path else None,
        "model": args.model or "parseq",
        "ocr_verify": verify,
        "tracks": {str(t): r for t, r in results.items()},
    }
    jerseys_path.write_text(json.dumps(doc, indent=2))

    named = sum(1 for r in results.values() if r["name"] or r["jersey"] is not None)
    dropped = sum(1 for r in results.values() if r["conflict"])
    by_source: dict[str, int] = {}
    for r in results.values():
        if r["source"]:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(by_source.items())) or "none"
    print(f"\nIdentified {named}/{len(results)} tracks ({breakdown}). Saved: {jerseys_path}")
    if dropped:
        print(f"{dropped} re-id match(es) dropped on a jersey conflict — see the "
              f"`conflict` records in jerseys.json")
    print(f"Next: soccer-vision reel --run {run_dir} --player <name>   (or --number <N>)")


def _blank() -> dict:
    return {"jersey": None, "name": None, "source": None, "confidence": 0.0,
            "n_obs": 0, "legible_frac": 0.0, "similarity": None,
            "crosscheck": None, "conflict": None}


def _resolve_gallery(flag, reid_cfg: dict, run_dir: Path) -> Path | None:
    """Gallery from the flag, else the profile, else the run's own — if it exists."""
    for candidate in (flag, reid_cfg.get("gallery"), run_dir / "gallery.npz"):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _run_reid(args, track_boxes, proxy_path, gallery_path, reid_cfg, profile, results):
    from soccer_vision.identify.gallery import load_gallery, match_track
    from soccer_vision.identify.reid import ReIDEmbedder, embed_tracks
    from soccer_vision.io.video import VideoReader

    gallery = load_gallery(gallery_path)
    print(f"Gallery: {len(gallery['names'])} players, "
          f"{len(gallery['emb'])} exemplars ({gallery_path})")

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

    for tid, emb in per_track.items():
        m = match_track(emb, gallery, min_similarity=min_sim, min_margin=min_margin)
        results[tid]["similarity"] = round(m.similarity, 3)
        results[tid]["n_obs"] = m.n_crops
        if m.name is not None:
            results[tid].update(name=m.name, source="reid",
                                confidence=round(m.similarity, 3),
                                jersey=_jersey_for(m.name, profile))

    matched = sum(1 for r in results.values() if r["source"] == "reid")
    print(f"Re-id matched {matched}/{len(track_boxes)} tracks "
          f"(min_similarity={min_sim}, min_margin={min_margin})")


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
    checked: dict[str, int] = {}

    for tid, v in votes.items():
        r = results[tid]
        if r["source"] == "reid":
            # Never rename a re-id match — only corroborate or veto it.
            verdict = _apply_crosscheck(tid, r, v, cc_kwargs)
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

        print(f"Cross-checked {sum(checked.values())} re-id matches: "
              f"{checked.get(AGREE, 0)} corroborated, "
              f"{checked.get(CONFLICT, 0)} dropped on a jersey conflict, "
              f"{checked.get(NO_EVIDENCE, 0)} no legible evidence "
              f"(min_reads={cc_kwargs['min_reads']}, "
              f"min_read_conf={cc_kwargs['min_read_conf']})")


def _apply_crosscheck(tid, r, vote, cc_kwargs) -> str:
    """Weigh a re-id-named track's jersey reads; drop the track if they contradict it.

    Returns the verdict. On a conflict the name and number are cleared — the
    reads say who this *isn't*, not who it is — while ``similarity`` and the
    ``conflict`` record stay so the drop is auditable in ``jerseys.json``.
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
          f"(conf {cc.confidence:.2f}) — dropped", flush=True)
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
