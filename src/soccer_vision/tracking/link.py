"""Reconnect ByteTrack lanes across detector dropouts, after the fact.

ByteTrack hands back a lane per unbroken run of detections and never revisits
one it lost. On this footage that means **17,395 lanes for 22 players**, a median
lane of 1.4 s, and — the reason this module exists — **1,184 lanes (6.8%) that
die with the ball still inside the on-ball radius**, cutting a clip off while the
player is on the ball.

The tracker is not being asked something unreasonable: consecutive-sample IoU is
0.65 at the median and only 1.6% of steps fall below the accept floor. Lanes die
to *detector dropout* and occlusion in traffic, not to motion. So the fix is not
a better association threshold inside the tracker; it is a second pass that looks
at the lanes it produced and asks which pairs are the same person.

**Why a post-pass rather than tracker surgery.** It re-runs in seconds against a
saved ``tracks.json`` instead of the ~1.6 h a re-``process`` costs, so a
threshold can actually be A/B-ed. It is also auditable: every link is written out
with the evidence that justified it.

**What linking is really buying.** A chain inherits the identity of whichever
member re-id managed to name, so this is label propagation along motion
continuity — it routes around the appearance-matching ceiling (issue #25) rather
than fighting it. That is worth more than the fragmentation fix itself.

Scoring a candidate pair, cheapest gate first:

1. **Kit** must agree (and be known), which is free and rejects most of the field.
2. **Motion.** A player mid-run keeps going, so lane A's last position is the
   wrong thing to compare against: extrapolate A forward by its exit velocity to
   the frame B starts on, and B backward by its entry velocity to the frame A
   died on. Requiring *both* to agree is what separates a player continuing
   through a dropout from a different player who happened to be standing where
   the first one was last seen.
3. **Speed plausibility.** The implied gap-crossing speed must be one a footballer
   can produce; anything faster is a coincidence of position, not a continuation.

Appearance is deliberately *not* in the default gate — see ``score_pair`` — but
``appearance_fn`` accepts one, because "is this the same person 0.6 s later"
(same pose, same light, same kit) is a far easier question than the one the
gallery fails at, and it is the natural way to break crossing-player ties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

# A footballer tops out around 10 m/s. At this venue a player box is ~29 px wide
# for a ~0.45 m shoulder width, so ~64 px/m, giving ~640 px/s. Left generous:
# the camera pans, which adds apparent speed the model doesn't know about.
MAX_SPEED_PX_S = 900.0

# Velocity is fit over at most this many samples at a lane's edge. Longer is
# steadier but stales across a turn; 5 samples is 1.0 s at the 5 fps `process`
# writes, which is about as long as a youth player holds a heading.
VELOCITY_SAMPLES = 5


@dataclass
class LinkConfig:
    """Gate for joining lane A's end to lane B's start."""

    max_gap_s: float = 2.0
    max_dist_px: float = 150.0
    require_kit: bool = True
    use_motion: bool = True
    bidirectional: bool = True
    max_speed_px_s: float = MAX_SPEED_PX_S
    #: Optional ``(track_a, track_b) -> cosine similarity`` appearance check.
    appearance_fn: Callable[[str, str], float] | None = None
    min_appearance: float = 0.5
    #: Global (Hungarian) assignment instead of tightest-first greedy. Measured
    #: *worse* — see ``_assign_global``. Kept for reproducing that result.
    global_assignment: bool = False


@dataclass
class Link:
    """One accepted join, kept so a chain can be explained after the fact."""

    a: str
    b: str
    gap_s: float
    dist_px: float
    naive_dist_px: float
    speed_px_s: float
    appearance: float | None = None


@dataclass
class LinkResult:
    parent: dict[str, str]
    links: list[Link]
    stats: dict = field(default_factory=dict)

    def chains(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for tid in self.parent:
            out.setdefault(_find(self.parent, tid), []).append(tid)
        return out


def _find(parent: dict[str, str], a: str) -> str:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _foot(bbox) -> tuple[float, float]:
    """Feet, not centroid — a player's ground contact is what moves smoothly.

    A box centre rides up and down as the detector clips the head or includes a
    raised leg; the bottom edge tracks the pitch position that motion is actually
    continuous in.
    """
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def _velocity(samples: list[dict], *, at_end: bool) -> tuple[float, float]:
    """Least-squares px/frame at one edge of a lane, ``(0, 0)`` if too short."""
    edge = samples[-VELOCITY_SAMPLES:] if at_end else samples[:VELOCITY_SAMPLES]
    if len(edge) < 2:
        return 0.0, 0.0
    f = np.array([s["frame"] for s in edge], dtype=float)
    pts = np.array([_foot(s["bbox"]) for s in edge], dtype=float)
    f = f - f.mean()
    denom = float((f * f).sum())
    if denom <= 0:
        return 0.0, 0.0
    vx = float((f * (pts[:, 0] - pts[:, 0].mean())).sum() / denom)
    vy = float((f * (pts[:, 1] - pts[:, 1].mean())).sum() / denom)
    return vx, vy


@dataclass
class _Edge:
    tid: str
    first_frame: int
    last_frame: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    v_in: tuple[float, float]
    v_out: tuple[float, float]
    kit: str | None


def _edges(tracks: dict[str, list[dict]], teams: dict[str, str]) -> dict[str, _Edge]:
    out = {}
    for tid, samples in tracks.items():
        if not samples:
            continue
        out[tid] = _Edge(
            tid=tid,
            first_frame=samples[0]["frame"],
            last_frame=samples[-1]["frame"],
            start_xy=_foot(samples[0]["bbox"]),
            end_xy=_foot(samples[-1]["bbox"]),
            v_in=_velocity(samples, at_end=False),
            v_out=_velocity(samples, at_end=True),
            kit=teams.get(tid),
        )
    return out


def score_pair(a: _Edge, b: _Edge, fps: float, cfg: LinkConfig) -> Link | None:
    """Evidence that ``b`` continues ``a``, or ``None`` if the gate rejects it.

    Appearance is consulted only as a *veto* and only when a function is
    supplied. Two teammates in one kit at ~29 px wide are near-identical to the
    embedder, so a similarity score is far better at saying "definitely not the
    same person" than at ranking which continuation is right — it belongs in the
    gate, not in the cost.
    """
    gap_frames = b.first_frame - a.last_frame
    if gap_frames <= 0:
        return None                       # overlapping in time: not a continuation
    gap_s = gap_frames / fps
    if gap_s > cfg.max_gap_s:
        return None
    if cfg.require_kit and (a.kit is None or a.kit != b.kit):
        return None

    naive = math.dist(a.end_xy, b.start_xy)
    if cfg.use_motion:
        fwd = (a.end_xy[0] + a.v_out[0] * gap_frames,
               a.end_xy[1] + a.v_out[1] * gap_frames)
        err_fwd = math.dist(fwd, b.start_xy)
        if cfg.bidirectional:
            back = (b.start_xy[0] - b.v_in[0] * gap_frames,
                    b.start_xy[1] - b.v_in[1] * gap_frames)
            err_back = math.dist(back, a.end_xy)
            # Both directions must agree: a coincidence of position usually
            # satisfies one and not the other.
            dist = max(err_fwd, err_back)
        else:
            dist = err_fwd
    else:
        dist = naive

    if dist > cfg.max_dist_px:
        return None
    speed = naive / gap_s if gap_s > 0 else float("inf")
    if speed > cfg.max_speed_px_s:
        return None

    app = None
    if cfg.appearance_fn is not None:
        app = cfg.appearance_fn(a.tid, b.tid)
        if app is not None and app < cfg.min_appearance:
            return None

    return Link(a=a.tid, b=b.tid, gap_s=gap_s, dist_px=dist,
                naive_dist_px=naive, speed_px_s=speed, appearance=app)


def _same_player_box(a, b, *, max_centre_frac: float, max_scale: float) -> float:
    """Agreement that two boxes in the *same frame* are on the same player, or 0.

    **IoU is the wrong test here.** When the detector mints a duplicate box the
    two boxes sit on one player but disagree on extent — the handoff that breaks
    Morgan's chain on the U14G run pairs a 30x59 box with a 51x83 one, same
    player, IoU 0.40, below any threshold loose enough to be safe. IoU penalises
    that scale disagreement twice; centre distance measured in box heights does
    not, and a separate ratio test still keeps a near player from being merged
    with a far one.
    """
    ha, hb = a[3] - a[1], b[3] - b[1]
    if ha <= 0 or hb <= 0:
        return 0.0
    ratio = ha / hb if ha < hb else hb / ha
    if ratio < 1.0 / max_scale:
        return 0.0
    ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    d = math.dist(ca, cb) / ((ha + hb) / 2.0)
    if d > max_centre_frac:
        return 0.0
    return (1.0 - d / max_centre_frac) * ratio


def merge_duplicate_lanes(doc: dict, *, max_overlap_s: float = 0.5,
                          max_centre_frac: float = 0.4,
                          max_scale: float = 2.0) -> tuple[dict, dict]:
    """Collapse lanes that are one player under two ids. Returns ``(doc, stats)``.

    **This is dedup, not linking, and it has to run first.** ``link_tracks``
    requires ``b.first_frame > a.last_frame`` — a successor born *before* its
    predecessor died is rejected as "overlapping in time". But a duplicate
    detection is exactly that: a second box lands on a player already tracked,
    takes a fresh id, and the old lane dies a frame or two later, overlapping.
    On the 30 fps U14G run that is how **9.9% of lanes lasting >=10 s end**, and
    it is the single easiest link in the file — same player, same frame — that
    the gate refuses by construction. Following one hand-verified player through
    it took her chain from 12.4 s to 23.5 s.

    Doing it as a pre-pass rather than as extra edges inside ``link_tracks``
    matters: unioning dedup groups into already-linked chains cascades, because
    one bad merge fuses two long chains. Measured on the same run, merging into
    the link result put 5.6 bad handoffs on the average chain (one chain
    collected 705) where dedup-then-link left it at 0.45, below the 0.50 that
    linking alone already carried.

    Kit agreement is required and lanes with no kit are left alone. Geometry
    cannot tell one player under two ids from two players in contact, and
    without that gate 16.3% of these merges joined lanes the team classifier had
    placed in *different* kits — one of them stitching a 62 s white-kit lane onto
    Morgan's black one.
    """
    tracks = doc["tracks"]
    teams = doc.get("teams", {}) or {}
    fps = float(doc["fps"])

    births: dict[int, list[str]] = {}
    for tid, s in tracks.items():
        if s:
            births.setdefault(s[0]["frame"], []).append(tid)
    by_frame = {tid: {s["frame"]: s["bbox"] for s in samples}
                for tid, samples in tracks.items()}

    win = int(round(max_overlap_s * fps))
    pairs: list[tuple[float, str, str]] = []
    for tid, samples in tracks.items():
        if not samples:
            continue
        death = samples[-1]["frame"]
        kit = teams.get(tid)
        if kit is None:
            continue
        for df in range(-win, 1):
            for cand in births.get(death + df, ()):
                if cand == tid or teams.get(cand) != kit:
                    continue
                cbox = by_frame[cand].get(death + df)
                pbox = by_frame[tid].get(death + df)
                if cbox is None or pbox is None:
                    continue
                v = _same_player_box(pbox, cbox, max_centre_frac=max_centre_frac,
                                     max_scale=max_scale)
                if v > 0:
                    pairs.append((v, tid, cand))

    parent = {tid: tid for tid in tracks}
    claimed: set[str] = set()
    used_pred: set[str] = set()
    n = 0
    for _v, pred, succ in sorted(pairs, key=lambda p: -p[0]):
        if succ in claimed or pred in used_pred:
            continue
        ra, rb = _find(parent, pred), _find(parent, succ)
        if ra == rb:
            continue
        parent[rb] = ra
        claimed.add(succ)
        used_pred.add(pred)
        n += 1

    groups: dict[str, list[str]] = {}
    for tid in tracks:
        groups.setdefault(_find(parent, tid), []).append(tid)

    new_tracks: dict[str, list[dict]] = {}
    new_teams: dict[str, str] = {}
    for root, members in groups.items():
        rows = [s for m in members for s in tracks[m]]
        rows.sort(key=lambda s: s["frame"])
        seen: set[int] = set()
        deduped = []
        for r in rows:
            if r["frame"] in seen:
                continue
            seen.add(r["frame"])
            deduped.append(r)
        new_tracks[root] = deduped
        kit = next((teams[m] for m in members if m in teams), None)
        if kit is not None:
            new_teams[root] = kit

    out = dict(doc)
    out["tracks"] = new_tracks
    out["teams"] = new_teams
    # ``alias`` maps every original lane id to the lane that now carries it, so a
    # caller holding a lane id from before the merge (a hand-verified seed, a
    # jerseys.json key) can still find it.
    return out, {"lanes_before": len(tracks), "lanes_after": len(new_tracks),
                 "merges": n,
                 "alias": {tid: _find(parent, tid) for tid in tracks}}


def remap_jerseys(jerseys: dict, alias: dict[str, str]) -> tuple[dict, int]:
    """Re-key ``jerseys.json`` onto deduped lane ids. Returns ``(doc, n_conflicts)``.

    Dedup gives merged lanes a single id, so identities read against the old ids
    have to follow. Two lanes that turn out to be one player can carry two
    different names — that is a naming error the merge has just exposed — so the
    higher-similarity name wins and the disagreement is counted rather than
    silently resolved.
    """
    tracks = jerseys.get("tracks", {})
    out: dict[str, dict] = {}
    conflicts = 0
    for tid, rec in tracks.items():
        key = alias.get(tid, tid)
        prev = out.get(key)
        if prev is None:
            out[key] = dict(rec)
            continue
        if prev.get("name") and rec.get("name") and prev["name"] != rec["name"]:
            conflicts += 1
        if (rec.get("similarity") or 0.0) > (prev.get("similarity") or 0.0):
            out[key] = dict(rec)
    doc = dict(jerseys)
    doc["tracks"] = out
    return doc, conflicts


def link_tracks(doc: dict, cfg: LinkConfig | None = None) -> LinkResult:
    """Join lanes end-to-start across dropouts. Never merges overlapping lanes.

    Each lane may be extended by at most one successor and may continue at most
    one predecessor, so chains stay linear — a player is in one place at a time,
    and letting a lane fan out to several would build a chain that is not a
    person.
    """
    cfg = cfg or LinkConfig()
    tracks = doc["tracks"]
    fps = float(doc["fps"])
    edges = _edges(tracks, doc.get("teams", {}) or {})

    by_start: dict[int, list[_Edge]] = {}
    for e in edges.values():
        by_start.setdefault(e.first_frame, []).append(e)
    start_frames = np.array(sorted(by_start), dtype=np.int64)

    max_gap_frames = int(round(cfg.max_gap_s * fps))
    candidates: list[Link] = []
    for a in edges.values():
        lo = np.searchsorted(start_frames, a.last_frame + 1, "left")
        hi = np.searchsorted(start_frames, a.last_frame + max_gap_frames, "right")
        for f in start_frames[lo:hi]:
            for b in by_start[int(f)]:
                if b.tid == a.tid:
                    continue
                link = score_pair(a, b, fps, cfg)
                if link is not None:
                    candidates.append(link)

    accepted = (_assign_global(candidates) if cfg.global_assignment
                else _assign_greedy(candidates))

    parent = {tid: tid for tid in tracks}
    links: list[Link] = []
    for link in accepted:
        ra, rb = _find(parent, link.a), _find(parent, link.b)
        if ra == rb:
            continue
        parent[rb] = ra
        links.append(link)

    n_chains = len({_find(parent, t) for t in parent})
    return LinkResult(parent=parent, links=links, stats={
        "lanes": len(tracks),
        "candidates": len(candidates),
        "links": len(links),
        "chains": n_chains,
    })


def _assign_greedy(candidates: Iterable[Link]) -> list[Link]:
    """Tightest link first, one successor and one predecessor per lane."""
    used_a: set[str] = set()
    used_b: set[str] = set()
    out = []
    for link in sorted(candidates, key=lambda l: l.dist_px):
        if link.a in used_a or link.b in used_b:
            continue
        used_a.add(link.a)
        used_b.add(link.b)
        out.append(link)
    return out


def _assign_global(candidates: list[Link]) -> list[Link]:
    """Hungarian assignment over the whole lane set. **Measured worse — don't.**

    The argument for it was crossing players: greedy takes the tightest pair
    first, and the tightest pair can be the *swap*, which then forces the two
    genuine continuations to be mis-paired. Solving jointly should fix that.

    It doesn't, because minimum-cost assignment also maximises how many pairs
    get matched, and leaving a lane unlinked costs nothing here. So it reaches
    for marginal links that greedy correctly declines, and precision drops:
    98% → 94% at a 0.2 s gap, 94% → 91% at 1.0 s, on 400 split lanes
    (``slurm/eval_track_linking.py``). Fixing it properly needs a dummy
    "no-link" column priced at the gate threshold; until someone does that,
    greedy wins. Crossings turn out to be rarer than marginal pairs.
    """
    if not candidates:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _assign_greedy(candidates)

    a_ids = sorted({l.a for l in candidates})
    b_ids = sorted({l.b for l in candidates})
    ai = {t: i for i, t in enumerate(a_ids)}
    bi = {t: i for i, t in enumerate(b_ids)}
    BIG = 1e6
    cost = np.full((len(a_ids), len(b_ids)), BIG, dtype=float)
    best: dict[tuple[int, int], Link] = {}
    for l in candidates:
        i, j = ai[l.a], bi[l.b]
        if l.dist_px < cost[i, j]:
            cost[i, j] = l.dist_px
            best[(i, j)] = l
    rows, cols = linear_sum_assignment(cost)
    return [best[(i, j)] for i, j in zip(rows, cols)
            if cost[i, j] < BIG and (i, j) in best]


def apply_links(doc: dict, result: LinkResult, *, interpolate: bool = True) -> dict:
    """Rewrite a tracks doc with chains merged, optionally filling the gaps.

    Interpolated samples are marked ``"interpolated": true`` and are **not**
    detections — they are where the player must have been if they walked a
    straight line through the dropout. Halos and on-ball geometry want them;
    anything training a detector must drop them, which is why they are flagged
    rather than silently blended in.
    """
    tracks = doc["tracks"]
    teams = doc.get("teams", {}) or {}
    chains = result.chains()
    interval = int(doc.get("sample_interval", 1)) or 1

    new_tracks: dict[str, list[dict]] = {}
    new_teams: dict[str, str] = {}
    for root, members in chains.items():
        members = sorted(members, key=lambda t: tracks[t][0]["frame"])
        samples: list[dict] = []
        for i, tid in enumerate(members):
            if i and interpolate:
                samples.extend(_bridge(samples[-1], tracks[tid][0], interval))
            samples.extend(tracks[tid])
        new_tracks[root] = samples
        kit = next((teams[t] for t in members if t in teams), None)
        if kit is not None:
            new_teams[root] = kit

    out = dict(doc)
    out["tracks"] = new_tracks
    out["teams"] = new_teams
    out["linking"] = {
        "lanes_before": result.stats["lanes"],
        "lanes_after": len(new_tracks),
        "links": result.stats["links"],
        "interpolated_samples": sum(
            1 for s in (x for v in new_tracks.values() for x in v)
            if s.get("interpolated")
        ),
    }
    return out


def _bridge(last: dict, nxt: dict, interval: int) -> list[dict]:
    """Straight-line fill between two samples, exclusive of both ends."""
    f0, f1 = last["frame"], nxt["frame"]
    steps = (f1 - f0) // interval
    if steps <= 1:
        return []
    a = np.array(last["bbox"], dtype=float)
    b = np.array(nxt["bbox"], dtype=float)
    out = []
    for k in range(1, steps):
        t = k / steps
        out.append({
            "frame": int(f0 + k * interval),
            "bbox": list(a + (b - a) * t),
            "interpolated": True,
        })
    return out


def propagate_names(jerseys: dict, result: LinkResult) -> tuple[dict, dict]:
    """Push each chain's identity onto every lane in it.

    This is the point of linking. Re-id names ~11% of lanes; a chain that
    contains one named lane can name the rest, without the embedder having to
    win on a crop it would have abstained on.

    Conflicting chains — two lanes named as *different* players — keep the
    higher-similarity name and are reported, because a conflict is evidence that
    either a link or a name is wrong and the count is the honest error signal
    available without hand labelling.

    A lane whose jersey reads vetoed its re-id name (``conflict`` in
    ``jerseys.json``, see :mod:`soccer_vision.identify.crosscheck`) will not
    inherit a *different* number than the one read off its shirt — otherwise
    linking would quietly hand back the identity the reader just disproved.
    """
    tracks = jerseys.get("tracks", {})
    chains = result.chains()
    conflicts = []
    blocked = 0
    out = {tid: dict(rec) for tid, rec in tracks.items()}
    for root, members in chains.items():
        named = [(tracks[t].get("similarity") or 0.0, t, tracks[t]["name"])
                 for t in members
                 if t in tracks and tracks[t].get("name")]
        if not named:
            continue
        distinct = {n for _, _, n in named}
        if len(distinct) > 1:
            conflicts.append({"chain": root, "names": sorted(distinct),
                              "members": sorted(members)})
        sim, src, name = max(named)
        # `jersey`, not just `name`, is what selection actually matches on
        # (`identify.resolve.tracks_for` maps a --player through the roster to a
        # number and compares that). Propagating the name alone leaves every
        # inherited lane invisible to --player/--number, which is the whole point.
        jersey = tracks[src].get("jersey")
        for t in members:
            rec = out.setdefault(t, {})
            if _vetoed_against(rec, jersey):
                blocked += 1
                continue
            if not rec.get("name"):
                rec["name"] = name
                rec["jersey"] = jersey
                rec["similarity"] = sim
                rec["source"] = "linked"
                rec["linked_from"] = src
    doc = dict(jerseys)
    doc["tracks"] = out
    return doc, {
        "chains_with_a_name": sum(
            1 for m in chains.values()
            if any(tracks.get(t, {}).get("name") for t in m)),
        "conflicts": len(conflicts),
        "conflict_detail": conflicts[:50],
        "blocked_by_jersey": blocked,
        "named_before": sum(1 for r in tracks.values() if r.get("name")),
        "named_after": sum(1 for r in out.values() if r.get("name")),
    }


def _vetoed_against(rec: dict, jersey) -> bool:
    """True when this lane's own jersey reads contradict the number being propagated."""
    conflict = rec.get("conflict")
    if not conflict or jersey is None:
        return False
    read = conflict.get("ocr_jersey")
    return read is not None and int(read) != int(jersey)
