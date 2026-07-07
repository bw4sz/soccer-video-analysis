"""YouTube search + Creative-Commons filtering + section download via yt-dlp.

The public entry point is :func:`harvest`. The pieces below it are split so the
pure decision logic (:func:`is_creative_commons`, :func:`midpoint_window`) is
unit-testable without a network, and the two yt-dlp touchpoints
(:func:`search_candidates`, :func:`probe_video`, :func:`download_section`) are
thin enough to monkeypatch in tests.

yt-dlp is an optional dependency (``pip install 'soccer-vision[harvest]'``) and
needs ffmpeg on PATH for the keyframe-accurate section cut. Both are imported
lazily so the rest of the package works without them.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from soccer_vision.harvest.manifest import ClipRecord, Manifest
from soccer_vision.harvest.queries import DEFAULT_QUERIES

# yt-dlp reports an uploader's licence in the ``license`` info-dict field. For
# CC-BY YouTube videos it reads "Creative Commons Attribution license (reuse
# allowed)"; everything else is "Standard YouTube License" or absent.
_CC_MARKER = "creative commons"

# YouTube's own search "Features: Creative Commons" filter, url-encoded, applied
# via the results-page ``sp`` param. Without it, plain search returns ~all
# Standard-licence videos and CC-BY yield is near zero. We still re-verify each
# video's ``license`` after probing (defence in depth).
_CC_SEARCH_FILTER = "EgIwAQ%3D%3D"


def _import_yt_dlp():
    try:
        import yt_dlp  # noqa: F401
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import download_range_func
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "yt-dlp is required to harvest clips. Install it with:\n"
            "    pip install 'soccer-vision[harvest]'\n"
            "and make sure ffmpeg is available (on HiPerGator: `module load ffmpeg`)."
        ) from e
    return YoutubeDL, download_range_func


# --------------------------------------------------------------------------- #
# Pure logic (no network) — the unit-test surface.
# --------------------------------------------------------------------------- #
def is_creative_commons(info: dict) -> bool:
    """True if the video's licence is a Creative Commons one (reuse allowed)."""
    lic = (info.get("license") or "").lower()
    return _CC_MARKER in lic


# American-football signals. "football" alone is ambiguous (soccer in most of the
# world), so we only reject on terms that are specific to the US game — this
# catches gridiron uploads that slipped through soccer-worded queries without
# nuking legitimate British/Australian "football" (= soccer) content.
_AMERICAN_FOOTBALL = re.compile(
    r"\b(nfl|american football|flag football|gridiron|quarterback|touchdown|"
    r"pop ?warner|tackle football|friday night lights|varsity football|"
    r"7v7 football|madden|end ?zone|wide receiver|running back)\b",
    re.IGNORECASE,
)
# The gridiron-ball emoji is an unambiguous American-football marker that youth
# US-football channels use heavily (and soccer channels essentially never).
_GRIDIRON_EMOJI = "\U0001f3c8"  # 🏈


def is_american_football(info: dict) -> bool:
    """True if the title/description look like American football, not soccer."""
    text = f"{info.get('title', '')} {(info.get('description') or '')[:400]}"
    return _GRIDIRON_EMOJI in text or bool(_AMERICAN_FOOTBALL.search(text))


def midpoint_window(
    duration_s: float, clip_len_s: float, position: str = "middle",
    frac: float = 0.6,
) -> tuple[float, float]:
    """Return ``(start_s, length_s)`` for a clip of ``clip_len_s``.

    ``middle`` centres the clip at ``frac`` of the match. We default to 0.6, not
    0.5, on purpose: the exact video centre lands on the halftime / second-half
    kickoff, so a 50% clip almost always catches a restart instead of live play.
    0.6 sits mid-second-half — continuous open play, past the kickoff and well
    before the final whistle.

    ``random`` places the clip uniformly within the interior (avoiding the
    first/last clip-length so we never clip the opening/closing whistle). Length
    is clamped for very short videos so ``start`` never goes negative.
    """
    length = min(clip_len_s, duration_s)
    if position == "random" and duration_s > 2 * length:
        start = random.uniform(length, duration_s - 2 * length)
    else:  # middle (default) and the short-video fallback
        start = max(0.0, duration_s * frac - length / 2)
    # Keep the whole window inside the video.
    start = min(start, max(0.0, duration_s - length))
    return start, length


# --------------------------------------------------------------------------- #
# yt-dlp touchpoints — thin, monkeypatchable.
# --------------------------------------------------------------------------- #
def search_candidates(query: str, limit: int) -> list[dict]:
    """Flat YouTube search (CC filter applied) → lightweight candidate dicts.

    Hits the results page with the Creative-Commons ``sp`` filter so we mostly
    get reusable videos up front; ``extract_flat`` keeps it to one cheap request
    per query. Licence and exact duration come later from :func:`probe_video`
    only for survivors.
    """
    YoutubeDL, _ = _import_yt_dlp()
    url = (
        f"https://www.youtube.com/results?search_query={quote(query)}"
        f"&sp={_CC_SEARCH_FILTER}"
    )
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist",
            "skip_download": True, "playlistend": limit}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = [e for e in (info.get("entries") or []) if e and e.get("id")]
    return entries[:limit]


def probe_video(url: str) -> dict | None:
    """Full metadata extract for one video (licence, duration, channel...).

    Returns ``None`` for videos yt-dlp can't resolve (private, geo-blocked,
    removed) so the caller can just skip them.
    """
    YoutubeDL, _ = _import_yt_dlp()
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception:  # noqa: BLE001 - unavailable video is not fatal
        return None


def _cleanup_partials(clips_dir: Path, video_id: str) -> None:
    """Remove yt-dlp/ffmpeg leftovers for a failed download (``*.part`` etc.)."""
    for pattern in (f"{video_id}*.part", f"{video_id}*.ytdl", f"{video_id}*.temp"):
        for leftover in clips_dir.glob(pattern):
            leftover.unlink(missing_ok=True)


def download_section(
    url: str, out_path: Path, start_s: float, length_s: float, *, max_height: int = 720
) -> None:
    """Download only ``[start_s, start_s+length_s]`` to ``out_path`` (mp4).

    We deliberately do **not** force keyframes at the cut points. Forcing them
    makes ffmpeg re-encode the boundary, which is slow and — fetching byte
    ranges from googlevideo — fails often (~half our attempts hit "ffmpeg exited
    with code 1"). That re-encode was the whole reliability problem. Without it,
    yt-dlp stream-copies from the nearest keyframe, so boundaries drift a few
    seconds; irrelevant for a ~10s live-play clip. Retries ride out transient
    googlevideo read errors.

    Format prefers merged ``bestvideo+bestaudio`` up to ``max_height`` (720p by
    default): YouTube's single progressive streams top out at 360p, so a
    progressive-first selector would silently cap resolution. Never downloads the
    full match.
    """
    YoutubeDL, download_range_func = _import_yt_dlp()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = str(out_path.with_suffix(""))
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "format": (
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": stem + ".%(ext)s",
        "download_ranges": download_range_func(None, [(start_s, start_s + length_s)]),
        "force_keyframes_at_cuts": False,
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 3,
        # Always re-fetch: yt-dlp skips a download when the target file already
        # exists, which silently keeps a stale/low-res clip on a re-run.
        "overwrites": True,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
@dataclass
class HarvestResult:
    downloaded: int
    scanned: int
    skipped_seen: int
    skipped_license: int
    skipped_channel_cap: int
    skipped_duration: int
    skipped_football: int
    manifest_path: Path


def harvest(
    out_dir: str | Path,
    *,
    target: int = 200,
    queries: list[str] | None = None,
    clip_len_s: float = 10.0,
    position: str = "middle",
    position_frac: float = 0.6,
    max_per_channel: int = 2,
    per_query: int = 50,
    min_duration_s: float = 300.0,
    max_height: int = 720,
    dry_run: bool = False,
    progress=print,
) -> HarvestResult:
    """Harvest up to ``target`` CC-BY youth-soccer clips into ``out_dir``.

    Resumes from any existing ``manifest.jsonl`` in ``out_dir`` and stops once
    ``target`` clips (counting prior runs) are on disk. ``dry_run`` performs
    search + licence filtering and records nothing — useful to gauge yield.
    """
    out_dir = Path(out_dir)
    clips_dir = out_dir / "clips"
    manifest = Manifest(out_dir / "manifest.jsonl")
    queries = queries or DEFAULT_QUERIES

    res = HarvestResult(
        downloaded=0, scanned=0, skipped_seen=0, skipped_license=0,
        skipped_channel_cap=0, skipped_duration=0, skipped_football=0,
        manifest_path=manifest.path,
    )
    # In-memory bookkeeping seeded from prior runs. Kept separate from disk
    # writes so ``dry_run`` can count without persisting anything.
    seen = set(manifest.seen_ids)
    channel_counts = manifest.channel_counts
    total = len(manifest)

    for query in queries:
        if total >= target:
            break
        progress(f"[search] {query!r}")
        try:
            candidates = search_candidates(query, per_query)
        except Exception as e:  # noqa: BLE001 - one bad query shouldn't abort
            progress(f"  search failed: {e}")
            continue

        for cand in candidates:
            if total >= target:
                break
            vid = cand["id"]
            if vid in seen:
                res.skipped_seen += 1
                continue
            seen.add(vid)

            url = cand.get("url") or f"https://www.youtube.com/watch?v={vid}"
            info = probe_video(url)
            res.scanned += 1
            if info is None:
                continue

            if not is_creative_commons(info):
                res.skipped_license += 1
                continue

            if is_american_football(info):
                res.skipped_football += 1
                continue

            duration = float(info.get("duration") or 0.0)
            if duration < min_duration_s:
                res.skipped_duration += 1
                continue

            channel_id = info.get("channel_id") or info.get("channel") or vid
            if channel_counts[channel_id] >= max_per_channel:
                res.skipped_channel_cap += 1
                continue

            start_s, length_s = midpoint_window(
                duration, clip_len_s, position, position_frac
            )
            clip_path = clips_dir / f"{vid}.mp4"
            title = info.get("title", "")
            progress(
                f"  [keep] {title[:60]!r} ({duration/60:.0f} min) "
                f"→ clip @ {start_s/60:.1f} min"
            )

            if not dry_run:
                try:
                    download_section(
                        info.get("webpage_url") or url, clip_path,
                        start_s, length_s, max_height=max_height,
                    )
                except Exception as e:  # noqa: BLE001 - skip a failed download
                    progress(f"    download failed, skipping: {e}")
                    _cleanup_partials(clips_dir, vid)
                    continue
                # yt-dlp can exit 0 yet leave only a .part (e.g. ffmpeg range
                # hiccup). Only record clips that actually landed on disk.
                if not (clip_path.exists() and clip_path.stat().st_size > 0):
                    progress("    no output produced, skipping")
                    _cleanup_partials(clips_dir, vid)
                    continue

            if not dry_run:
                record = ClipRecord(
                    video_id=vid,
                    url=info.get("webpage_url") or url,
                    title=title,
                    channel=info.get("channel") or info.get("uploader") or "",
                    channel_id=str(channel_id),
                    license=info.get("license") or "",
                    duration_s=duration,
                    clip_start_s=start_s,
                    clip_len_s=length_s,
                    query=query,
                    path=str(clip_path),
                )
                manifest.append(record)

            channel_counts[channel_id] += 1
            total += 1
            res.downloaded += 1

    if not dry_run and len(manifest):
        manifest.write_attribution()
    return res
