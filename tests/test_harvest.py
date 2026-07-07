"""Harvest tests — pure logic plus a fully-mocked orchestrator run (no network)."""

import json

import pytest

from soccer_vision.harvest import youtube
from soccer_vision.harvest.manifest import ClipRecord, Manifest
from soccer_vision.harvest.youtube import harvest, is_creative_commons, midpoint_window


# --------------------------------------------------------------------------- #
# Pure logic
# --------------------------------------------------------------------------- #
def test_is_creative_commons_accepts_cc_by():
    assert is_creative_commons(
        {"license": "Creative Commons Attribution license (reuse allowed)"}
    )


@pytest.mark.parametrize("lic", ["Standard YouTube License", "", None])
def test_is_creative_commons_rejects_everything_else(lic):
    assert not is_creative_commons({"license": lic})


def test_midpoint_window_defaults_past_halftime_kickoff():
    # Default frac=0.6 → mid-second-half, not the 50% halftime-kickoff point.
    start, length = midpoint_window(600.0, 10.0, "middle")
    assert length == 10.0
    assert start == pytest.approx(355.0)  # 600*0.6 - 5


def test_midpoint_window_frac_override():
    # frac=0.5 recovers the true centre; frac=0.35 lands in the first half.
    assert midpoint_window(600.0, 10.0, "middle", frac=0.5)[0] == pytest.approx(295.0)
    assert midpoint_window(600.0, 10.0, "middle", frac=0.35)[0] == pytest.approx(205.0)


def test_midpoint_window_clamps_short_video():
    # A clip longer than the video must not produce a negative start.
    start, length = midpoint_window(6.0, 10.0, "middle")
    assert start == 0.0
    assert length == 6.0


def test_midpoint_window_random_stays_interior():
    for _ in range(50):
        start, length = midpoint_window(600.0, 10.0, "random")
        assert length <= start <= 600.0 - 2 * length


# --------------------------------------------------------------------------- #
# Manifest: dedup / resume / channel caps / attribution
# --------------------------------------------------------------------------- #
def _record(vid, channel_id="chan", title="Game", query="q"):
    return ClipRecord(
        video_id=vid, url=f"https://youtu.be/{vid}", title=title,
        channel="Some Club", channel_id=channel_id,
        license="Creative Commons Attribution license (reuse allowed)",
        duration_s=600.0, clip_start_s=295.0, clip_len_s=10.0,
        query=query, path=f"clips/{vid}.mp4",
    )


def test_manifest_roundtrip_and_resume(tmp_path):
    path = tmp_path / "manifest.jsonl"
    m = Manifest(path)
    m.append(_record("aaa", channel_id="c1"))
    m.append(_record("bbb", channel_id="c1"))

    reloaded = Manifest(path)  # simulate a fresh run reading prior state
    assert reloaded.seen_ids == {"aaa", "bbb"}
    assert reloaded.channel_counts["c1"] == 2
    assert len(reloaded) == 2
    # File is valid JSONL.
    lines = path.read_text().strip().splitlines()
    assert all(json.loads(ln)["video_id"] for ln in lines)


def test_write_attribution_lists_every_source(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    m.append(_record("aaa", title="Cool Match"))
    out = m.write_attribution()
    text = out.read_text()
    assert "Cool Match" in text
    assert "aaa" in text


# --------------------------------------------------------------------------- #
# Orchestrator with all yt-dlp touchpoints mocked
# --------------------------------------------------------------------------- #
def _fake_info(vid, *, license="Creative Commons Attribution license (reuse allowed)",
               duration=600.0, channel_id="chan"):
    return {
        "id": vid,
        "webpage_url": f"https://www.youtube.com/watch?v={vid}",
        "title": f"Match {vid}",
        "channel": "Club",
        "channel_id": channel_id,
        "license": license,
        "duration": duration,
    }


@pytest.fixture
def mock_yt(monkeypatch):
    """Wire search/probe/download to in-memory fakes; record downloads."""
    downloads = []

    def fake_search(query, limit):
        # Two candidates per query, ids derived from the query label.
        return [{"id": f"{query}-{i}", "url": f"https://youtu.be/{query}-{i}"}
                for i in range(2)]

    def fake_probe(url):
        vid = url.rsplit("/", 1)[-1]
        return youtube_infos.get(vid)

    def fake_download(url, out_path, start, length, *, max_height=720):
        # Mimic yt-dlp landing a real file on disk (the orchestrator verifies it).
        from pathlib import Path
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-clip")
        downloads.append((url, str(out_path), start, length))

    youtube_infos = {}
    monkeypatch.setattr(youtube, "search_candidates", fake_search)
    monkeypatch.setattr(youtube, "probe_video", fake_probe)
    monkeypatch.setattr(youtube, "download_section", fake_download)
    return youtube_infos, downloads


def test_harvest_filters_license_and_respects_target(tmp_path, mock_yt):
    infos, downloads = mock_yt
    # Query "cc": both CC. Query "std": both standard licence.
    infos["cc-0"] = _fake_info("cc-0", channel_id="a")
    infos["cc-1"] = _fake_info("cc-1", channel_id="b")
    infos["std-0"] = _fake_info("std-0", license="Standard YouTube License")
    infos["std-1"] = _fake_info("std-1", license="Standard YouTube License")

    res = harvest(tmp_path, target=10, queries=["cc", "std"], max_per_channel=2)

    assert res.downloaded == 2
    assert res.skipped_license == 2
    assert len(downloads) == 2
    # Manifest + attribution written.
    assert (tmp_path / "manifest.jsonl").exists()
    assert (tmp_path / "ATTRIBUTION.md").exists()


def test_harvest_channel_cap(tmp_path, mock_yt):
    infos, downloads = mock_yt
    # Both candidates share a channel; cap of 1 keeps only the first.
    infos["cc-0"] = _fake_info("cc-0", channel_id="same")
    infos["cc-1"] = _fake_info("cc-1", channel_id="same")

    res = harvest(tmp_path, target=10, queries=["cc"], max_per_channel=1)

    assert res.downloaded == 1
    assert res.skipped_channel_cap == 1


def test_harvest_skips_short_videos(tmp_path, mock_yt):
    infos, _ = mock_yt
    infos["cc-0"] = _fake_info("cc-0", duration=60.0)   # too short
    infos["cc-1"] = _fake_info("cc-1", duration=600.0)

    res = harvest(tmp_path, target=10, queries=["cc"], min_duration_s=300.0)

    assert res.downloaded == 1
    assert res.skipped_duration == 1


def test_harvest_resumes_and_skips_seen(tmp_path, mock_yt):
    infos, downloads = mock_yt
    infos["cc-0"] = _fake_info("cc-0", channel_id="a")
    infos["cc-1"] = _fake_info("cc-1", channel_id="b")

    harvest(tmp_path, target=10, queries=["cc"])
    assert len(downloads) == 2

    downloads.clear()
    res = harvest(tmp_path, target=10, queries=["cc"])  # second run, same state
    assert res.downloaded == 0
    assert res.skipped_seen == 2
    assert downloads == []


def test_harvest_skips_and_cleans_when_no_file_lands(tmp_path, mock_yt, monkeypatch):
    infos, _ = mock_yt
    infos["cc-0"] = _fake_info("cc-0", channel_id="a")
    infos["cc-1"] = _fake_info("cc-1", channel_id="b")

    # Simulate yt-dlp exiting 0 but leaving only a .part (no final mp4).
    def bad_download(url, out_path, start, length, *, max_height=720):
        from pathlib import Path
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.parent / (p.name + ".part")).write_bytes(b"partial")

    monkeypatch.setattr(youtube, "download_section", bad_download)

    res = harvest(tmp_path, target=10, queries=["cc"])

    assert res.downloaded == 0
    # Nothing recorded, and the .part leftovers were cleaned up.
    assert not (tmp_path / "manifest.jsonl").exists()
    assert list((tmp_path / "clips").glob("*.part")) == []


def test_harvest_dry_run_records_nothing(tmp_path, mock_yt):
    infos, downloads = mock_yt
    infos["cc-0"] = _fake_info("cc-0", channel_id="a")
    infos["cc-1"] = _fake_info("cc-1", channel_id="b")

    res = harvest(tmp_path, target=10, queries=["cc"], dry_run=True)

    assert res.downloaded == 2      # counted as "would keep"
    assert downloads == []          # but nothing downloaded
    assert not (tmp_path / "manifest.jsonl").exists()
