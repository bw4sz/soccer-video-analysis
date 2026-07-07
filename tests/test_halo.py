"""Player halo annotation on clips."""

import json

import cv2
import numpy as np
import pytest

from soccer_vision.cli.extract import _load_halo
from soccer_vision.clips.extract import extract_event_clips
from soccer_vision.clips.halo import (
    interpolate_bbox,
    load_track_boxes,
    render_halo_clip,
)

FIXTURE = "tests/fixtures/clip_10s_a.mp4"


def _samples():
    return [
        (0, np.array([100, 100, 140, 200], dtype=np.float32)),
        (5, np.array([110, 100, 150, 200], dtype=np.float32)),
        (30, np.array([300, 100, 340, 200], dtype=np.float32)),  # big gap after 5
    ]


def test_interpolate_exact_sample():
    s = _samples()
    assert np.allclose(interpolate_bbox(s, 5, max_gap_frames=10), [110, 100, 150, 200])


def test_interpolate_midpoint():
    s = _samples()
    # frame 2 sits 2/5 of the way from sample 0 (x1=100) to sample 5 (x1=110)
    got = interpolate_bbox(s, 2, max_gap_frames=10)
    assert got is not None
    assert abs(got[0] - 104.0) < 0.5  # 100 + (2/5)*10


def test_interpolate_outside_span_returns_none():
    s = _samples()
    assert interpolate_bbox(s, -1, max_gap_frames=10) is None
    assert interpolate_bbox(s, 40, max_gap_frames=10) is None


def test_interpolate_gap_too_large_returns_none():
    s = _samples()
    # frame 20 sits in the 5→30 gap (25 frames) which exceeds max_gap 10
    assert interpolate_bbox(s, 20, max_gap_frames=10) is None


def test_load_track_boxes_roundtrip(tmp_path):
    doc = {
        "video": "proxy.mp4", "fps": 30.0, "sample_interval": 6,
        "tracks": {"7": [{"frame": 0, "bbox": [1, 2, 3, 4]},
                         {"frame": 6, "bbox": [5, 6, 7, 8]}]},
    }
    p = tmp_path / "tracks.json"
    p.write_text(json.dumps(doc))
    boxes = load_track_boxes(p)
    assert set(boxes) == {7}
    assert len(boxes[7]) == 2
    assert boxes[7][0][0] == 0
    assert np.allclose(boxes[7][1][1], [5, 6, 7, 8])


@pytest.mark.parametrize("style", ["ellipse", "circle"])
def test_render_halo_clip_writes_valid_video(tmp_path, style):
    # Track box that stays centred through the whole 2s window at 30fps.
    samples = [(f, np.array([400, 200, 480, 400], dtype=np.float32)) for f in range(0, 90, 5)]
    out = tmp_path / f"halo_{style}.mp4"
    render_halo_clip(
        FIXTURE, out, start_s=0.0, duration_s=2.0,
        track_samples=samples, style=style,
    )
    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    cap.release()


def test_render_halo_rejects_unknown_style(tmp_path):
    with pytest.raises(ValueError):
        render_halo_clip(FIXTURE, tmp_path / "x.mp4", start_s=0, duration_s=0.1,
                         track_samples=[], style="glow")


def test_extract_event_clips_halo_falls_back_without_track(tmp_path, monkeypatch):
    # Event has no track_id → plain cut (ffmpeg) even though halo_tracks is given.
    calls = {"ffmpeg": 0}
    monkeypatch.setattr(
        "soccer_vision.clips.extract.ffmpeg_extract_clip",
        lambda *a, **k: calls.__setitem__("ffmpeg", calls["ffmpeg"] + 1),
    )
    events = [{"timestamp_s": 1.0, "label": "goal_kick"}]
    paths = extract_event_clips(
        FIXTURE, events, tmp_path, pre_s=0.5, post_s=0.5,
        halo_tracks={7: [(0, np.array([1, 2, 3, 4], dtype=np.float32))]},
    )
    assert len(paths) == 1
    assert calls["ffmpeg"] == 1  # took the plain-cut branch, not the halo renderer


def test_extract_event_clips_halo_renders_matching_track(tmp_path, monkeypatch):
    # Event with a track_id present in halo_tracks → halo renderer, not ffmpeg.
    calls = {"ffmpeg": 0, "halo": 0}
    monkeypatch.setattr(
        "soccer_vision.clips.extract.ffmpeg_extract_clip",
        lambda *a, **k: calls.__setitem__("ffmpeg", calls["ffmpeg"] + 1),
    )
    monkeypatch.setattr(
        "soccer_vision.clips.halo.render_halo_clip",
        lambda *a, **k: calls.__setitem__("halo", calls["halo"] + 1),
    )
    events = [{"timestamp_s": 1.0, "label": "pass", "track_id": 7}]
    extract_event_clips(
        FIXTURE, events, tmp_path, pre_s=0.5, post_s=0.5,
        halo_tracks={7: [(0, np.array([1, 2, 3, 4], dtype=np.float32))]},
    )
    assert calls == {"ffmpeg": 0, "halo": 1}


def test_load_halo_disabled_returns_none(tmp_path):
    assert _load_halo(tmp_path, None) == (None, None, 0)


def test_load_halo_missing_tracks_warns_and_disables(tmp_path, capsys):
    got = _load_halo(tmp_path, "ellipse")
    assert got == (None, None, 0)
    assert "missing" in capsys.readouterr().out


def test_load_halo_reads_boxes_and_gap(tmp_path):
    (tmp_path / "tracks.json").write_text(json.dumps({
        "sample_interval": 6,
        "tracks": {"3": [{"frame": 0, "bbox": [1, 2, 3, 4]}]},
    }))
    boxes, style, max_gap = _load_halo(tmp_path, "circle")
    assert style == "circle"
    assert max_gap == 24  # sample_interval * 4
    assert set(boxes) == {3}
