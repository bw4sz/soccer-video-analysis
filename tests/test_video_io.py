"""Integration tests using real video fixtures (10s clips)."""

from pathlib import Path

import pytest

from soccer_vision.io.video import VideoReader

FIXTURES = Path(__file__).parent / "fixtures"
CLIP_A = FIXTURES / "clip_10s_a.mp4"
CLIP_B = FIXTURES / "clip_10s_b.mp4"

pytestmark = pytest.mark.skipif(
    not CLIP_A.exists(), reason="test fixtures not available"
)


def test_video_reader_metadata():
    with VideoReader(CLIP_A) as reader:
        assert reader.width == 960
        assert reader.height == 540
        assert 29.0 < reader.fps < 31.0
        assert 9.0 < reader.duration_s < 11.0
        assert reader.total_frames > 200


def test_video_reader_read_frame():
    with VideoReader(CLIP_A) as reader:
        frame = reader.read_frame(0)
        assert frame is not None
        assert frame.shape == (540, 960, 3)

        mid = reader.total_frames // 2
        frame_mid = reader.read_frame(mid)
        assert frame_mid is not None


def test_video_reader_sample_frames():
    with VideoReader(CLIP_B) as reader:
        frames = list(reader.sample_frames(interval=30))
        assert len(frames) >= 8
        for fn, frame in frames:
            assert frame.shape == (540, 960, 3)


def test_video_reader_nonexistent():
    with pytest.raises(FileNotFoundError):
        VideoReader("/nonexistent/video.mp4")
