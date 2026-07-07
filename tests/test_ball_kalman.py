"""Kalman ball-smoothing tests with synthetic flickery tracks."""

import math

from soccer_vision.tracking.ball_kalman import (
    KalmanBallFilter,
    smooth_ball_track,
    smooth_samples,
)


def _visible(t, x, y=300.0, conf=0.8):
    return {
        "frame": int(t * 5),
        "timestamp_s": round(t, 3),
        "visible": True,
        "pixel_x": float(x),
        "pixel_y": float(y),
        "confidence": conf,
    }


def _offscreen(t):
    return {
        "frame": int(t * 5),
        "timestamp_s": round(t, 3),
        "visible": False,
        "pixel_x": None,
        "pixel_y": None,
        "confidence": 0.0,
    }


def test_smooths_jitter_but_tracks_the_trend():
    # Ball marching right at ~200 px/s with ±15 px detector jitter.
    jitter = [10, -12, 8, -6, 14, -9, 5, -11, 7, -4]
    samples = [_visible(i * 0.2, 100 + 40 * i + jitter[i]) for i in range(10)]

    out = smooth_samples(samples)

    # Every visible sample keeps its raw detection and is flagged smoothed.
    for o, s in zip(out, samples):
        assert o["raw_pixel_x"] == s["pixel_x"]
        assert o["smoothed"] is True

    # Smoothed path is less jumpy than the raw one: sum of absolute
    # frame-to-frame deviations from the straight trend should shrink.
    def wobble(key):
        vals = [o[key] for o in out]
        trend = [100 + 40 * i for i in range(10)]
        return sum(abs(v - t) for v, t in zip(vals, trend))

    raw_wobble = sum(abs(s["pixel_x"] - (100 + 40 * i)) for i, s in enumerate(samples))
    assert wobble("pixel_x") < raw_wobble


def test_rejects_a_single_frame_teleport():
    # Ball sits near x=500; one frame the detector jumps to a jersey at x=1600.
    samples = [_visible(i * 0.2, 500 + (2 if i % 2 else -2)) for i in range(6)]
    samples[3] = _visible(3 * 0.2, 1600.0)  # flicker

    out = smooth_samples(samples)

    flick = out[3]
    assert flick.get("outlier") is True
    assert flick["raw_pixel_x"] == 1600.0
    # The smoothed position ignores the teleport and stays near the ball.
    assert abs(flick["pixel_x"] - 500.0) < 100.0
    # Neighbours are not dragged toward the outlier.
    assert abs(out[4]["pixel_x"] - 500.0) < 100.0


def test_relocks_after_a_real_move():
    # First the ball is near x=200, then it genuinely relocates to x=1500 and
    # stays there. The filter should re-acquire rather than fight forever.
    samples = [_visible(i * 0.2, 200.0) for i in range(4)]
    samples += [_visible((4 + i) * 0.2, 1500.0) for i in range(5)]

    out = smooth_samples(samples, reacquire_after=3)

    # By the tail the smoothed estimate has caught up to the new location.
    assert abs(out[-1]["pixel_x"] - 1500.0) < 100.0


def test_offscreen_samples_pass_through_untouched():
    samples = [_visible(0.0, 500.0), _offscreen(0.2), _visible(0.4, 505.0)]

    out = smooth_samples(samples)

    assert out[1] == _offscreen(0.2)  # unchanged, still marks a dead gap
    assert "raw_pixel_x" not in out[1]


def test_long_gap_reinitialises_instead_of_gating_out_reappearance():
    # Ball at x=300, disappears for 2 s, reappears far away at x=1400.
    samples = [_visible(0.0, 300.0), _visible(0.2, 305.0)]
    samples += [_offscreen(0.2 + 0.2 * i) for i in range(1, 11)]  # ~2 s offscreen
    samples.append(_visible(2.4, 1400.0))
    samples.append(_visible(2.6, 1405.0))

    out = smooth_samples(samples, reacquire_gap_s=1.0)

    reappearance = out[-2]
    # It re-initialised on reappearance rather than rejecting it as an outlier.
    assert reappearance.get("outlier") is None
    assert abs(reappearance["pixel_x"] - 1400.0) < 50.0


def test_smooth_ball_track_preserves_metadata():
    track = {
        "video": "m.mp4",
        "fps": 30.0,
        "sample_fps": 5.0,
        "width": 1920,
        "height": 1080,
        "total_frames": 900,
        "samples": [_visible(i * 0.2, 500 + i) for i in range(4)],
    }

    out = smooth_ball_track(track)

    assert out["smoothed"] is True
    assert out["fps"] == 30.0 and out["width"] == 1920
    assert out is not track and out["samples"] is not track["samples"]
    assert len(out["samples"]) == 4


def test_filter_starts_uninitialised_and_seeds_on_first_measurement():
    kf = KalmanBallFilter()
    assert not kf.initialised
    est, accepted = kf.step(0.0, None)
    assert est is None and accepted is False
    est, accepted = kf.step(0.2, (400.0, 250.0))
    assert accepted is True
    assert math.isclose(est[0], 400.0) and math.isclose(est[1], 250.0)
