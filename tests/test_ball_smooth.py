"""Robust ball-smoothing tests: does the median gate cut flicker, not the ball?"""

import numpy as np

from soccer_vision.tracking.ball_smooth import (
    gate_outliers,
    smooth_ball_track,
    smooth_samples,
)

FPS = 30.0


def _visible(i, x, y=500.0, conf=0.6, fps=FPS):
    return {
        "frame": i,
        "timestamp_s": round(i / fps, 4),
        "visible": True,
        "pixel_x": float(x),
        "pixel_y": float(y),
        "confidence": conf,
    }


def _offscreen(i, fps=FPS):
    return {
        "frame": i,
        "timestamp_s": round(i / fps, 4),
        "visible": False,
        "pixel_x": None,
        "pixel_y": None,
        "confidence": 0.0,
    }


def _smooth_run(n=120, x0=400.0, vx=6.0, y=500.0, fps=FPS):
    """A ball moving steadily right at vx px/frame -- physically plausible."""
    return [_visible(i, x0 + vx * i, y, fps=fps) for i in range(n)]


def test_clean_track_is_left_alone():
    samples = _smooth_run()
    out = smooth_samples(samples)

    assert all(o["visible"] for o in out)
    assert not any(o.get("outlier") for o in out)
    for o, s in zip(out, samples):
        assert o["pixel_x"] == s["pixel_x"]
        assert o["pixel_y"] == s["pixel_y"]


def test_single_frame_teleport_is_rejected_and_filled():
    samples = _smooth_run()
    truth_x = samples[60]["pixel_x"]
    samples[60] = _visible(60, 1850.0, 90.0)      # flicker onto the far crowd

    out = smooth_samples(samples)

    assert out[60]["outlier"] is True
    assert out[60]["interpolated"] is True
    assert out[60]["raw_pixel_x"] == 1850.0       # the raw detection is preserved
    # Filled from the neighbours, so it lands back on the real trajectory.
    assert abs(out[60]["pixel_x"] - truth_x) < 5.0
    assert out[60]["visible"] is True


def test_multi_frame_excursion_is_rejected():
    """The case the causal filter gets wrong -- it re-locks after 3 rejects."""
    samples = _smooth_run()
    for i in (60, 61, 62, 63):
        samples[i] = _visible(i, 1900.0, 80.0)

    out = smooth_samples(samples)

    assert all(out[i].get("outlier") for i in (60, 61, 62, 63))
    # And the run either side is untouched.
    assert not any(out[i].get("outlier") for i in range(40, 60))
    assert not any(out[i].get("outlier") for i in range(64, 90))


def test_genuine_motion_is_not_cut():
    """A ball crossing the frame at a plausible speed must survive."""
    samples = _smooth_run(vx=20.0, x0=100.0)      # 20 px/frame = ~600 px/s
    out = smooth_samples(samples)

    kept = sum(1 for o in out if o["visible"] and not o.get("outlier"))
    assert kept >= 0.9 * len(samples)


def test_long_gap_is_left_absent_not_invented():
    """Inventing a position across a long gap would erase real dead time."""
    samples = _smooth_run(n=60) + [_offscreen(i) for i in range(60, 120)] + [
        _visible(i, 400.0 + 6.0 * (i - 120), 500.0) for i in range(120, 180)
    ]
    out = smooth_samples(samples, max_fill_s=0.5)

    # The 2 s hole is far longer than max_fill_s, so nothing is fabricated.
    assert not any(out[i]["visible"] for i in range(75, 105))
    assert all(out[i].get("pixel_x") is None for i in range(75, 105))


def test_short_gap_is_filled():
    samples = _smooth_run()
    for i in range(60, 66):                        # 0.2 s hole
        samples[i] = _offscreen(i)

    out = smooth_samples(samples, max_fill_s=0.5)

    assert all(out[i]["visible"] for i in range(60, 66))
    assert all(out[i]["interpolated"] for i in range(60, 66))
    assert abs(out[62]["pixel_x"] - (400.0 + 6.0 * 62)) < 5.0


def test_window_is_seconds_so_the_gate_survives_a_low_sample_rate():
    """Same play, sampled at 5 fps: the gate must still find the flicker."""
    samples = _smooth_run(n=40, vx=36.0, fps=5.0)  # same px/s as 6 px/frame @30
    samples[20] = _visible(20, 1880.0, 70.0, fps=5.0)

    keep = gate_outliers(samples)

    assert not keep[20]
    assert keep.sum() >= 0.85 * len(samples)


def test_iteration_tightens_the_gate():
    """Outliers contaminate the first median; letting survivors re-vote helps."""
    samples = _smooth_run()
    for i in range(55, 62):                        # a heavy contiguous burst
        samples[i] = _visible(i, 1900.0, 80.0)

    once = gate_outliers(samples, iterations=1)
    thrice = gate_outliers(samples, iterations=3)

    assert thrice.sum() <= once.sum()
    assert not any(thrice[i] for i in range(55, 62))


def test_no_unphysical_steps_survive():
    rng = np.random.default_rng(0)
    samples = _smooth_run(n=300)
    for i in rng.choice(np.arange(10, 290), size=45, replace=False):
        samples[int(i)] = _visible(int(i), rng.uniform(0, 1920), rng.uniform(0, 1080))

    out = smooth_samples(samples)
    pts = [(o["timestamp_s"], o["pixel_x"], o["pixel_y"]) for o in out
           if o["visible"] and o["pixel_x"] is not None]
    a = np.array(pts)
    step = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2]))
    adjacent = np.diff(a[:, 0]) < 1.5 / FPS

    # A 30 m/s ball is ~35 px/frame on this framing; nothing should exceed 70.
    assert (step[adjacent] > 70).mean() < 0.02


def test_track_wrapper_preserves_metadata():
    track = {"video": "m.mp4", "fps": FPS, "sample_fps": FPS,
             "width": 1920, "height": 1080, "samples": _smooth_run(n=60)}
    out = smooth_ball_track(track)

    assert out["smoothed"] is True
    assert out["video"] == "m.mp4"
    assert out["width"] == 1920
    assert len(out["samples"]) == 60
    assert track["samples"][0]["pixel_x"] == 400.0   # input not mutated


def test_empty_and_tiny_tracks_do_not_crash():
    assert smooth_samples([]) == []
    assert len(smooth_samples([_visible(0, 100.0)])) == 1
    assert len(smooth_samples([_offscreen(0), _offscreen(1)])) == 2
