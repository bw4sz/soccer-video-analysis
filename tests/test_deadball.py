"""Dead-time detection tests with synthetic ball tracks."""

from soccer_vision.events.deadball import (
    OFFSCREEN,
    STATIONARY,
    classify_samples,
    find_removed_segments,
    invert_segments,
    plan_trim,
)


def _moving(t, x0=100.0, speed=200.0):
    """A visible, clearly-moving sample at time t."""
    return {
        "frame": int(t * 5),
        "timestamp_s": round(t, 3),
        "visible": True,
        "pixel_x": x0 + speed * t,
        "pixel_y": 300.0,
        "confidence": 0.8,
    }


def _still(t, x=500.0):
    """A visible sample parked at a fixed position."""
    return {
        "frame": int(t * 5),
        "timestamp_s": round(t, 3),
        "visible": True,
        "pixel_x": x,
        "pixel_y": 300.0,
        "confidence": 0.8,
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


def _track(samples):
    return samples


def test_classify_offscreen():
    samples = [_offscreen(i * 0.2) for i in range(5)]
    labels = classify_samples(samples)
    assert all(lbl == OFFSCREEN for lbl in labels)


def test_classify_stationary_after_window():
    # 5 fps, parked the whole time → later samples flagged stationary once the
    # 1s smoothing window is full.
    samples = [_still(i * 0.2) for i in range(20)]
    labels = classify_samples(samples, stationary_window_s=1.0)
    assert labels[0] is None  # too early to judge
    assert labels[-1] == STATIONARY


def test_classify_moving_is_live():
    samples = [_moving(i * 0.2) for i in range(20)]
    labels = classify_samples(samples)
    assert all(lbl is None for lbl in labels)


def test_removed_offscreen_run_over_threshold():
    # 3s moving, 8s offscreen, 3s moving.
    samples = []
    t = 0.0
    while t < 3.0:
        samples.append(_moving(t))
        t += 0.2
    while t < 11.0:
        samples.append(_offscreen(t))
        t += 0.2
    while t < 14.0:
        samples.append(_moving(t, x0=5000.0))
        t += 0.2

    removed = find_removed_segments(samples, total_duration_s=14.0, min_dead_s=5.0, pad_s=0.5)
    assert len(removed) == 1
    seg = removed[0]
    assert seg["reason"] == OFFSCREEN
    # ~3s..11s dead, shrunk 0.5s each side.
    assert seg["start_s"] > 3.0
    assert seg["end_s"] < 11.0
    assert seg["duration_s"] > 5.0


def test_short_dead_run_kept():
    # Only 2s offscreen → below the 5s threshold, nothing removed.
    samples = []
    t = 0.0
    while t < 3.0:
        samples.append(_moving(t))
        t += 0.2
    while t < 5.0:
        samples.append(_offscreen(t))
        t += 0.2
    while t < 8.0:
        samples.append(_moving(t, x0=5000.0))
        t += 0.2

    removed = find_removed_segments(samples, total_duration_s=8.0, min_dead_s=5.0)
    assert removed == []


def test_invert_segments_complement():
    removed = [{"start_s": 3.0, "end_s": 10.0, "duration_s": 7.0, "reason": OFFSCREEN}]
    keep = invert_segments(removed, total_duration_s=14.0)
    assert keep == [
        {"start_s": 0.0, "end_s": 3.0, "duration_s": 3.0},
        {"start_s": 10.0, "end_s": 14.0, "duration_s": 4.0},
    ]


def test_invert_no_removed_keeps_all():
    keep = invert_segments([], total_duration_s=10.0)
    assert keep == [{"start_s": 0.0, "end_s": 10.0, "duration_s": 10.0}]


def test_plan_trim_bookkeeping():
    samples = []
    t = 0.0
    while t < 3.0:
        samples.append(_moving(t))
        t += 0.2
    while t < 11.0:
        samples.append(_offscreen(t))
        t += 0.2
    while t < 14.0:
        samples.append(_moving(t, x0=5000.0))
        t += 0.2

    plan = plan_trim(samples, 14.0, min_dead_s=5.0, pad_s=0.5)
    assert plan["source_duration_s"] == 14.0
    # kept + removed reconstruct the source.
    assert abs(plan["kept_duration_s"] + plan["removed_duration_s"] - 14.0) < 1e-6
    assert plan["removed_duration_s"] > 5.0
    assert len(plan["keep_segments"]) == 2


def test_stationary_ball_is_trimmed():
    # Ball visible but parked for 8s between two moving spans.
    samples = []
    t = 0.0
    while t < 3.0:
        samples.append(_moving(t))
        t += 0.2
    while t < 11.0:
        samples.append(_still(t, x=800.0))
        t += 0.2
    while t < 14.0:
        samples.append(_moving(t, x0=5000.0))
        t += 0.2

    removed = find_removed_segments(samples, total_duration_s=14.0, min_dead_s=5.0)
    assert len(removed) == 1
    assert removed[0]["reason"] == STATIONARY
