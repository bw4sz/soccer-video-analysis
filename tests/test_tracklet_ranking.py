"""Lanes are ranked by how far they moved, not by how long they stayed in shot.

The distinction only exists because the camera pans: a spectator standing on the
touchline sweeps across the frame as fast as a player runs, so the old ranking
and a naive motion ranking both put the crowd on page 1. These fixtures are built
with a pan applied to every box so that a test passing means the pan was actually
removed, not that motion happened to correlate with something.
"""
import numpy as np

from soccer_vision.annotate.tracklets import (
    camera_pan,
    choose_windows,
    motion_score,
)

FPS = 30.0
PAN_PER_FRAME = 8.0  # px/frame, well above this camera's real p99


def _lane(x0, y0, *, n=90, dx=0.0, dy=0.0, start=0, h=60.0):
    """A lane drifting (dx, dy) px/frame of its own, plus the shared camera pan."""
    out = []
    for i in range(n):
        f = start + i
        x = x0 + dx * i + PAN_PER_FRAME * f
        y = y0 + dy * i
        out.append((f, np.array([x, y, x + h / 2, y + h], dtype=float)))
    return out


def _padding(samples, n=120):
    """Extra movers, so the pan estimate clears MIN_PAN_BOXES."""
    for i in range(3, 9):
        samples[i] = _lane(100 * i, 500, n=n, dx=(i - 5.5) * 2.0)
    return samples


def test_pan_is_recovered_from_the_boxes():
    """Everyone sharing a displacement is the camera, whatever else they do."""
    present = [(90, tid, _lane(100 * i, 500, dx=(i - 2.5) * 3.0))
               for i, tid in enumerate([1, 2, 3, 4, 5, 6])]
    pan = camera_pan(present)
    assert np.allclose([pan[f][0] for f in range(1, 90)], PAN_PER_FRAME, atol=1e-6)


def test_a_still_person_scores_near_zero_despite_the_pan():
    present = [(90, tid, _lane(100 * i, 500, dx=(i - 2.5) * 3.0))
               for i, tid in enumerate([1, 2, 3, 4, 5, 6])]
    pan = camera_pan(present)
    standing = _lane(900, 500)  # no motion of its own, only the camera's
    assert motion_score(standing, pan) < 0.1


def test_a_runner_outscores_a_stander_by_an_order_of_magnitude():
    present = [(90, tid, _lane(100 * i, 500, dx=(i - 2.5) * 3.0))
               for i, tid in enumerate([1, 2, 3, 4, 5, 6])]
    pan = camera_pan(present)
    assert motion_score(_lane(900, 500, dx=6.0), pan) > 5.0
    assert motion_score(_lane(900, 500), pan) < 0.1


def test_the_runner_gets_slot_one_even_when_the_stander_is_on_screen_longer():
    """The whole point: length would rank these the other way round."""
    samples = _padding({
        # On screen for the entire window, never moves: the touchline case.
        10: _lane(1500, 480, n=120),
        # Half the window, running: the footballer case.
        20: _lane(300, 700, n=60, dx=7.0, start=30),
    })

    by_motion = choose_windows(
        samples, fps=FPS, window_s=4.0, n_windows=1, min_track_frames=30,
        max_lanes=8, start_s=0.0)
    by_length = choose_windows(
        samples, fps=FPS, window_s=4.0, n_windows=1, min_track_frames=30,
        max_lanes=8, start_s=0.0, rank="length")

    moved = {lane["track_id"]: lane["moved_body_heights"]
             for lane in by_motion[0]["lanes"]}
    # An order of magnitude apart, not zero-versus-something: the stander keeps a
    # small residual because the pan is a median over whoever is on screen, and
    # eight boxes with a runner among them do not cancel exactly. A real window
    # carries thirty to fifty.
    assert moved[20] > 5 * moved[10]

    # Ranking decides the pages, not the slot numbers, which stay chronological.
    ranked = lambda ws: [lane["track_id"] for lane in ws[0]["lanes"]]
    assert 20 in ranked(by_motion)
    assert by_length[0]["lanes"] != by_motion[0]["lanes"] or len(by_length) == 1


def test_the_stander_sinks_to_a_later_page():
    """With one lane per page, ranking is visible as page order."""
    samples = _padding({
        10: _lane(1500, 480, n=120),
        20: _lane(300, 700, n=120, dx=7.0),
    })

    windows = choose_windows(
        samples, fps=FPS, window_s=4.0, n_windows=1, min_track_frames=30,
        max_lanes=1, all_lanes=True, start_s=0.0)
    order = [w["lanes"][0]["track_id"] for w in windows]
    assert order.index(20) < order.index(10)


def test_length_ranking_is_still_available_and_still_prefers_the_stander():
    samples = _padding({
        10: _lane(1500, 480, n=120),
        20: _lane(300, 700, n=60, dx=7.0, start=30),
    }, n=90)

    windows = choose_windows(
        samples, fps=FPS, window_s=4.0, n_windows=1, min_track_frames=30,
        max_lanes=1, all_lanes=True, start_s=0.0, rank="length")
    order = [w["lanes"][0]["track_id"] for w in windows]
    assert order.index(10) < order.index(20)
