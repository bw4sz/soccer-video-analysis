"""The two offsets and the interpolation are the whole risk in this parser.

A box landing one frame or one window out is invisible in the export and wrong in
every number computed from it, so the frame arithmetic is pinned here rather than
trusted to a reading of the format.
"""
import numpy as np

from soccer_vision.annotate.hand_tracking import boxes_from_export

MANIFEST = {"windows": [{"window": 1, "start_frame": 71928}]}


def _kf(frame, x, y, enabled=True):
    return {"frame": frame, "x": x, "y": y, "width": 10.0, "height": 20.0,
            "enabled": enabled, "rotation": 0}


def _task(sequence, label="Morgan", start_frame=71928):
    return {
        "data": {"window": 1, "start_frame": start_frame},
        "annotations": [{"result": [{
            "type": "videorectangle", "from_name": "box", "to_name": "video",
            "value": {"labels": [label], "sequence": sequence},
        }]}],
    }


def test_clip_frame_one_is_the_window_start_frame():
    """Label Studio counts from 1, our tracks from 0, and both meet here."""
    boxes, _ = boxes_from_export([_task([_kf(1, 0.0, 0.0)])], MANIFEST,
                                 frame_w=1920, frame_h=1080)
    assert [f for f, _, _ in boxes] == [71928]


def test_percentages_become_pixels():
    boxes, _ = boxes_from_export([_task([_kf(1, 25.0, 50.0)])], MANIFEST,
                                 frame_w=1920, frame_h=1080)
    _, bbox, _ = boxes[0]
    assert np.allclose(bbox, [480, 540, 672, 756])


def test_between_keyframes_is_interpolated_not_held():
    """A consumer reading keyframes only sees a player teleporting once a second."""
    boxes, _ = boxes_from_export([_task([_kf(1, 0.0, 0.0), _kf(11, 10.0, 0.0)])],
                                 MANIFEST, frame_w=1000, frame_h=1000)
    assert len(boxes) == 11
    xs = [bbox[0] for _, bbox, _ in boxes]
    assert xs[0] == 0.0 and xs[-1] == 100.0
    assert np.allclose(np.diff(xs), 10.0)


def test_a_disabled_keyframe_ends_the_span():
    """The player left the shot; no box glides across the gap she was absent for."""
    seq = [_kf(1, 0.0, 0.0), _kf(3, 2.0, 0.0, enabled=False), _kf(9, 50.0, 0.0)]
    boxes, _ = boxes_from_export([_task(seq)], MANIFEST,
                                 frame_w=1000, frame_h=1000)
    frames = [f - 71928 + 1 for f, _, _ in boxes]
    assert 4 not in frames and 8 not in frames
    assert frames[:3] == [1, 2, 3]


def test_unsure_is_excluded_unless_asked_for():
    boxes, summary = boxes_from_export([_task([_kf(1, 0.0, 0.0)], label="unsure")],
                                       MANIFEST, frame_w=1920, frame_h=1080)
    assert boxes == [] and summary["unsure_skipped"] == 1

    kept, _ = boxes_from_export([_task([_kf(1, 0.0, 0.0)], label="unsure")],
                                MANIFEST, frame_w=1920, frame_h=1080,
                                keep_unsure=True)
    assert len(kept) == 1


def test_task_start_frame_beats_the_manifest():
    """An export can outlive a re-render; the task carries its own truth."""
    boxes, _ = boxes_from_export([_task([_kf(1, 0.0, 0.0)], start_frame=999)],
                                 MANIFEST, frame_w=1920, frame_h=1080)
    assert [f for f, _, _ in boxes] == [999]
