"""Virtual broadcast crop window smoothing tests."""

from soccer_vision.broadcast.virtual_cam import (
    BroadcastConfig,
    _compute_crop_rect,
    _smooth_position,
)


def test_smooth_position_converges():
    current = (100.0, 100.0)
    target = (200.0, 200.0)
    alpha = 0.5

    result = _smooth_position(current, target, alpha)
    assert result == (150.0, 150.0)

    result2 = _smooth_position(result, target, alpha)
    assert result2 == (175.0, 175.0)


def test_smooth_position_alpha_zero():
    current = (100.0, 100.0)
    target = (200.0, 200.0)
    result = _smooth_position(current, target, 0.0)
    assert result == (100.0, 100.0)


def test_smooth_position_alpha_one():
    current = (100.0, 100.0)
    target = (200.0, 200.0)
    result = _smooth_position(current, target, 1.0)
    assert result == (200.0, 200.0)


def test_crop_rect_centered():
    x, y, w, h = _compute_crop_rect(500, 300, 400, 200, 1000, 600)
    assert x == 300
    assert y == 200
    assert w == 400
    assert h == 200


def test_crop_rect_clamped_left():
    x, y, w, h = _compute_crop_rect(50, 300, 400, 200, 1000, 600)
    assert x == 0  # clamped to left edge


def test_crop_rect_clamped_right():
    x, y, w, h = _compute_crop_rect(950, 300, 400, 200, 1000, 600)
    assert x == 600  # clamped to right edge (1000 - 400)


def test_broadcast_config_defaults():
    config = BroadcastConfig()
    assert config.output_width == 1920
    assert config.output_height == 1080
    assert config.detect_fps == 5.0
