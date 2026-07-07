"""Model-free helpers of the jersey recognizer: crop, legibility, digit parse."""

import numpy as np

from soccer_vision.identify.jersey_ocr import (
    crop_number_region,
    is_legible,
    parse_number,
)


def test_parse_single_and_double_digit():
    assert parse_number("6").number == 6
    assert parse_number("23").number == 23


def test_parse_confidence_is_min_over_digits():
    r = parse_number("23", [0.9, 0.4])
    assert r.number == 23
    assert abs(r.confidence - 0.4) < 1e-9


def test_parse_rejects_non_numeric_and_too_long():
    assert parse_number("").number is None
    assert parse_number("ABC").number is None
    assert parse_number("123").number is None  # 3+ digits: not a jersey


def test_parse_strips_stray_letters():
    assert parse_number("A7").number == 7


def test_crop_region_rejects_tiny_box():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert crop_number_region(frame, (10, 10, 18, 20)) is None


def test_crop_region_takes_upper_torso():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = crop_number_region(frame, (50, 40, 90, 140))  # w=40, h=100
    assert crop is not None
    # top 55% of a 100px-tall box → ~55px tall.
    assert 45 <= crop.shape[0] <= 60


def test_legibility_gate():
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert not is_legible(flat)              # no contrast
    noisy = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    assert is_legible(noisy)                 # plenty of contrast
    assert not is_legible(noisy[:8, :8])     # too small
