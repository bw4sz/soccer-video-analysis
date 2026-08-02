"""`predict_split` must be exactly the two calls it replaces, not an approximation.

The speedup is only safe if a single forward at the lower threshold, split
afterwards, returns what `predict()` at the player threshold and `predict_ball()`
at the ball threshold returned between them. If it ever drifts, every run's
detections change silently.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from soccer_vision.detection.ball import ball_position_from
from soccer_vision.detection.rfdetr import (
    ALL_PERSON_CLASS_IDS,
    RFDETRSoccerDetector,
)


class FakeModel:
    """Scores don't depend on the threshold they were requested at — only the
    cutoff does. That is the property `predict_split` relies on."""

    def __init__(self, dets: sv.Detections):
        self.dets = dets
        self.calls = 0

    def predict(self, image, threshold: float):
        self.calls += 1
        keep = self.dets.confidence >= threshold
        return self.dets[keep]


def _detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(
            [
                [0.0, 0.0, 10.0, 20.0],    # player, 0.90
                [20.0, 0.0, 30.0, 20.0],   # player, 0.25  (below 0.3)
                [40.0, 0.0, 44.0, 4.0],    # ball,   0.55
                [60.0, 0.0, 64.0, 4.0],    # ball,   0.24  (below 0.3, above 0.2)
                [80.0, 0.0, 84.0, 4.0],    # ball,   0.10  (below both)
                [90.0, 0.0, 99.0, 20.0],   # referee, 0.40
            ],
            dtype=np.float32,
        ),
        confidence=np.array([0.90, 0.25, 0.55, 0.24, 0.10, 0.40], dtype=np.float32),
        class_id=np.array([1, 1, 0, 0, 0, 2], dtype=int),
    )


def _detector() -> tuple[RFDETRSoccerDetector, FakeModel]:
    model = FakeModel(_detections())
    return RFDETRSoccerDetector(model, device="cpu", conf_threshold=0.3), model


def _frame() -> np.ndarray:
    return np.zeros((32, 100, 3), dtype=np.uint8)


def test_split_matches_the_two_calls_it_replaces():
    det, model = _detector()
    frame = _frame()

    mixed = det.predict(frame, conf_threshold=0.3)
    want_people = mixed[np.isin(mixed.class_id, list(ALL_PERSON_CLASS_IDS))]
    want_ball = det.predict_ball(frame, conf_threshold=0.2)
    model.calls = 0

    people, ball = det.predict_split(frame, player_conf=0.3, ball_conf=0.2)

    assert model.calls == 1, "the whole point is one forward, not two"
    np.testing.assert_array_equal(people.xyxy, want_people.xyxy)
    np.testing.assert_array_equal(people.class_id, want_people.class_id)
    np.testing.assert_array_equal(ball.xyxy, want_ball.xyxy)
    np.testing.assert_array_equal(ball.confidence, want_ball.confidence)


def test_referees_count_as_people_and_the_ball_keeps_its_looser_floor():
    det, _ = _detector()
    people, ball = det.predict_split(_frame(), player_conf=0.3, ball_conf=0.2)

    assert sorted(people.class_id) == [1, 2], "player + referee, not the 0.25 player"
    # 0.55 and 0.24 clear the ball floor; 0.10 does not.
    np.testing.assert_allclose(sorted(ball.confidence), [0.24, 0.55], rtol=1e-6)


def test_ball_position_picks_the_most_confident_and_returns_its_centre():
    det, _ = _detector()
    _, ball = det.predict_split(_frame(), player_conf=0.3, ball_conf=0.2)

    cx, cy, conf = ball_position_from(ball)
    assert (cx, cy) == (42.0, 2.0)
    assert conf == np.float32(0.55)


def test_ball_position_is_none_when_nothing_clears_the_floor():
    det, _ = _detector()
    _, ball = det.predict_split(_frame(), player_conf=0.3, ball_conf=0.99)
    assert ball_position_from(ball) is None
    assert ball_position_from(sv.Detections.empty()) is None


def test_empty_detections_split_cleanly():
    model = FakeModel(_detections())
    det = RFDETRSoccerDetector(model, device="cpu", conf_threshold=0.99)
    people, ball = det.predict_split(_frame(), player_conf=0.99, ball_conf=0.99)
    assert len(people) == 0 and len(ball) == 0
