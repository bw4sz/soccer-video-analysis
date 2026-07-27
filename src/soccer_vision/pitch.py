"""Nominal pitch dimensions for a youth 7v7 field.

These are *declared* dimensions, not measured ones. Nothing in the pipeline
projects pixels into this coordinate system: field registration was removed
after both estimators failed outright on Veo footage (Hough locked onto
apartment rooftops and stadium walls; DeepLabv3 sn-calib emitted diffuse noise
— 0/6 frames each, and the venue paints blue/red/white lines from several
overlapping pitches, so even a perfect line detector cannot say which touchline
is *the* touchline).

These constants therefore have exactly one honest use: declaring
``field_dimensions`` in the OSL export, where the schema wants a nominal pitch
size rather than a measurement. There is deliberately nothing else — the metric
helpers that used to default to these bounds (``metrics/heatmap.py``,
``metrics/shots.py::is_shot_toward_goal``, and the distance figure in
``stats.json``) were deleted rather than left unreachable, so that importing this
module is never mistaken for having a working pitch coordinate system.

Anything that needs to reason about *where on the pitch* something happened works
in pixel space instead — see :mod:`soccer_vision.events.on_ball` and
:mod:`soccer_vision.detection.field_filter`.
"""

from __future__ import annotations

FIELD_W_M = 55.0  # touchline to touchline (7v7)
FIELD_H_M = 36.0  # goal line to goal line
