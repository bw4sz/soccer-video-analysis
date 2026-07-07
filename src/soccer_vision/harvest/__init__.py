"""Harvest short, openly-licensed youth-soccer clips from YouTube.

The goal is a *diverse* annotation set — many different games, cameras,
countries, and kit colours — not depth on any one team or camera. We therefore
pull a single short clip from the middle of each match, cap how many clips come
from any one channel, and only keep videos the uploader released under a
Creative Commons Attribution (CC BY) licence.

CC BY *requires* crediting the creator, so every downloaded clip is recorded in
a provenance manifest (:mod:`soccer_vision.harvest.manifest`) that also powers
dedup/resume across runs.
"""

from soccer_vision.harvest.manifest import ClipRecord, Manifest
from soccer_vision.harvest.youtube import (
    HarvestResult,
    harvest,
    is_creative_commons,
    midpoint_window,
)

__all__ = [
    "ClipRecord",
    "Manifest",
    "HarvestResult",
    "harvest",
    "is_creative_commons",
    "midpoint_window",
]
