"""Filter detections down to on-field players.

A person prompt finds people anywhere in frame — coaches, subs, parents on the
touchline — so detections need a notion of "on the field" before they become
tracks.

This used to project each foot-point through a homography and keep the ones
landing inside the pitch rectangle. That branch is gone with field registration
(see :mod:`soccer_vision.pitch`): it never worked on this footage, and it failed
*destructively* rather than visibly. Measured in job 37877533, ``compute_homography``
returned ``ok=True`` with a garbage matrix, surviving detections went 20 → 0 and
stayed there, leaving 52 track-frames where ~6000 were expected. The guard added
to catch that — fall back to the hull when the homography rejects essentially
everyone — meant the hull was doing the work in practice, with the homography
contributing only the risk of a bad-but-not-bad-enough matrix quietly dropping
real players.

So the geometric cut is now the only path. It is crude and it is the honest state
of the art here until turf-mask segmentation lands (issues #7/#20), which would
give a real field polygon without needing lines or metres.

**The cut is asymmetric, because the frame is.** It was a centred rectangle
keeping the middle 70% of width *and* height, and that shape does not match how
these cameras are set up:

- **The bottom of the frame is always our own pitch.** The camera sits close to
  the touchline, so the near half of the field runs off the bottom edge. Anything
  standing there is a player — there is no room between the camera and the pitch
  for someone else's match.
- **The sides are our pitch too.** A wide Veo frame is one pitch across; the
  neighbouring matches are *beyond* the far touchline, not left and right of it.
- **The top is where the intruders are** — the far touchline crowd, the next
  pitch over, the car park, the trees.

Cutting the bottom and the sides therefore threw away real players to remove
nothing. Job 38180242 measured the old rectangle discarding **36% of detected
people** (22/frame → 14/frame) on U14G Veo footage while *keeping* actual
spectators along the far touchline, and on ``runs/saints-u14g-full`` it clipped
every player box into x ∈ [288, 1632], y ≤ 918 — which is why a player standing
in the near corner never got a track id, and so never appeared in a tracklet
labelling clip.

Note what the top cut can and cannot do: it removes whatever is above a fixed
line, which on a wide shot is sky, trees and rooftops. The far-touchline crowd
and the next pitch sit *below* that line, mixed in with our own far-side players,
and no horizontal cut separates them — that is the pitch-region problem
(issue #21), not this function's job.
"""

from __future__ import annotations

import supervision as sv

# Fraction of frame height cut from the top. 0.15 is what the old centred
# rectangle used, kept so the default only ever *adds* detections back.
DEFAULT_TOP_FRAC = 0.15
# Sides and bottom are open by default: see the module docstring for why every
# pixel there is our own pitch on the cameras we shoot with.
DEFAULT_SIDE_FRAC = 0.0
DEFAULT_BOTTOM_FRAC = 0.0


def _box_foot_point(xyxy):
    """Bottom-center of each bounding box (foot position)."""
    import numpy as np

    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = xyxy[:, 3]  # bottom edge
    return np.column_stack([cx, cy])


def filter_by_field_hull(
    detections: sv.Detections,
    frame_shape: tuple[int, int],
    top_frac: float = DEFAULT_TOP_FRAC,
    side_frac: float = DEFAULT_SIDE_FRAC,
    bottom_frac: float = DEFAULT_BOTTOM_FRAC,
) -> sv.Detections:
    """Keep detections whose foot-point sits inside the field band.

    Each fraction is the share of the frame cut from that edge, measured against
    the **foot point** (bottom-centre of the box) rather than the whole box, so a
    player whose head or raised arm leaves the band is still kept.

    Defaults cut the top only. Pass ``side_frac`` / ``bottom_frac`` for a venue
    where that assumption breaks — a camera set well back from the touchline, or
    a pitch that genuinely ends before the frame does.
    """
    if len(detections) == 0:
        return detections

    h, w = frame_shape[:2]
    margin_x = w * side_frac
    top = h * top_frac
    bottom = h * (1 - bottom_frac)

    feet = _box_foot_point(detections.xyxy)
    inside = (
        (feet[:, 0] >= margin_x)
        & (feet[:, 0] <= w - margin_x)
        & (feet[:, 1] >= top)
        & (feet[:, 1] <= bottom)
    )
    return detections[inside]


def filter_spectators(
    detections: sv.Detections,
    frame_shape: tuple[int, int],
    top_frac: float = DEFAULT_TOP_FRAC,
    side_frac: float = DEFAULT_SIDE_FRAC,
    bottom_frac: float = DEFAULT_BOTTOM_FRAC,
) -> sv.Detections:
    """Drop off-field detections (spectators, coaches, subs)."""
    return filter_by_field_hull(
        detections, frame_shape, top_frac, side_frac, bottom_frac
    )
