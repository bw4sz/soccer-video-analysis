"""Kalman smoothing for the flickery single-ball track.

The RF-DETR ball detector fires one detection per frame, but on youth footage
it *flickers*: it latches onto a white jersey number or a sponsor logo for a
frame or two, so the reported position teleports across the pitch and snaps
back. There is only ever **one ball**, and between frames it moves smoothly, so
a constant-velocity Kalman filter is a good fit — it predicts where the ball
should be, accepts detections that land near that prediction, and *rejects the
huge jumps* as outliers.

This module is pure (no video, no ffmpeg, numpy only) so it can be unit-tested
against synthetic tracks, and it operates on the same ball-track schema the rest
of the trim workflow uses (see :mod:`soccer_vision.events.deadball`). It reads
``visible`` / ``pixel_x`` / ``pixel_y`` and returns a *copy* of the track with
smoothed positions, preserving every other field.

State model
-----------
Per axis the filter tracks position and velocity, ``[x, y, vx, vy]``, under a
constant-velocity (random-acceleration) model. Each visible detection is a
measurement of ``(x, y)``. A detection is **gated**: its Mahalanobis distance
from the prediction must fall under ``gate_chi2`` (a chi-square threshold on 2
degrees of freedom, default 9.21 ≈ 99%). Detections that fail the gate are
treated as flicker — the filter coasts on its prediction instead. If several
detections in a row fail the gate, or the ball has been offscreen for a while,
the filter assumes the scene really did change and **re-locks** onto the latest
detection.

Offscreen samples (``visible: false``) are passed through untouched so the
dead-time detector still sees genuine out-of-play gaps.
"""

from __future__ import annotations

import numpy as np

# Defaults tuned for ~5 fps youth footage; all overridable.
MEASUREMENT_PX = 8.0      # detector centre noise, 1σ in pixels
PROCESS_ACCEL = 300.0     # ball acceleration 1σ in px/s² (motion the filter allows)
GATE_CHI2 = 9.21          # χ² gate, 2 dof, ~99%: reject detections beyond this
REACQUIRE_AFTER = 3       # consecutive rejects before we re-lock to the detection
REACQUIRE_GAP_S = 1.0     # offscreen longer than this re-inits on the next detection
INIT_VEL_PX_S = 1000.0    # 1σ prior on velocity at (re)initialisation, px/s


class KalmanBallFilter:
    """Constant-velocity Kalman filter for a single ball, with outlier gating.

    Feed measurements in time order via :meth:`step`. Each call returns the
    filtered ``(x, y)`` estimate and whether the measurement was accepted; a
    rejected or missing measurement makes the filter coast on its prediction.
    """

    def __init__(
        self,
        *,
        measurement_px: float = MEASUREMENT_PX,
        process_accel: float = PROCESS_ACCEL,
        gate_chi2: float = GATE_CHI2,
        reacquire_after: int = REACQUIRE_AFTER,
        init_vel_px_s: float = INIT_VEL_PX_S,
    ) -> None:
        self.measurement_px = measurement_px
        self.process_accel = process_accel
        self.gate_chi2 = gate_chi2
        self.reacquire_after = reacquire_after
        self.init_vel_px_s = init_vel_px_s

        self._H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self._R = (measurement_px**2) * np.eye(2)

        self.x: np.ndarray | None = None   # state [x, y, vx, vy]
        self.P: np.ndarray | None = None   # 4×4 covariance
        self._consecutive_rejects = 0

    @property
    def initialised(self) -> bool:
        return self.x is not None

    def reset(self, mx: float, my: float) -> None:
        """(Re)initialise the state on a fresh detection at ``(mx, my)``."""
        self.x = np.array([mx, my, 0.0, 0.0])
        self.P = np.diag([
            self.measurement_px**2,
            self.measurement_px**2,
            self.init_vel_px_s**2,
            self.init_vel_px_s**2,
        ])
        self._consecutive_rejects = 0

    def _predict(self, dt: float) -> None:
        F = np.array([
            [1.0, 0, dt, 0],
            [0, 1.0, 0, dt],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0],
        ])
        # Random-acceleration process noise for a constant-velocity model.
        q = self.process_accel**2
        dt2, dt3, dt4 = dt * dt, dt * dt * dt, dt * dt * dt * dt
        Q = q * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def step(
        self,
        dt: float,
        measurement: tuple[float, float] | None,
    ) -> tuple[tuple[float, float] | None, bool]:
        """Advance ``dt`` seconds and fold in ``measurement`` (or ``None``).

        Returns ``((x, y), accepted)``. ``accepted`` is True when the detection
        passed the gate and updated the filter; False when it was rejected as an
        outlier, was missing, or the filter had to re-lock. Before the first
        detection the estimate is ``None``.
        """
        # Uninitialised: the first detection seeds the state.
        if not self.initialised:
            if measurement is None:
                return None, False
            self.reset(*measurement)
            return (float(self.x[0]), float(self.x[1])), True

        if dt > 0:
            self._predict(dt)

        if measurement is None:
            # Coast: no detection to fold in.
            return (float(self.x[0]), float(self.x[1])), False

        # Gate the detection by its Mahalanobis distance from the prediction.
        z = np.array(measurement)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        d2 = float(y @ np.linalg.solve(S, y))

        if d2 > self.gate_chi2:
            self._consecutive_rejects += 1
            if self._consecutive_rejects >= self.reacquire_after:
                # Detections keep disagreeing with us — the ball really moved.
                self.reset(*measurement)
                return (float(self.x[0]), float(self.x[1])), True
            # Treat as flicker: coast on the prediction.
            return (float(self.x[0]), float(self.x[1])), False

        # Accept: standard Kalman update.
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self._H) @ self.P
        self._consecutive_rejects = 0
        return (float(self.x[0]), float(self.x[1])), True


def smooth_samples(
    samples: list[dict],
    *,
    measurement_px: float = MEASUREMENT_PX,
    process_accel: float = PROCESS_ACCEL,
    gate_chi2: float = GATE_CHI2,
    reacquire_after: int = REACQUIRE_AFTER,
    reacquire_gap_s: float = REACQUIRE_GAP_S,
    init_vel_px_s: float = INIT_VEL_PX_S,
) -> list[dict]:
    """Return a smoothed copy of ``samples`` (deadball schema).

    Visible detections that pass the gate are replaced by the filtered estimate;
    flicker jumps are rejected and the ball coasts on the prediction (still
    marked visible, since the ball is on the pitch — just detected wrong). The
    original detection is preserved under ``raw_pixel_x`` / ``raw_pixel_y``, and
    each smoothed sample gains ``smoothed: true``; rejected jumps also get
    ``outlier: true``. Offscreen samples pass through unchanged so genuine
    out-of-play gaps still read as dead time downstream.
    """
    kf = KalmanBallFilter(
        measurement_px=measurement_px,
        process_accel=process_accel,
        gate_chi2=gate_chi2,
        reacquire_after=reacquire_after,
        init_vel_px_s=init_vel_px_s,
    )

    out: list[dict] = []
    prev_t: float | None = None
    last_visible_t: float | None = None

    for s in samples:
        t = s["timestamp_s"]
        dt = 0.0 if prev_t is None else max(0.0, t - prev_t)
        prev_t = t

        visible = s.get("visible", False) and s.get("pixel_x") is not None
        if not visible:
            # Pass offscreen/undetected samples through untouched.
            out.append(dict(s))
            continue

        # A long blackout means the old velocity is stale — start fresh.
        if (
            last_visible_t is not None
            and kf.initialised
            and (t - last_visible_t) > reacquire_gap_s
        ):
            kf.reset(s["pixel_x"], s["pixel_y"])
        last_visible_t = t

        est, accepted = kf.step(dt, (s["pixel_x"], s["pixel_y"]))
        new = dict(s)
        new["raw_pixel_x"] = s["pixel_x"]
        new["raw_pixel_y"] = s["pixel_y"]
        new["pixel_x"] = round(est[0], 1)
        new["pixel_y"] = round(est[1], 1)
        new["smoothed"] = True
        if not accepted:
            new["outlier"] = True
        out.append(new)

    return out


def smooth_ball_track(track: dict, **kwargs) -> dict:
    """Return a copy of ``track`` with Kalman-smoothed ``samples``.

    Thin wrapper over :func:`smooth_samples`; forwards tuning keyword arguments.
    """
    smoothed = dict(track)
    smoothed["samples"] = smooth_samples(track.get("samples", []), **kwargs)
    smoothed["smoothed"] = True
    return smoothed
