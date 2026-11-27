"""MOVING / STATIONARY / UNKNOWN classification from ground-relative (ref) track positions.

Per confirmed track and processed frame: fit position = a + v*t by least squares over the last
``window_s`` seconds of ref-plane centres, normalise speed and its standard error by the vehicle's
apparent length (median long box side), so speeds are in **body lengths per second (BL/s)** and
thresholds do not depend on resolution or altitude. Decide with a 2-sigma margin and hysteresis
(``min_dwell_s`` before switching between MOVING and STATIONARY). UNKNOWN when history is too
short or the window contains a failed camera registration.

``confidence`` is Phi(margin / se) from the threshold midpoint: a distance-from-boundary score,
not a calibrated probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

MOVING, STATIONARY, UNKNOWN = "MOVING", "STATIONARY", "UNKNOWN"
STATES = (MOVING, STATIONARY, UNKNOWN)


@dataclass
class ActivityConfig:
    window_s: float = 1.0
    min_span_s: float = 0.5
    min_points: int = 5
    # Measured on nadir city footage (cars ~20 px): parked cars 0.04-0.16 BL/s from box jitter,
    # driving cars 1.3-2.2 BL/s. 0.5 BL/s is ~2 m/s (7 km/h) for a 4.5 m car.
    moving_bl_s: float = 0.5
    stationary_bl_s: float = 0.2
    z: float = 2.0
    min_dwell_s: float = 0.5
    min_se_bl_s: float = 0.01


@dataclass
class MotionEstimate:
    vx: float  # ref px / s
    vy: float
    speed_bl_s: float
    se_bl_s: float
    body_len_px: float
    heading_deg: float  # image convention: 0 = +x (right), 90 = +y (down)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate_motion(t: np.ndarray, xy: np.ndarray, body_len: float) -> MotionEstimate:
    """Least-squares ground velocity and its standard error from (t, xy) samples."""
    t = np.asarray(t, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    tc = t - t.mean()
    stt = float((tc ** 2).sum()) or 1e-12
    v = (tc[:, None] * (xy - xy.mean(0))).sum(0) / stt
    resid = xy - xy.mean(0) - tc[:, None] * v
    dof = max(1, 2 * len(t) - 4)
    sigma = math.sqrt(float((resid ** 2).sum()) / dof)
    se = sigma / math.sqrt(stt)
    bl = max(float(body_len), 1e-6)
    speed = float(np.hypot(v[0], v[1]))
    return MotionEstimate(float(v[0]), float(v[1]), speed / bl, se / bl, bl,
                          float(math.degrees(math.atan2(v[1], v[0]))))


class TrackActivity:
    """Per-track state machine."""

    def __init__(self, cfg: ActivityConfig):
        self.cfg = cfg
        self.state = UNKNOWN
        self.confidence: Optional[float] = None
        self.estimate: Optional[MotionEstimate] = None
        self.reason = "insufficient_history"
        self._pending: Optional[str] = None
        self._pending_since: Optional[float] = None

    def _unknown(self, reason: str) -> str:
        self.state, self.confidence, self.reason, self.estimate = UNKNOWN, None, reason, None
        self._pending = self._pending_since = None
        return self.state

    def update(self, t_now: float, ts, xy, sizes, motion_ok: bool) -> str:
        cfg = self.cfg
        ts = np.asarray(ts, dtype=np.float64)
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        sizes = np.asarray(sizes, dtype=np.float64)
        sel = ts > t_now - cfg.window_s - 1e-9
        ts, xy, sizes = ts[sel], xy[sel], sizes[sel]
        if not motion_ok:
            return self._unknown("camera_registration_failed")
        if len(ts) < cfg.min_points or ts[-1] - ts[0] < cfg.min_span_s:
            return self._unknown("insufficient_history")
        est = estimate_motion(ts, xy, float(np.median(sizes)))
        self.estimate = est
        se = max(est.se_bl_s, cfg.min_se_bl_s)
        s = est.speed_bl_s
        if s >= cfg.moving_bl_s and s - cfg.z * se > cfg.stationary_bl_s:
            cand: Optional[str] = MOVING
        elif s <= cfg.stationary_bl_s and s + cfg.z * se < cfg.moving_bl_s:
            cand = STATIONARY
        else:
            cand = None

        if cand is not None:
            if self.state == UNKNOWN or cand == self.state:
                self.state, self._pending, self._pending_since = cand, None, None
            elif self._pending != cand:
                self._pending, self._pending_since = cand, t_now
            elif t_now - self._pending_since >= cfg.min_dwell_s - 1e-9:
                self.state, self._pending, self._pending_since = cand, None, None
        else:
            self._pending = self._pending_since = None
        self.reason = "measured" if cand is not None else ("hysteresis" if self.state != UNKNOWN else "ambiguous")
        mid = 0.5 * (cfg.moving_bl_s + cfg.stationary_bl_s)
        if self.state == MOVING:
            self.confidence = _phi((s - mid) / se)
        elif self.state == STATIONARY:
            self.confidence = _phi((mid - s) / se)
        else:
            self.confidence = None
        return self.state
