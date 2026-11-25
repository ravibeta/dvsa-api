"""Multi-object vehicle tracker: ByteTrack-style association in camera-compensated (ref) coordinates.

- Boxes are tracked in the ref plane from :mod:`core.vehicles.motion`, so drone motion never shows
  up as vehicle motion and the Kalman filter only models the vehicles' own movement.
- Two-stage association (ByteTrack): confident detections first; low-confidence detections may only
  *continue* confirmed tracks, never start one.
- Lost tracks can be re-found by centre distance, gated in body lengths and growing with time lost.
- A track needs ``min_hits`` matches before it gets a public ID, so one-frame false positives do not
  inflate counts. No appearance re-ID: a vehicle that leaves and re-enters gets a new ID.
- All gates and ages are in seconds, so results do not depend on frame stride or VFR.
- Each track keeps a class vote (car / truck / ...) over its detections.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boxes import box_iou

TENTATIVE, CONFIRMED, LOST, REMOVED = "tentative", "confirmed", "lost", "removed"


@dataclass
class TrackerConfig:
    high_thresh: float = 0.25
    low_thresh: float = 0.10
    new_track_thresh: float = 0.30
    match_iou: float = 0.20
    low_match_iou: float = 0.30
    tentative_iou: float = 0.30
    refind_dist_bl: float = 1.0
    refind_growth_bl_s: float = 1.5
    refind_max_bl: float = 4.0
    min_hits: int = 3
    max_lost_s: float = 2.0
    lost_velocity_decay_s: float = 5.0


class KalmanBox:
    """Constant-velocity Kalman filter on (cx, cy, w, h); velocities in units per second.

    Noise is proportional to box size (ByteTrack's scheme) and scaled to per-second time steps
    (``dt * 30`` equivalent 30 fps frames), so irregular steps are handled correctly.
    """

    std_pos = 1.0 / 20
    std_vel = 30.0 / 160

    def __init__(self, box_xyxy: np.ndarray):
        cx, cy, w, h = self._xywh(box_xyxy)
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        s = max(w, h, 1.0)
        self.P = np.diag(np.square([2 * self.std_pos * s] * 4 + [10 * self.std_vel * s] * 4))

    @staticmethod
    def _xywh(b) -> Tuple[float, float, float, float]:
        b = np.asarray(b, dtype=np.float64)
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]

    def predict(self, dt: float, velocity_scale: float = 1.0) -> None:
        if dt <= 0:
            return
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        self.x[4:] *= velocity_scale
        s = max(self.x[2], self.x[3], 1.0)
        Q = np.diag(np.square([self.std_pos * s] * 4 + [self.std_vel * s] * 4)) * (dt * 30.0)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, box_xyxy: np.ndarray) -> None:
        z = np.array(self._xywh(box_xyxy))
        s = max(z[2], z[3], 1.0)
        R = np.diag(np.square([self.std_pos * s] * 4))
        Hm = np.hstack([np.eye(4), np.zeros((4, 4))])
        S = Hm @ self.P @ Hm.T + R
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - Hm @ self.x)
        self.P = (np.eye(8) - K @ Hm) @ self.P

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Observation:
    t: float
    frame: int
    box_ref: np.ndarray
    box_img: np.ndarray
    conf: float
    stage: str  # "init" | "high" | "low" | "refind"
    class_name: str


@dataclass
class Track:
    uid: int
    kf: KalmanBox
    first_t: float
    last_t: float
    state: str = TENTATIVE
    id: Optional[int] = None
    hits: int = 1
    obs: List[Observation] = field(default_factory=list)
    gaps: List[Tuple[float, float]] = field(default_factory=list)
    lost_since: Optional[float] = None
    end_reason: Optional[str] = None
    class_votes: Counter = field(default_factory=Counter)
    class_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> float:
        b = self.kf.box
        return float(max(b[2] - b[0], b[3] - b[1]))

    @property
    def class_name(self) -> str:
        if not self.class_votes:
            return "vehicle"
        return max(self.class_votes, key=lambda c: (self.class_votes[c], self.class_scores.get(c, 0.0)))

    def add(self, ob: Observation) -> None:
        self.obs.append(ob)
        self.class_votes[ob.class_name] += 1
        self.class_scores[ob.class_name] = self.class_scores.get(ob.class_name, 0.0) + ob.conf


Match = List[Tuple[Track, int]]


class VehicleTracker:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.cfg = config or TrackerConfig()
        self.tracks: List[Track] = []
        self.finished: List[Track] = []
        self._next_uid = 1
        self._next_id = 1
        self._t: Optional[float] = None

    @staticmethod
    def _assign(cost: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
        if cost.size == 0:
            return []
        c = np.where(cost <= max_cost, cost, 1e6)
        rows, cols = linear_sum_assignment(c)
        return [(i, j) for i, j in zip(rows, cols) if c[i, j] <= max_cost]

    @staticmethod
    def _split(tracks, det_idx, pairs):
        mt = {i for i, _ in pairs}
        md = {j for _, j in pairs}
        return ([(tracks[i], det_idx[j]) for i, j in pairs],
                [t for i, t in enumerate(tracks) if i not in mt],
                [d for j, d in enumerate(det_idx) if j not in md])

    def _iou_match(self, tracks: List[Track], boxes: np.ndarray, det_idx: List[int], min_iou: float):
        if not tracks or not det_idx:
            return [], list(tracks), list(det_idx)
        iou = box_iou(np.stack([t.kf.box for t in tracks]), boxes[det_idx])
        return self._split(tracks, det_idx, self._assign(1.0 - iou, 1.0 - min_iou))

    def _dist_match(self, tracks: List[Track], boxes: np.ndarray, det_idx: List[int], t_now: float):
        if not tracks or not det_idx:
            return [], list(tracks), list(det_idx)
        cfg = self.cfg
        tc = np.stack([t.kf.x[:2] for t in tracks])
        b = boxes[det_idx]
        dc = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
        size = np.array([max(t.size, 1.0) for t in tracks])[:, None]
        gate = np.array([min(cfg.refind_max_bl, cfg.refind_dist_bl + cfg.refind_growth_bl_s * (t_now - t.last_t))
                         for t in tracks])[:, None]
        d = np.linalg.norm(tc[:, None] - dc[None], axis=2) / size / gate
        return self._split(tracks, det_idx, self._assign(d, 1.0))

    def update(self, t: float, frame: int, boxes_ref, boxes_img, scores, classes: Sequence[str],
               in_view: Optional[Callable[[Track], bool]] = None) -> Match:
        """Advance to time ``t`` with this frame's detections.

        Returns ``(confirmed track, detection index)`` for every confirmed track matched now.
        ``in_view(track)`` says whether a lost track's predicted position is still in the image.
        """
        cfg = self.cfg
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        boxes_ref = np.asarray(boxes_ref, dtype=np.float64).reshape(-1, 4)
        boxes_img = np.asarray(boxes_img, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        classes = list(classes) if classes is not None else ["vehicle"] * len(scores)

        for tr in self.tracks:
            decay = float(np.exp(-dt / cfg.lost_velocity_decay_s)) if tr.state == LOST else 1.0
            tr.kf.predict(dt, decay)

        high = [i for i in range(len(scores)) if scores[i] >= cfg.high_thresh]
        low = [i for i in range(len(scores)) if cfg.low_thresh <= scores[i] < cfg.high_thresh]
        active = [tr for tr in self.tracks if tr.state in (CONFIRMED, LOST)]
        tentative = [tr for tr in self.tracks if tr.state == TENTATIVE]

        m1, rest_tr, rest_high = self._iou_match(active, boxes_ref, high, cfg.match_iou)
        cont = [tr for tr in rest_tr if tr.state == CONFIRMED]
        m2, _, _ = self._iou_match(cont, boxes_ref, low, cfg.low_match_iou)
        matched2 = {id(tr) for tr, _ in m2}
        rest_tr = [tr for tr in rest_tr if id(tr) not in matched2]
        m3, rest_tent, rest_high = self._iou_match(tentative, boxes_ref, rest_high, cfg.tentative_iou)
        m4, rest_tr, rest_high = self._dist_match(rest_tr, boxes_ref, rest_high, t)

        out: Match = []
        for stage, pairs in (("high", m1), ("low", m2), ("high", m3), ("refind", m4)):
            for tr, j in pairs:
                tr.kf.update(boxes_ref[j])
                tr.hits += 1
                if tr.state == LOST:
                    tr.gaps.append((tr.lost_since, t))
                    tr.lost_since = None
                    tr.state = CONFIRMED
                tr.last_t = t
                tr.add(Observation(t, frame, boxes_ref[j].copy(), boxes_img[j].copy(), float(scores[j]),
                                   stage, str(classes[j])))
                if tr.state == TENTATIVE and tr.hits >= cfg.min_hits:
                    tr.state = CONFIRMED
                    tr.id = self._next_id
                    self._next_id += 1
                if tr.state == CONFIRMED:
                    out.append((tr, j))

        for tr in rest_tent:  # a tentative track that misses once is dropped
            tr.state, tr.end_reason = REMOVED, "unconfirmed"
        for tr in rest_tr:
            if tr.state == CONFIRMED:
                tr.state, tr.lost_since = LOST, tr.last_t
            if t - tr.last_t > cfg.max_lost_s:
                tr.state = REMOVED
                tr.end_reason = "lost" if (in_view is None or in_view(tr)) else "left_view"

        for j in rest_high:
            if scores[j] >= cfg.new_track_thresh:
                tr = Track(self._next_uid, KalmanBox(boxes_ref[j]), t, t)
                tr.add(Observation(t, frame, boxes_ref[j].copy(), boxes_img[j].copy(), float(scores[j]),
                                   "init", str(classes[j])))
                self._next_uid += 1
                self.tracks.append(tr)

        self.finished.extend(tr for tr in self.tracks if tr.state == REMOVED and tr.id is not None)
        self.tracks = [tr for tr in self.tracks if tr.state != REMOVED]
        return out

    def finalize(self) -> List[Track]:
        """End all tracks (end of video) and return every confirmed track, ordered by id."""
        for tr in self.tracks:
            if tr.id is not None:
                tr.end_reason = "lost_at_end" if tr.state == LOST else "end_of_video"
                self.finished.append(tr)
        self.tracks = []
        return sorted(self.finished, key=lambda tr: tr.id)

    @property
    def n_confirmed(self) -> int:
        return self._next_id - 1
