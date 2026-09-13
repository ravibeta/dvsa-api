"""Multi-object tracking and trajectory reconstruction.

Ports the "multi-object tracker" + "trajectory reconstruction" capabilities from
``my_droneworld_api`` (see ``to-port-from-ezvision.md``). dvsa-api previously
emitted detections only, with a greedy ``CentroidTracker`` (in
``zone_counting.py``) for simple ID assignment; this module adds a SORT tracker (a
constant-velocity Kalman filter per object + Hungarian association) and a
video-level routine that turns per-frame detections into **per-object
trajectories** across a clip.

Training-free and dependency-light: only ``numpy`` and ``scipy`` (both already in
``requirements/base.txt``) — no appearance model, so no new dependency or model
weights (unlike DeepSORT). Everything is pure, deterministic, and unit-testable
without a video runtime or Azure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import Detection, register

BBox = Tuple[int, int, int, int]  # (x, y, w, h) in pixels
Point = Tuple[float, float]

# Reported track: (track_id, bbox_xywh, centroid, label, score)
Reported = Tuple[int, BBox, Point, str, float]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _xywh_to_xyxy(bbox: Sequence[float]) -> np.ndarray:
    x, y, w, h = (float(v) for v in bbox)
    return np.array([x, y, x + w, y + h], dtype=float)


def _xyxy_to_xywh(box: np.ndarray) -> BBox:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (int(round(x1)), int(round(y1)),
            int(round(max(0.0, x2 - x1))), int(round(max(0.0, y2 - y1))))


def _centroid_of(box: np.ndarray) -> Point:
    x1, y1, x2, y2 = (float(v) for v in box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Kalman box tracker (constant-velocity SORT state)
# ---------------------------------------------------------------------------
class KalmanBoxTracker:
    """A single object's constant-velocity Kalman filter.

    State ``x = [cx, cy, s, r, vcx, vcy, vs]`` — box centre, scale (area), aspect
    ratio, and their velocities (``r`` assumed constant). Hand-rolled with numpy
    (no ``filterpy`` dependency) using the standard SORT matrices.
    """

    def __init__(self, bbox_xyxy: np.ndarray, track_id: int,
                 label: str = "object", score: float = 1.0):
        self.id = track_id
        self.label = label
        self.score = float(score)

        # Motion (F) and measurement (H) models.
        self._F = np.eye(7)
        for i in range(3):
            self._F[i, i + 4] = 1.0
        self._H = np.zeros((4, 7))
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = self._H[3, 3] = 1.0

        # Covariances (SORT defaults): high uncertainty on unobserved velocities.
        self._P = np.eye(7)
        self._P[4:, 4:] *= 1000.0
        self._P *= 10.0
        self._Q = np.eye(7)
        self._Q[4:, 4:] *= 0.01
        self._R = np.eye(4)
        self._R[2:, 2:] *= 10.0

        self._x = np.zeros((7, 1))
        self._x[:4, 0] = self._to_z(bbox_xyxy)

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    @staticmethod
    def _to_z(box: np.ndarray) -> np.ndarray:
        w = max(1e-6, box[2] - box[0])
        h = max(1e-6, box[3] - box[1])
        cx = box[0] + w / 2.0
        cy = box[1] + h / 2.0
        return np.array([cx, cy, w * h, w / h])

    @staticmethod
    def _to_box(x: np.ndarray) -> np.ndarray:
        cx, cy, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        s = max(0.0, s)
        r = max(1e-6, r)
        w = float(np.sqrt(s * r))
        h = float(s / w) if w > 0 else 0.0
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])

    def predict(self) -> np.ndarray:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._to_box(self._x)

    def update(self, bbox_xyxy: np.ndarray, label: str, score: float) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.label = label
        self.score = float(score)

        z = self._to_z(bbox_xyxy).reshape(4, 1)
        y = z - self._H @ self._x
        s = self._H @ self._P @ self._H.T + self._R
        k = self._P @ self._H.T @ np.linalg.inv(s)
        self._x = self._x + k @ y
        self._P = (np.eye(7) - k @ self._H) @ self._P

    def current_box(self) -> np.ndarray:
        return self._to_box(self._x)


# ---------------------------------------------------------------------------
# SORT tracker
# ---------------------------------------------------------------------------
class SortTracker:
    """SORT: Kalman prediction + IoU/Hungarian association across frames.

    Call :meth:`update` once per (sampled) frame with that frame's detections;
    it returns the confirmed tracks visible in this frame as
    ``(track_id, bbox_xywh, centroid, label, score)`` tuples. IDs are stable and
    assigned per-instance, so a run is deterministic.
    """

    def __init__(self, *, max_age: int = 10, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self._trackers: List[KalmanBoxTracker] = []
        self._next_id = 0
        self._frame_count = 0

    def update(self, detections: Sequence[Detection]) -> List[Reported]:
        self._frame_count += 1

        det_boxes = [_xywh_to_xyxy(d.bbox) for d in detections]
        labels = [d.label for d in detections]
        scores = [d.score for d in detections]

        # Predict existing tracks; drop any that went numerically invalid.
        predicted: List[np.ndarray] = []
        alive: List[KalmanBoxTracker] = []
        for trk in self._trackers:
            box = trk.predict()
            if not np.any(np.isnan(box)):
                predicted.append(box)
                alive.append(trk)
        self._trackers = alive

        matches, unmatched_dets, unmatched_trks = self._associate(det_boxes, predicted)

        for det_idx, trk_idx in matches:
            self._trackers[trk_idx].update(det_boxes[det_idx], labels[det_idx],
                                           scores[det_idx])
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(det_boxes[det_idx], self._next_id,
                                   labels[det_idx], scores[det_idx])
            self._next_id += 1
            self._trackers.append(trk)

        # Report confirmed tracks; retire stale ones.
        reported: List[Reported] = []
        kept: List[KalmanBoxTracker] = []
        for trk in self._trackers:
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self._frame_count <= self.min_hits
            ):
                box = trk.current_box()
                reported.append((trk.id, _xyxy_to_xywh(box), _centroid_of(box),
                                 trk.label, trk.score))
            if trk.time_since_update <= self.max_age:
                kept.append(trk)
        self._trackers = kept
        reported.sort(key=lambda r: r[0])
        return reported

    def _associate(self, det_boxes: List[np.ndarray], trk_boxes: List[np.ndarray]):
        """Hungarian association on IoU, gated by ``iou_threshold``."""
        if not det_boxes or not trk_boxes:
            return [], list(range(len(det_boxes))), list(range(len(trk_boxes)))

        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        iou_mat = np.zeros((len(det_boxes), len(trk_boxes)), dtype=float)
        for di, db in enumerate(det_boxes):
            for ti, tb in enumerate(trk_boxes):
                iou_mat[di, ti] = iou(db, tb)

        det_idx, trk_idx = linear_sum_assignment(-iou_mat)
        matches: List[Tuple[int, int]] = []
        matched_d, matched_t = set(), set()
        for di, ti in zip(det_idx, trk_idx):
            if iou_mat[di, ti] < self.iou_threshold:
                continue
            matches.append((int(di), int(ti)))
            matched_d.add(int(di))
            matched_t.add(int(ti))

        unmatched_dets = [i for i in range(len(det_boxes)) if i not in matched_d]
        unmatched_trks = [i for i in range(len(trk_boxes)) if i not in matched_t]
        return matches, unmatched_dets, unmatched_trks


# ---------------------------------------------------------------------------
# Trajectory data structures
# ---------------------------------------------------------------------------
@dataclass
class TrackPoint:
    frame: int
    t: Optional[float]
    centroid: Point
    bbox: BBox
    score: float

    def to_dict(self) -> dict:
        return {
            "frame": int(self.frame),
            "t": (None if self.t is None else round(float(self.t), 3)),
            "centroid": [round(float(self.centroid[0]), 2), round(float(self.centroid[1]), 2)],
            "bbox": [int(v) for v in self.bbox],
            "score": round(float(self.score), 4),
        }


@dataclass
class Trajectory:
    track_id: int
    label: str = "object"
    points: List[TrackPoint] = field(default_factory=list)

    def _path_length(self) -> float:
        total = 0.0
        for a, b in zip(self.points, self.points[1:]):
            total += float(np.hypot(b.centroid[0] - a.centroid[0],
                                    b.centroid[1] - a.centroid[1]))
        return total

    def _displacement(self) -> float:
        if len(self.points) < 2:
            return 0.0
        a, b = self.points[0].centroid, self.points[-1].centroid
        return float(np.hypot(b[0] - a[0], b[1] - a[1]))

    def _mean_bbox_diag(self) -> float:
        if not self.points:
            return 0.0
        diags = [float(np.hypot(p.bbox[2], p.bbox[3])) for p in self.points]
        return float(np.mean(diags))

    def to_dict(self) -> dict:
        path_len = self._path_length()
        disp = self._displacement()
        first, last = self.points[0], self.points[-1]
        duration = (None if first.t is None or last.t is None
                    else round(float(last.t - first.t), 3))
        mean_speed = (round(path_len / duration, 2)
                      if duration and duration > 0 else None)
        stationary_gate = max(5.0, 0.5 * self._mean_bbox_diag())
        return {
            "track_id": int(self.track_id),
            "label": self.label,
            "first_frame": int(first.frame),
            "last_frame": int(last.frame),
            "num_points": len(self.points),
            "path_length_px": round(path_len, 2),
            "displacement_px": round(disp, 2),
            "duration_s": duration,
            "mean_speed_px_s": mean_speed,
            "is_stationary": bool(disp < stationary_gate),
            "points": [p.to_dict() for p in self.points],
        }


# ---------------------------------------------------------------------------
# Video-level routine
# ---------------------------------------------------------------------------
def _det_from_dict(d: dict) -> Detection:
    bbox = tuple(int(v) for v in d.get("bbox", (0, 0, 0, 0)))
    if "centroid" in d and d["centroid"] is not None:
        centroid = (float(d["centroid"][0]), float(d["centroid"][1]))
    else:
        centroid = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
    return Detection(
        bbox=bbox, centroid=centroid,
        area=float(d.get("area", bbox[2] * bbox[3])),
        label=d.get("label", "object"), score=float(d.get("score", 1.0)),
    )


@register(
    "trajectory_reconstruction",
    "Reconstruct per-object trajectories across a clip via SORT tracking.",
    level="video",
)
def trajectory_reconstruction_routine(
    frames,
    *,
    detections_per_frame: Optional[Sequence[Sequence[dict]]] = None,
    fps: Optional[float] = None,
    frame_step: int = 1,
    min_track_len: int = 2,
    max_age: int = 10,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    lower_hsv: Optional[Sequence[int]] = None,
    upper_hsv: Optional[Sequence[int]] = None,
    min_area: float = 100.0,
    **_: object,
) -> dict:
    """Track objects across the frame stream and return their trajectories.

    Detections come from ``detections_per_frame`` (a list indexed by sampled
    frame ordinal, each item a list of ``{bbox, centroid?, label?, score?}``) when
    supplied; otherwise objects are detected per frame via colour segmentation so
    the routine is self-contained. Timestamps (``t``) are populated when ``fps``
    is given (``t = ordinal * frame_step / fps``).
    """
    tracker = SortTracker(max_age=max_age, min_hits=min_hits,
                          iou_threshold=iou_threshold)
    trajectories: Dict[int, Trajectory] = {}
    num_frames = 0

    for i, frame in enumerate(frames):
        num_frames = i + 1
        if detections_per_frame is not None:
            raw = detections_per_frame[i] if i < len(detections_per_frame) else []
            dets = [_det_from_dict(d) for d in raw]
        else:
            from .detection import detect_by_color  # noqa: PLC0415

            dets = detect_by_color(
                frame, lower_hsv or [0, 70, 50], upper_hsv or [10, 255, 255],
                min_area=min_area,
            )
        t = (i * frame_step / fps) if fps else None
        for track_id, bbox, centroid, label, score in tracker.update(dets):
            traj = trajectories.setdefault(
                track_id, Trajectory(track_id=track_id, label=label))
            traj.label = label or traj.label
            traj.points.append(TrackPoint(frame=i, t=t, centroid=centroid,
                                          bbox=bbox, score=score))

    kept = [tr for tr in trajectories.values() if len(tr.points) >= min_track_len]
    kept.sort(key=lambda tr: tr.track_id)
    dicts = [tr.to_dict() for tr in kept]

    moving = sum(1 for d in dicts if not d["is_stationary"])
    return {
        "routine": "trajectory_reconstruction",
        "summary": {
            "num_tracks": len(dicts),
            "num_frames": num_frames,
            "fps": fps,
            "total_path_length_px": round(sum(d["path_length_px"] for d in dicts), 2),
            "moving_tracks": moving,
            "stationary_tracks": len(dicts) - moving,
        },
        "trajectories": dicts,
    }
