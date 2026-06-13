"""Zone-based counting and a lightweight centroid tracker.

Implements the dynamic polygonal zone counting described in
``plan-of-action.md`` §4.1 (the ANTLINGS_Drone style analytic) using
``shapely`` for point-in-polygon tests, plus a dependency-free nearest-neighbour
:class:`CentroidTracker` that assigns persistent IDs so that *unique* objects
(rather than per-frame occupancy) can be counted across a clip.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import Detection, register


# ---------------------------------------------------------------------------
# Point-in-zone counting
# ---------------------------------------------------------------------------


def count_in_zones(
    detections: Sequence[Detection],
    zones: Sequence[dict],
) -> Dict[str, int]:
    """Count detections whose centroid falls inside each polygonal zone.

    Parameters
    ----------
    detections:
        Sequence of :class:`Detection`.
    zones:
        Sequence of ``{"name": str, "polygon": [[x, y], ...]}`` dicts. Each
        polygon must have at least three vertices.

    Returns a mapping ``{zone_name: count}``.
    """

    from shapely.geometry import Point, Polygon

    polys = []
    for z in zones:
        poly = Polygon(z["polygon"])
        if not poly.is_valid:
            poly = poly.buffer(0)  # repair self-intersections
        polys.append((z["name"], poly))

    counts = {name: 0 for name, _ in polys}
    for det in detections:
        pt = Point(det.centroid)
        for name, poly in polys:
            if poly.contains(pt):
                counts[name] += 1
    return counts


# ---------------------------------------------------------------------------
# Centroid tracker (persistent IDs)
# ---------------------------------------------------------------------------


class CentroidTracker:
    """Greedy nearest-neighbour tracker assigning persistent integer IDs.

    A classical, training-free tracker (as referenced in §3/§4): each frame's
    centroids are matched to existing tracks by Euclidean distance. Tracks that
    are unmatched for ``max_disappeared`` frames are retired.
    """

    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 10):
        self.max_distance = float(max_distance)
        self.max_disappeared = int(max_disappeared)
        self._next_id = 0
        self.objects: "OrderedDict[int, Tuple[float, float]]" = OrderedDict()
        self.disappeared: "OrderedDict[int, int]" = OrderedDict()

    def _register(self, centroid: Tuple[float, float]) -> int:
        oid = self._next_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self._next_id += 1
        return oid

    def _deregister(self, oid: int) -> None:
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, centroids: Sequence[Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
        """Feed one frame's centroids; return ``{track_id: centroid}``."""

        if len(centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return dict(self.objects)

        if len(self.objects) == 0:
            for c in centroids:
                self._register(tuple(c))
            return dict(self.objects)

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()), dtype=float)
        input_centroids = np.array(centroids, dtype=float)

        # Pairwise distances (objects x inputs).
        d = np.linalg.norm(
            object_centroids[:, None, :] - input_centroids[None, :, :], axis=2
        )

        # Greedy assignment: smallest distances first.
        rows = d.min(axis=1).argsort()
        cols = d.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if d[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = tuple(input_centroids[col])
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing objects -> disappeared.
        for row in set(range(d.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        # Unmatched new centroids -> register.
        for col in set(range(d.shape[1])) - used_cols:
            self._register(tuple(input_centroids[col]))

        return dict(self.objects)


@register(
    "zone_counting",
    "Count detection centroids inside named polygonal zones (per frame).",
    level="frame",
)
def zone_counting_routine(
    frame: np.ndarray,
    zones: Optional[Sequence[dict]] = None,
    detections: Optional[Sequence[dict]] = None,
    lower_hsv: Optional[Sequence[int]] = None,
    upper_hsv: Optional[Sequence[int]] = None,
    min_area: float = 100.0,
    **_: object,
) -> dict:
    """Count objects per zone.

    If ``detections`` (list of dicts with a ``centroid``) are supplied they are
    used directly; otherwise objects are detected via color segmentation so the
    routine is self-contained.
    """

    if not zones:
        raise ValueError("zone_counting requires at least one zone")

    if detections is not None:
        dets = [
            Detection(
                bbox=tuple(d.get("bbox", (0, 0, 0, 0))),
                centroid=tuple(d["centroid"]),
                area=float(d.get("area", 0.0)),
                label=d.get("label", "object"),
            )
            for d in detections
        ]
    else:
        from .detection import detect_by_color

        dets = detect_by_color(
            frame,
            lower_hsv or [0, 70, 50],
            upper_hsv or [10, 255, 255],
            min_area=min_area,
        )

    counts = count_in_zones(dets, zones)
    return {
        "routine": "zone_counting",
        "summary": {"zone_counts": counts, "total": int(sum(counts.values()))},
        "detections": [d.to_dict() for d in dets],
    }
