"""Homography estimation and geometric transforms.

Implements the perspective / geo-mapping analytic from ``plan-of-action.md``
§6.3: estimate a homography between image pixels and a reference plane (e.g.
ground / map coordinates), then warp imagery or map detection points into that
plane for spatial reasoning and measurement.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import register


def estimate_homography(
    src_points: Sequence[Sequence[float]],
    dst_points: Sequence[Sequence[float]],
    ransac_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a 3x3 homography mapping ``src_points`` -> ``dst_points``.

    Requires at least four non-collinear correspondences. Returns
    ``(H, mask)`` where ``mask`` flags RANSAC inliers.
    """

    import cv2

    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)
    if src.shape[0] < 4 or dst.shape[0] < 4:
        raise ValueError("Homography needs at least 4 point correspondences")
    if src.shape != dst.shape:
        raise ValueError("src_points and dst_points must have the same shape")

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    if H is None:
        raise ValueError("Homography estimation failed (degenerate correspondences)")
    return H, mask


def map_points(H: np.ndarray, points: Sequence[Sequence[float]]) -> List[List[float]]:
    """Apply homography ``H`` to a list of 2-D points."""

    import cv2

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64))
    return mapped.reshape(-1, 2).tolist()


def warp_perspective(
    frame: np.ndarray, H: np.ndarray, size: Tuple[int, int]
) -> np.ndarray:
    """Warp ``frame`` by homography ``H`` into an output of ``size`` (w, h)."""

    import cv2

    return cv2.warpPerspective(frame, np.asarray(H, dtype=np.float64), size)


@register(
    "homography_map",
    "Estimate a homography from correspondences and map points into the plane.",
    level="frame",
)
def homography_routine(
    frame: np.ndarray,
    src_points: Optional[Sequence[Sequence[float]]] = None,
    dst_points: Optional[Sequence[Sequence[float]]] = None,
    map_pts: Optional[Sequence[Sequence[float]]] = None,
    ransac_threshold: float = 3.0,
    **_: object,
) -> dict:
    """Estimate H from ``src_points``/``dst_points`` and optionally map points.

    If ``map_pts`` is omitted, the four image corners are mapped so callers can
    see where the frame lands in the destination plane.
    """

    if not src_points or not dst_points:
        raise ValueError("homography_map requires src_points and dst_points")

    H, mask = estimate_homography(src_points, dst_points, ransac_threshold)

    if map_pts is None:
        h, w = frame.shape[:2]
        map_pts = [[0, 0], [w, 0], [w, h], [0, h]]

    mapped = map_points(H, map_pts)
    return {
        "routine": "homography_map",
        "summary": {
            "inliers": int(mask.sum()) if mask is not None else None,
            "n_correspondences": len(src_points),
        },
        "homography": H.tolist(),
        "mapped_points": mapped,
    }
