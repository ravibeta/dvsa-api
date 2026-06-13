"""Unsupervised spatial clustering of points / detections.

Implements the DBSCAN / HDBSCAN pattern-detection analytic from
``plan-of-action.md`` §5.1. Given a set of 2-D points (typically detection
centroids), it groups them into density-based clusters and reports per-cluster
centroids and sizes — useful for crowd/density estimation and anomaly (noise)
flagging. Both algorithms ship with ``scikit-learn`` (>=1.3), so no extra
dependency is required.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import register


def cluster_points(
    points: Sequence[Sequence[float]],
    algorithm: str = "dbscan",
    eps: float = 50.0,
    min_samples: int = 5,
    min_cluster_size: int = 5,
) -> dict:
    """Cluster 2-D points and summarise the result.

    Parameters
    ----------
    algorithm:
        ``"dbscan"`` or ``"hdbscan"``.
    eps, min_samples:
        DBSCAN parameters (``eps`` is the neighbourhood radius in pixels).
    min_cluster_size:
        HDBSCAN parameter.

    Returns a dict with ``labels``, ``n_clusters``, ``n_noise`` and a list of
    per-cluster ``{label, size, centroid}`` summaries.
    """

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return {"labels": [], "n_clusters": 0, "n_noise": 0, "clusters": []}
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array of [x, y]")

    algo = algorithm.lower()
    if algo == "dbscan":
        from sklearn.cluster import DBSCAN

        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algo == "hdbscan":
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(min_cluster_size=min_cluster_size)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")

    labels = model.fit_predict(pts)

    clusters: List[dict] = []
    for lbl in sorted(set(labels)):
        if lbl == -1:  # noise
            continue
        members = pts[labels == lbl]
        clusters.append(
            {
                "label": int(lbl),
                "size": int(members.shape[0]),
                "centroid": [float(members[:, 0].mean()), float(members[:, 1].mean())],
            }
        )

    n_noise = int(np.count_nonzero(labels == -1))
    return {
        "labels": [int(v) for v in labels],
        "n_clusters": len(clusters),
        "n_noise": n_noise,
        "clusters": clusters,
    }


@register(
    "spatial_clustering",
    "Density-based clustering (DBSCAN/HDBSCAN) of detection centroids.",
    level="frame",
)
def spatial_clustering_routine(
    frame: np.ndarray,
    points: Optional[Sequence[Sequence[float]]] = None,
    detections: Optional[Sequence[dict]] = None,
    algorithm: str = "dbscan",
    eps: float = 50.0,
    min_samples: int = 5,
    min_cluster_size: int = 5,
    lower_hsv: Optional[Sequence[int]] = None,
    upper_hsv: Optional[Sequence[int]] = None,
    min_area: float = 100.0,
    **_: object,
) -> dict:
    """Cluster supplied points/detections, or auto-detect centroids by color."""

    if points is None:
        if detections is not None:
            points = [d["centroid"] for d in detections]
        else:
            from .detection import detect_by_color

            dets = detect_by_color(
                frame,
                lower_hsv or [0, 70, 50],
                upper_hsv or [10, 255, 255],
                min_area=min_area,
            )
            points = [d.centroid for d in dets]

    result = cluster_points(
        points,
        algorithm=algorithm,
        eps=eps,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    )
    return {
        "routine": "spatial_clustering",
        "summary": {
            "algorithm": algorithm,
            "n_points": len(points),
            "n_clusters": result["n_clusters"],
            "n_noise": result["n_noise"],
        },
        "clusters": result["clusters"],
    }
