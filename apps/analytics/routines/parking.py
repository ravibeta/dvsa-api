"""Parking-spot occupancy detection.

Implements the "mask-based segmentation + SVM" parking analytic from
``plan-of-action.md`` §4.2 / §18. Each parking spot is a fixed rectangle. For
every spot we crop the ROI and compute a small feature vector (edge density,
mean intensity, intensity variance). Occupancy is decided either by a trained
``scikit-learn`` classifier (preferred, §18) or, when none is supplied, by an
edge-density threshold heuristic so the routine runs out of the box.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import register, to_gray

# A spot is (x, y, w, h) in pixels.
Spot = Tuple[int, int, int, int]


def spot_features(frame: np.ndarray, spot: Spot) -> np.ndarray:
    """Compute the [edge_density, mean_intensity, std_intensity] feature vector.

    A high edge density and intensity variance typically indicate a vehicle
    occupies the spot, whereas empty tarmac is smooth and low-variance.
    """

    import cv2

    x, y, w, h = (int(v) for v in spot)
    gray = to_gray(frame)
    roi = gray[max(y, 0):y + h, max(x, 0):x + w]
    if roi.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    edges = cv2.Canny(roi, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(roi.size)
    return np.array(
        [edge_density, float(roi.mean()), float(roi.std())], dtype=float
    )


def train_occupancy_classifier(features: Sequence[Sequence[float]], labels: Sequence[int]):
    """Train and return a scikit-learn SVM occupancy classifier.

    ``features`` rows come from :func:`spot_features`; ``labels`` are
    ``1`` (occupied) / ``0`` (empty). Features are standardised inside a
    pipeline so callers only need to feed raw vectors at inference time.
    """

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
    clf.fit(np.asarray(features, dtype=float), np.asarray(labels))
    return clf


def classify_spots(
    frame: np.ndarray,
    spots: Sequence[Spot],
    classifier=None,
    edge_threshold: float = 0.08,
) -> List[dict]:
    """Classify every spot as occupied/empty.

    Returns a list of ``{"spot": [x,y,w,h], "occupied": bool, "score": float}``.
    """

    results: List[dict] = []
    feats = np.array([spot_features(frame, s) for s in spots], dtype=float)

    if classifier is not None and len(spots) > 0:
        preds = classifier.predict(feats)
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(feats)
            scores = proba[:, list(classifier.classes_).index(1)] \
                if 1 in classifier.classes_ else proba.max(axis=1)
        else:
            scores = preds.astype(float)
        for spot, occ, sc in zip(spots, preds, scores):
            results.append(
                {"spot": [int(v) for v in spot], "occupied": bool(occ), "score": float(sc)}
            )
    else:
        # Heuristic fallback: edge density is feature column 0.
        for spot, f in zip(spots, feats):
            density = float(f[0])
            results.append(
                {
                    "spot": [int(v) for v in spot],
                    "occupied": bool(density >= edge_threshold),
                    "score": density,
                }
            )
    return results


@register(
    "parking_occupancy",
    "Classify parking-spot occupancy (SVM if trained, else edge heuristic).",
    level="frame",
)
def parking_occupancy_routine(
    frame: np.ndarray,
    spots: Optional[Sequence[Spot]] = None,
    edge_threshold: float = 0.08,
    classifier=None,
    **_: object,
) -> dict:
    if not spots:
        raise ValueError("parking_occupancy requires a list of spots [(x,y,w,h), ...]")

    spot_results = classify_spots(
        frame, spots, classifier=classifier, edge_threshold=edge_threshold
    )
    occupied = sum(1 for s in spot_results if s["occupied"])
    total = len(spot_results)
    return {
        "routine": "parking_occupancy",
        "summary": {
            "total_spots": total,
            "occupied": occupied,
            "available": total - occupied,
            "occupancy_rate": round(occupied / total, 4) if total else 0.0,
        },
        "spots": spot_results,
    }
