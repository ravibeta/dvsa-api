"""RGB / HSV color histograms and color-summary descriptors.

Implements the lightweight color-analysis analytic from ``plan-of-action.md``
§6.1: compact per-channel histograms plus summary statistics suitable for scene
characterisation, land-cover hints and anomaly detection as a cheap
preprocessing step before heavier inference.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import register


def rgb_histogram(frame: np.ndarray, bins: int = 32) -> dict:
    """Compute a normalised per-channel histogram for a BGR frame.

    Returns histograms keyed by ``"b"``, ``"g"``, ``"r"`` (each summing to 1)
    plus the ``bins`` count and per-channel means.
    """

    import cv2

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("rgb_histogram expects a 3-channel BGR frame")

    hist = {}
    means = {}
    for i, ch in enumerate(("b", "g", "r")):
        h = cv2.calcHist([frame], [i], None, [bins], [0, 256]).flatten()
        total = float(h.sum())
        hist[ch] = (h / total).tolist() if total else h.tolist()
        means[ch] = float(frame[:, :, i].mean())

    return {"bins": bins, "histogram": hist, "channel_means": means}


def dominant_colors(frame: np.ndarray, k: int = 3) -> List[dict]:
    """Return the ``k`` dominant colors via k-means on the pixel colors.

    Each entry is ``{"color_bgr": [b,g,r], "fraction": float}`` sorted by
    descending fraction.
    """

    from sklearn.cluster import KMeans

    pixels = frame.reshape(-1, 3).astype(float)
    # Subsample for speed on large aerial frames.
    if pixels.shape[0] > 20000:
        idx = np.random.default_rng(0).choice(pixels.shape[0], 20000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(sample)
    labels = km.predict(pixels)
    counts = np.bincount(labels, minlength=k).astype(float)
    fractions = counts / counts.sum()

    out = [
        {
            "color_bgr": [int(round(v)) for v in km.cluster_centers_[i]],
            "fraction": float(fractions[i]),
        }
        for i in range(k)
    ]
    out.sort(key=lambda d: d["fraction"], reverse=True)
    return out


@register(
    "color_histogram",
    "Per-channel RGB histogram + dominant colors for scene characterisation.",
    level="frame",
)
def color_histogram_routine(
    frame: np.ndarray,
    bins: int = 32,
    dominant_k: int = 3,
    **_: object,
) -> dict:
    hist = rgb_histogram(frame, bins=bins)
    summary = {"channel_means": hist["channel_means"]}
    if dominant_k and dominant_k > 0:
        summary["dominant_colors"] = dominant_colors(frame, k=dominant_k)
    return {
        "routine": "color_histogram",
        "summary": summary,
        "histogram": hist["histogram"],
        "bins": bins,
    }
