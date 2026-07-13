"""Deterministic helpers shared by every kit.

All functions here are pure and reproducible: given the same inputs they return
the same outputs (no wall-clock, no RNG without an explicit seed). This is what
lets partitioned/orchestrated runs merge to a single canonical result.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence

from .schemas import Detection


def deterministic_run_id(*parts: Any, prefix: str = "run") -> str:
    """Return a stable run id derived from ``parts``.

    The same logical inputs always yield the same id, so re-running a partition
    is idempotent and outputs can be correlated across retries.
    """
    canonical = json.dumps([_stringify(p) for p in parts], sort_keys=True,
                           separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _stringify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    return value


def stable_sort_detections(detections: Sequence[Detection]) -> List[Detection]:
    """Sort detections by (timestamp, class_name, -confidence, bbox) — total order."""
    return sorted(
        detections,
        key=lambda d: (d.timestamp, d.class_name, -d.confidence, tuple(d.bbox)),
    )


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two ``[x, y, w, h]`` boxes (0.0–1.0)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def dedupe_detections(
    detections: Iterable[Detection],
    *,
    iou_threshold: float = 0.5,
    same_class_only: bool = True,
) -> List[Detection]:
    """Remove overlapping duplicates, keeping the highest-confidence detection.

    Two detections are duplicates when their boxes overlap above ``iou_threshold``
    (and, when ``same_class_only``, share a class). Deterministic: input is stably
    sorted first, then a greedy highest-confidence-wins pass runs.
    """
    ordered = stable_sort_detections(list(detections))
    kept: List[Detection] = []
    for cand in sorted(ordered, key=lambda d: (-d.confidence, d.timestamp, d.class_name)):
        duplicate = False
        for keep in kept:
            if same_class_only and keep.class_name != cand.class_name:
                continue
            if abs(keep.timestamp - cand.timestamp) > 1e-9:
                continue
            if iou(keep.bbox, cand.bbox) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return stable_sort_detections(kept)


def merge_detection_sets(
    sets: Iterable[Sequence[Detection]],
    *,
    iou_threshold: float = 0.5,
) -> List[Detection]:
    """Deterministically merge many detection lists into one canonical list."""
    combined: List[Detection] = []
    for group in sets:
        combined.extend(group)
    return dedupe_detections(combined, iou_threshold=iou_threshold)


def summarize_detections(detections: Sequence[Detection]) -> Dict[str, Any]:
    """Build a compact, deterministic summary (counts + span) of detections."""
    by_class: Dict[str, int] = {}
    for det in detections:
        by_class[det.class_name] = by_class.get(det.class_name, 0) + 1
    timestamps = [d.timestamp for d in detections]
    return {
        "total_detections": len(detections),
        "by_class": dict(sorted(by_class.items())),
        "time_span": [min(timestamps), max(timestamps)] if timestamps else [0.0, 0.0],
    }


__all__ = [
    "deterministic_run_id",
    "stable_sort_detections",
    "iou",
    "dedupe_detections",
    "merge_detection_sets",
    "summarize_detections",
]
