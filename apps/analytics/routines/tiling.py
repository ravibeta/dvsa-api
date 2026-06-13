"""SAHI-style frame slicing and detection merging.

Implements the Slicing Aided Hyper Inference idea from ``plan-of-action.md``
§13: split a high-resolution frame into overlapping tiles so small,
high-altitude objects survive downsampling, run a (pluggable) per-tile detector,
then merge the tile-local detections back into full-frame coordinates with
non-maximum suppression to remove duplicates from the overlap regions.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .base import Detection, register


def slice_frame(
    frame: np.ndarray,
    tile_size: Tuple[int, int] = (640, 640),
    overlap: float = 0.2,
) -> List[dict]:
    """Slice ``frame`` into overlapping tiles.

    Returns a list of ``{"x": int, "y": int, "tile": ndarray}`` where ``x, y``
    is the tile's top-left offset in the full frame.
    """

    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    h, w = frame.shape[:2]
    tw, th = tile_size
    step_x = max(1, int(tw * (1 - overlap)))
    step_y = max(1, int(th * (1 - overlap)))

    tiles: List[dict] = []
    y = 0
    while y < h:
        x = 0
        # Clamp the final row/column so tiles stay inside the frame.
        y0 = min(y, max(0, h - th))
        while x < w:
            x0 = min(x, max(0, w - tw))
            tile = frame[y0:y0 + th, x0:x0 + tw]
            tiles.append({"x": int(x0), "y": int(y0), "tile": tile})
            if x0 + tw >= w:
                break
            x += step_x
        if y0 + th >= h:
            break
        y += step_y
    return tiles


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections: Sequence[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Greedy NMS over :class:`Detection` objects (highest score kept)."""

    dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for det in dets:
        if all(_iou(det.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def merge_tile_detections(
    tile_results: Sequence[Tuple[int, int, Sequence[Detection]]],
    iou_threshold: float = 0.45,
) -> List[Detection]:
    """Offset tile-local detections to full-frame coords and run NMS.

    ``tile_results`` is a sequence of ``(offset_x, offset_y, detections)``.
    """

    merged: List[Detection] = []
    for ox, oy, dets in tile_results:
        for d in dets:
            x, y, w, h = d.bbox
            merged.append(
                Detection(
                    bbox=(int(x + ox), int(y + oy), int(w), int(h)),
                    centroid=(d.centroid[0] + ox, d.centroid[1] + oy),
                    area=d.area,
                    label=d.label,
                    score=d.score,
                )
            )
    return non_max_suppression(merged, iou_threshold=iou_threshold)


def sliced_detection(
    frame: np.ndarray,
    detector: Callable[[np.ndarray], Sequence[Detection]],
    tile_size: Tuple[int, int] = (640, 640),
    overlap: float = 0.2,
    iou_threshold: float = 0.45,
) -> List[Detection]:
    """Run ``detector`` over every tile and merge results to full-frame coords.

    ``detector`` takes a tile (ndarray) and returns a sequence of
    :class:`Detection` in tile-local coordinates.
    """

    tile_results = []
    for t in slice_frame(frame, tile_size=tile_size, overlap=overlap):
        dets = detector(t["tile"])
        tile_results.append((t["x"], t["y"], list(dets)))
    return merge_tile_detections(tile_results, iou_threshold=iou_threshold)


@register(
    "sliced_detection",
    "SAHI-style tiled color detection for small/high-altitude objects.",
    level="frame",
)
def sliced_detection_routine(
    frame: np.ndarray,
    tile_size: Optional[Sequence[int]] = None,
    overlap: float = 0.2,
    iou_threshold: float = 0.45,
    lower_hsv: Optional[Sequence[int]] = None,
    upper_hsv: Optional[Sequence[int]] = None,
    min_area: float = 30.0,
    **_: object,
) -> dict:
    """Tiled wrapper around the classical color detector (the pluggable default).

    A real deployment would pass a YOLO/Detectron tile detector here; the color
    detector keeps the routine runnable with the installed dependency set.
    """

    from .detection import detect_by_color

    ts = tuple(tile_size) if tile_size else (640, 640)

    def _detector(tile: np.ndarray):
        return detect_by_color(
            tile, lower_hsv or [0, 70, 50], upper_hsv or [10, 255, 255], min_area=min_area
        )

    dets = sliced_detection(
        frame, _detector, tile_size=ts, overlap=overlap, iou_threshold=iou_threshold
    )
    return {
        "routine": "sliced_detection",
        "summary": {"count": len(dets), "tile_size": list(ts), "overlap": overlap},
        "detections": [d.to_dict() for d in dets],
    }
