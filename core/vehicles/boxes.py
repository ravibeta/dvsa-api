"""Pure-NumPy box helpers: IoU, greedy NMS and overlapping-tile windows (xyxy boxes)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy arrays -> (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices, highest score first."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        overlap = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][overlap <= iou_threshold]
    return np.asarray(keep, dtype=int)


def tile_windows(width: int, height: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) windows of side ``tile`` covering the image with ``overlap`` fraction."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(round(tile * (1.0 - overlap))))

    def starts(length: int) -> List[int]:
        if length <= tile:
            return [0]
        s = list(range(0, length - tile, stride)) + [length - tile]
        return sorted(set(s))

    return [(x0, y0, min(x0 + tile, width), min(y0 + tile, height))
            for y0 in starts(height) for x0 in starts(width)]


def touches_interior_edge(boxes: np.ndarray, window: Tuple[int, int, int, int], width: int, height: int,
                          margin: float = 2.0) -> np.ndarray:
    """Mask of boxes (absolute xyxy) touching a window edge that is not also an image edge.

    Such a tile detection is most likely a vehicle cut by the crop; the overlapping neighbour tile
    or the full-frame pass sees it whole.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x0, y0, x1, y1 = window
    m = np.zeros(len(boxes), dtype=bool)
    if len(boxes) == 0:
        return m
    if x0 > 0:
        m |= boxes[:, 0] <= x0 + margin
    if y0 > 0:
        m |= boxes[:, 1] <= y0 + margin
    if x1 < width:
        m |= boxes[:, 2] >= x1 - margin
    if y1 < height:
        m |= boxes[:, 3] >= y1 - margin
    return m
