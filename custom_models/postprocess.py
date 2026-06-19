"""Shared detection post-processing for the ``custom_models`` adapters.

Every adapter ultimately produces the same normalised detection dict::

    {"label": "vehicle", "score": 0.92, "bbox": [x, y, w, h]}

where ``bbox`` is ``(x, y, w, h)`` — top-left corner plus width/height — in
**original frame pixel space**, matching
``apps.analytics.routines.base.Detection`` used throughout dvsa-api.

This module centralises the box scaling/formatting so the Torch and YOLO
adapters stay tiny. The corner→``xywh`` conversion and greedy NMS are imported
from the already-tested :mod:`custom_model.onnx_inference` to avoid duplicating
proven code.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Re-use the proven helpers from the existing ONNX adapter.
from custom_model.onnx_inference import _iou as iou  # noqa: F401  (re-exported)
from custom_model.onnx_inference import _nms as nms
from custom_model.onnx_inference import xywh_to_xyxy, xyxy_to_xywh

__all__ = ["to_detections", "nms", "iou", "xyxy_to_xywh", "xywh_to_xyxy"]

# Below this magnitude every coordinate is treated as normalised to [0, 1]
# rather than expressed in input-pixel space (mirrors custom_model's heuristic).
_NORMALISED_COORD_MAX = 1.5

LabelMap = Union[Mapping[int, str], Sequence[str], None]


def _label_for(label_map: LabelMap, cls: int) -> str:
    """Resolve a class id to a label, tolerating dict / list / ``None`` maps."""

    if label_map is None:
        return str(cls)
    if isinstance(label_map, Mapping):
        if cls in label_map:
            return str(label_map[cls])
        # Ultralytics ``model.names`` may use string keys.
        if str(cls) in label_map:  # type: ignore[operator]
            return str(label_map[str(cls)])  # type: ignore[index]
        return str(cls)
    # Sequence
    if 0 <= cls < len(label_map):
        return str(label_map[cls])
    return str(cls)


def to_detections(
    boxes,
    scores,
    labels,
    label_map: LabelMap,
    *,
    frame_hw: Tuple[int, int],
    region: Optional[Tuple[int, int, int, int]] = None,
    model_input_size: Optional[Tuple[int, int]] = None,
    score_threshold: float = 0.0,
) -> List[Dict]:
    """Map raw model arrays onto normalised detection dicts.

    Parameters
    ----------
    boxes:
        ``(N, 4)`` array-like of corner boxes ``[x1, y1, x2, y2]``.
    scores:
        ``(N,)`` confidence scores.
    labels:
        ``(N,)`` integer (or float) class ids.
    label_map:
        ``{class_id: label}`` mapping, a list of names, or ``None``.
    frame_hw:
        ``(height, width)`` of the **original** frame.
    region:
        ``(x_off, y_off, w, h)`` of the region (tile) the boxes came from, in
        original-frame pixels. ``None`` => the whole frame.
    model_input_size:
        ``(in_w, in_h)`` the model consumed. When given, boxes are scaled from
        input/normalised space back to the region. When ``None`` the boxes are
        assumed to already be in original-frame pixel coordinates (the common
        case for Ultralytics YOLO, which rescales internally).
    score_threshold:
        Minimum score to keep.
    """

    H, W = int(frame_hw[0]), int(frame_hw[1])
    if region is None:
        rx, ry, rw, rh = 0, 0, W, H
    else:
        rx, ry, rw, rh = (int(region[0]), int(region[1]), int(region[2]), int(region[3]))

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    dets: List[Dict] = []
    for box, score, cls_raw in zip(boxes, scores, labels):
        s = float(score)
        if s < score_threshold:
            continue

        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if model_input_size is not None:
            in_w, in_h = model_input_size
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= _NORMALISED_COORD_MAX:
                sx, sy = rw, rh  # normalised [0, 1] -> region size
            else:
                sx, sy = rw / float(in_w), rh / float(in_h)  # input-px -> region size
            x1, y1 = rx + x1 * sx, ry + y1 * sy
            x2, y2 = rx + x2 * sx, ry + y2 * sy

        x1, x2 = sorted((max(0.0, x1), max(0.0, x2)))
        y1, y2 = sorted((max(0.0, y1), max(0.0, y2)))

        cls = int(round(float(cls_raw)))
        dets.append(
            {
                "label": _label_for(label_map, cls),
                "score": s,
                "bbox": xyxy_to_xywh((x1, y1, x2, y2)),
            }
        )
    return dets
