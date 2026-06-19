"""``custom_models`` — a pluggable, multi-format model-selection layer for dvsa-api.

This package lets the repo *choose* among several curated drone-detection models
(VisDrone YOLOv8, TPH-YOLOv5, DOTA Faster R-CNN, Ultralytics COCO, …) and run
any of them behind a single detector interface:

    detector = get_detector(spec)        # spec: ModelSpec or catalog id
    detector.load()
    detections = detector.infer(frame)   # [{"label","score","bbox":[x,y,w,h]}]
    detector.close()

Every adapter returns ``bbox`` as ``(x, y, w, h)`` in original-frame pixels, so
detections map straight onto ``apps.analytics.routines.base.Detection``.

The existing, proven single-format ``custom_model`` (singular) ONNX package is
left untouched; the ONNX adapter here delegates to it.
"""

from __future__ import annotations

from .loader import ModelSpec, discover_model, load_label_map
from .registry import available_formats, get_adapter_class, get_detector, register
from .selector import ModelSelector, SelectionQuery

__all__ = [
    "ModelSpec",
    "discover_model",
    "load_label_map",
    "get_detector",
    "get_adapter_class",
    "available_formats",
    "register",
    "ModelSelector",
    "SelectionQuery",
]
