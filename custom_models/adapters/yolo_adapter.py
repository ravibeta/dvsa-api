"""Ultralytics YOLO adapter — wraps ``ultralytics.YOLO`` for ``.pt`` or ONNX.

Ultralytics' :class:`~ultralytics.YOLO` auto-detects the artifact type (a
``.pt`` checkpoint or an exported ``.onnx``) and rescales detections back to the
source image internally, so this adapter is a thin wrapper that maps Ultralytics
``Results`` onto the common detection dicts
``{"label","score","bbox":[x,y,w,h]}`` in original-frame pixels.

Class names come from ``spec.labels_file`` when provided, otherwise from the
model's embedded ``model.names``.

The ``model_factory`` hook is injectable so the adapter can be unit-tested by
mocking ``ultralytics.YOLO`` (no real ``ultralytics``/``torch`` install needed).

CLI smoke test::

    python -m custom_models.adapters.yolo_adapter yolov8x.pt image.jpg
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from ..loader import ModelSpec, load_label_map
from ..postprocess import to_detections
from ..registry import register

logger = logging.getLogger(__name__)


@register("yolo")
class YOLOAdapter:
    """Run an Ultralytics YOLO model described by a :class:`ModelSpec`.

    Parameters
    ----------
    spec:
        The :class:`ModelSpec`; ``spec.path`` points at a ``.pt`` or ``.onnx``
        YOLO artifact. ``spec.labels_file`` is optional (YOLO embeds names).
    model_factory:
        Optional callable ``(path) -> model`` used by :meth:`load`. Defaults to
        ``ultralytics.YOLO``. Inject a stub to mock the model in tests.
    score_threshold:
        Minimum score to keep (also passed to YOLO as ``conf`` when supported).
    """

    format = "yolo"

    def __init__(
        self,
        spec: ModelSpec,
        *,
        model_factory: Optional[Callable[[str], object]] = None,
        score_threshold: float = 0.25,
    ) -> None:
        self.spec = spec
        self._model_factory = model_factory
        self._score_threshold = score_threshold
        self._model = None
        self._label_map = None

    # ------------------------------------------------------------------ #
    def load(self) -> "YOLOAdapter":
        """Construct the YOLO model and resolve the label map."""

        factory = self._model_factory or self._default_factory
        self._model = factory(self.spec.path)

        if self.spec.labels_file:
            self._label_map = load_label_map(self.spec.labels_file)
        else:
            # Fall back to the model's embedded class names.
            self._label_map = getattr(self._model, "names", None)
        logger.info("YOLOAdapter loaded model %s", self.spec.id)
        return self

    @staticmethod
    def _default_factory(path: str):  # pragma: no cover - requires ultralytics
        from ultralytics import YOLO

        return YOLO(path)

    def close(self) -> None:
        """Drop the model reference. Safe to call repeatedly."""

        self._model = None

    # ------------------------------------------------------------------ #
    def infer(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO over ``frame`` and return detection dicts."""

        if self._model is None:
            raise RuntimeError("YOLOAdapter not loaded; call load() first")
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be an HxWx3 BGR numpy array")

        h, w = frame.shape[:2]
        results = self._predict(frame)
        boxes, scores, labels = _parse_yolo_results(results)
        # Ultralytics returns boxes already in original-image pixel coords, so we
        # pass model_input_size=None (no rescaling) — only clip + xywh convert.
        return to_detections(
            boxes,
            scores,
            labels,
            self._label_map,
            frame_hw=(h, w),
            model_input_size=None,
            score_threshold=self._score_threshold,
        )

    def _predict(self, frame: np.ndarray):
        """Call the model, passing ``conf`` / ``verbose`` when the call accepts them."""

        try:
            return self._model(frame, conf=self._score_threshold, verbose=False)
        except TypeError:
            # Stub / older signature that doesn't take kwargs.
            return self._model(frame)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_array(value) -> np.ndarray:
    """Convert a torch tensor / list / ndarray to a numpy array (CPU)."""

    if value is None:
        return np.empty((0,))
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    npfn = getattr(value, "numpy", None)
    if callable(npfn):
        return np.asarray(npfn())
    return np.asarray(value)


def _parse_yolo_results(results):
    """Extract ``(boxes_xyxy, scores, classes)`` from Ultralytics ``Results``."""

    # ``model(frame)`` returns a list of Results; take the first image.
    result = results[0] if isinstance(results, (list, tuple)) else results
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    xyxy = _as_array(getattr(boxes_obj, "xyxy", None)).reshape(-1, 4)
    conf = _as_array(getattr(boxes_obj, "conf", None)).reshape(-1)
    cls = _as_array(getattr(boxes_obj, "cls", None)).reshape(-1)
    return xyxy, conf, cls


def _smoke_test(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    import cv2

    from ..loader import discover_model

    parser = argparse.ArgumentParser(description="YOLO adapter smoke test")
    parser.add_argument("model", help="path to a YOLO .pt or .onnx model")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--labels", default=None, help="optional label_map.json")
    args = parser.parse_args(argv)

    spec = discover_model(args.model, labels_file=args.labels)
    spec.format = "yolo"
    detector = YOLOAdapter(spec).load()
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read image: {args.image}")
        return 2
    for d in detector.infer(frame):
        print(f"  {d['label']:>12}  {d['score']:.3f}  bbox={d['bbox']}")
    detector.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_smoke_test())
