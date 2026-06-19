"""ONNX adapter — a thin shim over the proven ``custom_model`` ONNX detector.

Rather than reimplement ONNX preprocessing / output-parsing / tiling / NMS, this
adapter delegates to :class:`custom_model.onnx_inference.CustomONNXDetector` via
:func:`custom_model.model_loader.create_detector`. It exposes the common
``load()`` / ``infer(frame)`` / ``close()`` interface used across
``custom_models`` and accepts a ``session_factory`` so the ONNX runtime session
can be mocked in tests (no real ``.onnx`` binary or ``onnxruntime`` install
required).

CLI smoke test::

    python -m custom_models.adapters.onnx_adapter model.onnx label_map.json image.jpg
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from ..loader import ModelSpec
from ..registry import register

logger = logging.getLogger(__name__)


@register("onnx")
class ONNXAdapter:
    """Run a custom ONNX detection model described by a :class:`ModelSpec`.

    Parameters
    ----------
    spec:
        The :class:`ModelSpec`. ``spec.path`` must point at a ``.onnx`` file and
        ``spec.labels_file`` at a ``label_map.json``.
    session_factory:
        Optional callable ``(onnx_path) -> session`` used to build (or mock) the
        ``onnxruntime`` session. When ``None`` a real CPU session is created at
        :meth:`load` time.
    score_threshold, tile_overlap, mean, std:
        Optional preprocessing overrides forwarded to the underlying
        ``ModelConfig``. ``tile_size`` is taken from ``spec.tile_recommendation``.
    """

    format = "onnx"

    def __init__(
        self,
        spec: ModelSpec,
        *,
        session_factory: Optional[Callable[[str], object]] = None,
        score_threshold: float = 0.0,
        tile_overlap: float = 0.2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ) -> None:
        self.spec = spec
        self._session_factory = session_factory
        self._score_threshold = score_threshold
        self._tile_overlap = tile_overlap
        self._mean = mean
        self._std = std
        self._detector = None

    def load(self) -> "ONNXAdapter":
        """Build the underlying ``CustomONNXDetector`` and load its labels."""

        from custom_model.model_loader import ModelConfig, create_detector

        if not self.spec.labels_file:
            raise ValueError(
                f"ONNX model {self.spec.id!r} has no labels_file; set ModelSpec.labels_file"
            )

        config = ModelConfig(
            onnx_path=self.spec.path,
            labels_path=self.spec.labels_file,
            input_size=self.spec.input_size,
            mean=self._mean,
            std=self._std,
            tile_size=self.spec.tile_recommendation,
            tile_overlap=self._tile_overlap,
            score_threshold=self._score_threshold,
        )
        self._detector = create_detector(config, session_factory=self._session_factory)
        self._detector.load()
        logger.info("ONNXAdapter loaded model %s", self.spec.id)
        return self

    def infer(self, frame: np.ndarray) -> List[Dict]:
        """Return detection dicts ``[{"label","score","bbox":[x,y,w,h]}]``."""

        if self._detector is None:
            raise RuntimeError("ONNXAdapter not loaded; call load() first")
        return self._detector.infer(frame)

    def close(self) -> None:
        """Release the underlying ONNX session. Safe to call repeatedly."""

        if self._detector is not None:
            self._detector.close()
        self._detector = None


def _smoke_test(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    import cv2

    from ..loader import discover_model

    parser = argparse.ArgumentParser(description="ONNX adapter smoke test")
    parser.add_argument("model", help="path to .onnx model")
    parser.add_argument("labels", help="path to label_map.json")
    parser.add_argument("image", help="path to an image file")
    args = parser.parse_args(argv)

    spec = discover_model(args.model, labels_file=args.labels)
    spec.format = "onnx"
    detector = ONNXAdapter(spec).load()
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read image: {args.image}")
        return 2
    dets = detector.infer(frame)
    print(f"{len(dets)} detection(s):")
    for d in dets:
        print(f"  {d['label']:>12}  {d['score']:.3f}  bbox={d['bbox']}")
    detector.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_smoke_test())
