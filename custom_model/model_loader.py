"""Configuration, label handling and the detector factory for custom ONNX models.

This module is intentionally free of heavy imports (``onnxruntime``, ``cv2``)
so it can be imported cheaply for configuration/validation purposes. The actual
detector implementation lives in :mod:`custom_model.onnx_inference` and is
imported lazily by :func:`create_detector`.

Typical usage
-------------
>>> from custom_model.model_loader import ModelConfig, create_detector
>>> cfg = ModelConfig(
...     onnx_path="/models/custom_model.onnx",
...     labels_path="/models/label_map.json",
...     input_size=(640, 640),
...     mean=(0.485, 0.456, 0.406),
...     std=(0.229, 0.224, 0.225),
... )
>>> detector = create_detector(cfg)
>>> detector.load()              # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a heavy import at runtime
    from .onnx_inference import CustomONNXDetector

logger = logging.getLogger(__name__)

# A session factory takes the ONNX model path and returns an object exposing the
# onnxruntime.InferenceSession API (``run``, ``get_inputs``, ``get_outputs``).
SessionFactory = Callable[[str], object]


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------


def load_label_map(path: str) -> Dict[int, str]:
    """Load a ``{class_id: label}`` mapping from a JSON file.

    The JSON keys may be strings (as produced by ``json.dump``) or integers;
    they are normalised to ``int``. Values must be strings.

    Parameters
    ----------
    path:
        Path to the ``label_map.json`` file.

    Returns
    -------
    dict
        Mapping of integer class id to label string.

    Raises
    ------
    ValueError
        If the file is missing, is not valid JSON, is not a JSON object, or
        contains keys that cannot be coerced to ``int``.
    """

    if not path or not os.path.isfile(path):
        raise ValueError(f"Label map not found: {path!r}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Label map {path!r} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Label map {path!r} must be a JSON object of "
            f"{{class_id: label}}, got {type(raw).__name__}"
        )

    label_map: Dict[int, str] = {}
    for key, value in raw.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Label map {path!r} has a non-integer class id {key!r}"
            ) from exc
        label_map[class_id] = str(value)

    if not label_map:
        raise ValueError(f"Label map {path!r} is empty")

    logger.debug("Loaded %d labels from %s", len(label_map), path)
    return label_map


def validate_model_file(path: str) -> None:
    """Validate that ``path`` points to an existing ``.onnx`` file.

    Raises
    ------
    ValueError
        If the path is empty or does not have a ``.onnx`` extension.
    FileNotFoundError
        If the file does not exist on disk.
    """

    if not path:
        raise ValueError("onnx_path must be a non-empty path")
    if not path.lower().endswith(".onnx"):
        raise ValueError(f"Model file must have a .onnx extension: {path!r}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ONNX model file not found: {path!r}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration describing a custom ONNX detector and its preprocessing.

    Attributes
    ----------
    onnx_path:
        Filesystem path to the exported ``.onnx`` model.
    labels_path:
        Filesystem path to the ``label_map.json`` (``{class_id: label}``).
    input_size:
        ``(width, height)`` the model expects, in pixels.
    mean, std:
        Per-channel (R, G, B) normalisation applied after scaling pixels to
        ``[0, 1]``: ``(pixel/255 - mean) / std``. Use ``mean=(0,0,0)`` and
        ``std=(1,1,1)`` for plain ``[0, 1]`` scaling.
    tile_size:
        Optional ``(width, height)`` of tiles for large aerial frames. When
        set, the frame is split into overlapping tiles, each inferred
        independently and the detections merged. ``None`` disables tiling.
    tile_overlap:
        Fractional overlap between adjacent tiles in ``[0, 1)`` (e.g. ``0.2``).
    score_threshold:
        Minimum score for a detection to be kept.
    iou_threshold:
        IoU threshold used by NMS when merging tiled detections.
    """

    onnx_path: str
    labels_path: str
    input_size: Tuple[int, int] = (640, 640)
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    tile_size: Optional[Tuple[int, int]] = None
    tile_overlap: float = 0.0
    score_threshold: float = 0.0
    iou_threshold: float = 0.5
    # Optional explicit ONNX output ordering; see onnx_inference.postprocess.
    output_order: Optional[Tuple[str, str, str]] = field(default=None)

    def __post_init__(self) -> None:
        if len(self.input_size) != 2:
            raise ValueError(f"input_size must be (width, height), got {self.input_size!r}")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must each be 3-tuples (R, G, B)")
        if any(s == 0 for s in self.std):
            raise ValueError("std components must be non-zero")
        if not (0.0 <= self.tile_overlap < 1.0):
            raise ValueError("tile_overlap must be in [0, 1)")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_detector(
    config: ModelConfig,
    session_factory: Optional[SessionFactory] = None,
) -> "CustomONNXDetector":
    """Instantiate a :class:`~custom_model.onnx_inference.CustomONNXDetector`.

    Parameters
    ----------
    config:
        The :class:`ModelConfig` describing the model.
    session_factory:
        Optional callable ``(onnx_path) -> session`` used to build the ONNX
        runtime session. Injecting a factory is the recommended way to supply
        a fake/mock session in tests. When ``None``, the detector creates a
        real ``onnxruntime.InferenceSession`` at :meth:`load` time.

    Returns
    -------
    CustomONNXDetector
        An *unloaded* detector. Call :meth:`load` before :meth:`infer`.
    """

    # Imported here (not at module top) to keep model_loader free of the heavy
    # onnxruntime/cv2 imports and to avoid a circular import.
    from .onnx_inference import CustomONNXDetector

    logger.debug("Creating CustomONNXDetector for %s", config.onnx_path)
    return CustomONNXDetector(config, session_factory=session_factory)
