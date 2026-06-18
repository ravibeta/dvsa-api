"""dvsa-api wiring for the custom ONNX detector (``custom_model/``).

This module bridges the standalone :mod:`custom_model` adapter into the analytics
routine registry. It:

* builds a :class:`~custom_model.model_loader.ModelConfig` from environment
  variables (so deployments configure the model without code changes),
* lazily constructs a single, process-wide loaded detector, and
* registers a frame-level routine ``"custom_onnx_detection"`` that maps the
  adapter's detection dicts (``bbox = [x, y, w, h]``) onto the repo's
  :class:`~apps.analytics.routines.base.Detection` / :class:`RoutineResult`.

Configuration (environment variables)
-------------------------------------
``CUSTOM_MODEL_ONNX_PATH``     Path to the ``.onnx`` model (required to enable).
``CUSTOM_MODEL_LABELS_PATH``   Path to ``label_map.json`` (required to enable).
``CUSTOM_MODEL_INPUT_SIZE``    ``WxH`` (default ``640x640``).
``CUSTOM_MODEL_MEAN``          ``r,g,b`` (default ``0.485,0.456,0.406``).
``CUSTOM_MODEL_STD``           ``r,g,b`` (default ``0.229,0.224,0.225``).
``CUSTOM_MODEL_TILE_SIZE``     ``WxH`` to enable tiling (default: disabled).
``CUSTOM_MODEL_TILE_OVERLAP``  Fraction in ``[0, 1)`` (default ``0.2``).
``CUSTOM_MODEL_SCORE_THRESHOLD``  Minimum score to keep (default ``0.0``).

When ``CUSTOM_MODEL_ONNX_PATH`` / ``CUSTOM_MODEL_LABELS_PATH`` are unset, the
routine is still registered but raises a clear ``RuntimeError`` if invoked, so
existing analytics are unaffected.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Tuple

from .base import Detection, RoutineResult, register

logger = logging.getLogger("apps.analytics")

ROUTINE_NAME = "custom_onnx_detection"


# --------------------------------------------------------------------------- #
# Environment parsing
# --------------------------------------------------------------------------- #


def _parse_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse ``"WxH"`` into ``(W, H)``; ``None``/empty -> ``None``."""

    if not value:
        return None
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Expected a 'WxH' size, got {value!r}") from exc


def _parse_triplet(value: Optional[str], default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Parse ``"r,g,b"`` floats; ``None``/empty -> ``default``."""

    if not value:
        return default
    parts = [p for p in value.split(",") if p.strip() != ""]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated floats, got {value!r}")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def build_config_from_env(env: Optional[Mapping[str, str]] = None):
    """Build a :class:`ModelConfig` from environment variables.

    Returns ``None`` when the model is not configured (no ONNX/labels paths),
    signalling that the custom detector is unavailable.
    """

    # Imported here to keep this module importable (and the registry populated)
    # even in environments where custom_model's optional deps are absent.
    from custom_model.model_loader import ModelConfig

    env = env if env is not None else os.environ

    onnx_path = env.get("CUSTOM_MODEL_ONNX_PATH")
    labels_path = env.get("CUSTOM_MODEL_LABELS_PATH")
    if not onnx_path or not labels_path:
        return None

    tile_size = _parse_size(env.get("CUSTOM_MODEL_TILE_SIZE"))
    return ModelConfig(
        onnx_path=onnx_path,
        labels_path=labels_path,
        input_size=_parse_size(env.get("CUSTOM_MODEL_INPUT_SIZE")) or (640, 640),
        mean=_parse_triplet(env.get("CUSTOM_MODEL_MEAN"), (0.485, 0.456, 0.406)),
        std=_parse_triplet(env.get("CUSTOM_MODEL_STD"), (0.229, 0.224, 0.225)),
        tile_size=tile_size,
        tile_overlap=float(env.get("CUSTOM_MODEL_TILE_OVERLAP", "0.2")),
        score_threshold=float(env.get("CUSTOM_MODEL_SCORE_THRESHOLD", "0.0")),
    )


# --------------------------------------------------------------------------- #
# Detector singleton
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_custom_detector():
    """Return a loaded :class:`CustomONNXDetector`, or ``None`` if unconfigured.

    The detector is built once and cached for the process. Call
    :func:`reset_custom_detector` to force a rebuild (e.g. after changing env).
    """

    from custom_model.model_loader import create_detector

    config = build_config_from_env()
    if config is None:
        logger.info("Custom ONNX model not configured (set CUSTOM_MODEL_ONNX_PATH/LABELS_PATH)")
        return None

    logger.info("Loading custom ONNX detector from %s", config.onnx_path)
    detector = create_detector(config)
    detector.load()
    return detector


def reset_custom_detector() -> None:
    """Clear the cached detector so the next call rebuilds it from env."""

    get_custom_detector.cache_clear()


# --------------------------------------------------------------------------- #
# Mapping adapter detections -> repo Detection / RoutineResult
# --------------------------------------------------------------------------- #


def _detection_from_dict(det: Dict) -> Detection:
    """Convert one adapter dict ``{label, score, bbox:[x,y,w,h]}`` to a Detection."""

    x, y, w, h = det["bbox"]  # adapter already returns (x, y, w, h)
    return Detection(
        bbox=(int(x), int(y), int(w), int(h)),
        centroid=(x + w / 2.0, y + h / 2.0),
        area=float(w) * float(h),
        label=str(det.get("label", "object")),
        score=float(det.get("score", 1.0)),
    )


def build_routine_result(raw_detections: List[Dict]) -> dict:
    """Wrap adapter detections in the standard :class:`RoutineResult` envelope."""

    detections = [_detection_from_dict(d) for d in raw_detections]
    labels = sorted({d.label for d in detections})
    return RoutineResult(
        routine=ROUTINE_NAME,
        summary={"count": len(detections), "labels": labels},
        detections=detections,
    ).to_dict()


# --------------------------------------------------------------------------- #
# Registered routine
# --------------------------------------------------------------------------- #


@register(
    ROUTINE_NAME,
    "Run a custom ONNX object-detection model (configured via env vars).",
    level="frame",
)
def custom_onnx_detection_routine(frame, **_: object) -> dict:
    """Frame-level routine that dispatches to the env-configured ONNX detector.

    Raises a clear ``RuntimeError`` (caught per-frame by the analysis task) if
    the model has not been configured via ``CUSTOM_MODEL_ONNX_PATH`` /
    ``CUSTOM_MODEL_LABELS_PATH``.
    """

    detector = get_custom_detector()
    if detector is None:
        raise RuntimeError(
            "Custom ONNX model is not configured. Set CUSTOM_MODEL_ONNX_PATH and "
            "CUSTOM_MODEL_LABELS_PATH (see custom_model/README.md)."
        )
    return build_routine_result(detector.infer(frame))
