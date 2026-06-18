"""Custom ONNX object-detection adapter for dvsa-api.

Public API
----------
* :class:`~custom_model.model_loader.ModelConfig`
* :func:`~custom_model.model_loader.create_detector`
* :func:`~custom_model.model_loader.load_label_map`
* :class:`~custom_model.onnx_inference.CustomONNXDetector`
"""

from .model_loader import (
    ModelConfig,
    create_detector,
    load_label_map,
    validate_model_file,
)
from .onnx_inference import CustomONNXDetector

__all__ = [
    "ModelConfig",
    "create_detector",
    "load_label_map",
    "validate_model_file",
    "CustomONNXDetector",
]
