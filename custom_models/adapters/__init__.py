"""Concrete model adapters for the ``custom_models`` package.

Importing this package registers the built-in adapters (``onnx``, ``torch``,
``yolo``) with :mod:`custom_models.registry` via their ``@register(...)``
decorators.
"""

from __future__ import annotations

from .onnx_adapter import ONNXAdapter  # noqa: F401
from .torch_adapter import TorchAdapter  # noqa: F401
from .yolo_adapter import YOLOAdapter  # noqa: F401

__all__ = ["ONNXAdapter", "TorchAdapter", "YOLOAdapter"]
