"""DVSA classical computer-vision / ML routines.

A modular, pluggable collection of training-free vision analytics curated from
``plan-of-action.md`` (the routines whose dependencies — OpenCV, scikit-image,
scikit-learn, shapely — are already part of the project). Importing this package
populates the routine registry; use :func:`available_routines`,
:func:`get_routine` and :func:`run_frame_routine` to discover and dispatch them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import (
    Detection,
    RoutineResult,
    RoutineSpec,
    available_routines,
    get_routine,
    iter_frames,
    register,
    video_metadata,
)

# Import routine modules for their registration side effects.
from . import clustering  # noqa: F401
from . import custom_onnx  # noqa: F401
from . import detection  # noqa: F401
from . import geometry  # noqa: F401
from . import histograms  # noqa: F401
from . import motion  # noqa: F401
from . import parking  # noqa: F401
from . import tiling  # noqa: F401
from . import zone_counting  # noqa: F401


def run_frame_routine(name: str, frame: np.ndarray, **params) -> dict:
    """Dispatch a frame-level routine by name against a single frame."""

    spec = get_routine(name)
    return spec.func(frame, **params)


__all__ = [
    "Detection",
    "RoutineResult",
    "RoutineSpec",
    "available_routines",
    "get_routine",
    "iter_frames",
    "register",
    "video_metadata",
    "run_frame_routine",
]
