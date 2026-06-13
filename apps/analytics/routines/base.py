"""Shared primitives for DVSA vision routines.

This module defines the lightweight data structures and helpers shared by every
routine, plus a small registry that makes routines *pluggable* (see the
"modular, extensible architecture" recommendation in ``plan-of-action.md`` §21).

Design notes
------------
* Routines operate on plain ``numpy`` arrays (BGR frames, as returned by
  OpenCV) and return JSON-serialisable ``dict`` payloads. This keeps them
  decoupled from Django so they can be unit-tested in isolation and stored
  directly in ``Analysis.results`` (a ``JSONField``).
* Heavy / optional imports (cv2, sklearn, shapely) are performed lazily inside
  the routine modules so that importing the registry never fails in an
  environment that only needs a subset of the stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """A single detected object.

    Coordinates are in pixels relative to the top-left of the source frame.
    ``bbox`` is ``(x, y, w, h)`` and ``centroid`` is ``(cx, cy)``.
    """

    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: float
    label: str = "object"
    score: float = 1.0
    track_id: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure plain Python types for JSON serialisation.
        d["bbox"] = [int(v) for v in self.bbox]
        d["centroid"] = [float(v) for v in self.centroid]
        d["area"] = float(self.area)
        d["score"] = float(self.score)
        return d


@dataclass
class RoutineResult:
    """Standard envelope returned by every routine."""

    routine: str
    summary: Dict = field(default_factory=dict)
    detections: List[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "routine": self.routine,
            "summary": self.summary,
            "detections": [d.to_dict() for d in self.detections],
        }


# ---------------------------------------------------------------------------
# Registry — makes routines discoverable and pluggable
# ---------------------------------------------------------------------------


@dataclass
class RoutineSpec:
    name: str
    func: Callable
    description: str
    level: str  # "frame" or "video"


_REGISTRY: Dict[str, RoutineSpec] = {}


def register(name: str, description: str = "", level: str = "frame") -> Callable:
    """Decorator registering ``func`` under ``name`` in the global registry."""

    def _wrap(func: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"Routine '{name}' is already registered")
        _REGISTRY[name] = RoutineSpec(
            name=name,
            func=func,
            description=description or (func.__doc__ or "").strip().split("\n")[0],
            level=level,
        )
        return func

    return _wrap


def get_routine(name: str) -> RoutineSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - trivial
        raise KeyError(
            f"Unknown routine '{name}'. Available: {sorted(_REGISTRY)}"
        ) from exc


def available_routines() -> List[dict]:
    """Return JSON-serialisable metadata for every registered routine."""

    return [
        {"name": s.name, "description": s.description, "level": s.level}
        for s in sorted(_REGISTRY.values(), key=lambda s: s.name)
    ]


# ---------------------------------------------------------------------------
# Frame / video helpers
# ---------------------------------------------------------------------------


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR (or already-gray) frame to single-channel grayscale."""

    import cv2

    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def iter_frames(path: str, step: int = 1, max_frames: Optional[int] = None):
    """Yield ``(index, frame)`` pairs from a video file using ``cv2.VideoCapture``.

    Parameters
    ----------
    path:
        Path to the video file.
    step:
        Sample every ``step`` frames (``1`` = every frame).
    max_frames:
        Optional cap on the number of *yielded* frames.
    """

    import cv2

    if step < 1:
        raise ValueError("step must be >= 1")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    yielded = 0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def video_metadata(path: str) -> dict:
    """Return basic metadata (fps, frame count, resolution) for a video."""

    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()
