"""Vehicle tracking for drone video: detect -> compensate camera motion -> track -> classify -> render.

Pure Python/NumPy/OpenCV (no Django imports) so it can run from the CLI (``python -m core.vehicles``)
and be unit-tested offline. The detector runtime (``ultralytics`` + ``torch``) is optional and imported
lazily by :mod:`core.vehicles.detector`; everything else only needs numpy, scipy and opencv.

Design notes and parameters: ``docs/vehicle_tracking_plan.md``.
"""

from .detector import VehiclesUnavailable  # noqa: F401
