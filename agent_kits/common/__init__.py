"""Shared foundation for the DVSA agent kits — schemas, adapters, logging, utils.

Import from here rather than the submodules directly, e.g.::

    from agent_kits.common import RunInput, run_pipeline, RunLogger
"""

from __future__ import annotations

from .adapters import (
    Frame,
    FrameExtractor,
    InferenceAdapter,
    LocalFileStorageAdapter,
    LocalVideoFetcher,
    MockInferenceAdapter,
    StorageAdapter,
    SyntheticFrameExtractor,
    VideoFetcher,
)
from .logging import RunLogger, configure_logging, new_trace_id
from .pipeline import run_pipeline
from .schemas import (
    Detection,
    Provenance,
    RunInput,
    RunOutput,
    TimeWindow,
    export_schemas,
)
from .utils import (
    dedupe_detections,
    deterministic_run_id,
    iou,
    merge_detection_sets,
    stable_sort_detections,
    summarize_detections,
)

__all__ = [
    # schemas
    "RunInput",
    "Detection",
    "RunOutput",
    "TimeWindow",
    "Provenance",
    "export_schemas",
    # adapters
    "Frame",
    "VideoFetcher",
    "FrameExtractor",
    "InferenceAdapter",
    "StorageAdapter",
    "LocalVideoFetcher",
    "SyntheticFrameExtractor",
    "MockInferenceAdapter",
    "LocalFileStorageAdapter",
    # pipeline
    "run_pipeline",
    # logging
    "RunLogger",
    "configure_logging",
    "new_trace_id",
    # utils
    "deterministic_run_id",
    "stable_sort_detections",
    "iou",
    "dedupe_detections",
    "merge_detection_sets",
    "summarize_detections",
]
