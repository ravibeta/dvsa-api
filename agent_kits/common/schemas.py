"""Shared data schemas for all agent kits.

Defines the three contracts that flow through every kit — :class:`RunInput`,
:class:`Detection`, :class:`RunOutput` — as Pydantic models with programmatic
JSON-schema export via :func:`export_schemas`. Models are deterministic and
serialise stably so runs are reproducible and auditable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._compat import BaseModel, Field

MODEL_VERSION_UNKNOWN = "unknown"


class TimeWindow(BaseModel):
    """Optional processing window as ISO-8601 timestamps."""

    start: Optional[str] = Field(default=None, description="ISO-8601 start timestamp")
    end: Optional[str] = Field(default=None, description="ISO-8601 end timestamp")


class Provenance(BaseModel):
    """Where a value came from — for auditability across kits."""

    source: str = Field(description="Component that produced the value")
    method: Optional[str] = Field(default=None, description="Algorithm / adapter used")
    trace_id: Optional[str] = Field(default=None, description="Correlates a single run")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Free-form context")


class RunInput(BaseModel):
    """Request to process a drone video (or a partition of one)."""

    video_uri: str = Field(description="URI of the source video (file://, s3://, ...)")
    time_window: Optional[TimeWindow] = Field(
        default=None, description="Optional start/end ISO timestamps to process")
    sensor_id: Optional[str] = Field(default=None, description="Originating sensor id")
    processing_flags: Dict[str, Any] = Field(
        default_factory=dict, description="Tunable knobs (fps, model, tiles, ...)")
    agent_id: Optional[str] = Field(default=None, description="Agent executing the run")
    run_id: Optional[str] = Field(default=None, description="Stable id for this run")


class Detection(BaseModel):
    """A single object detection within a frame."""

    timestamp: float = Field(description="Seconds from video start")
    bbox: List[float] = Field(description="Bounding box [x, y, w, h]")
    class_name: str = Field(description="Detected class label")
    confidence: float = Field(description="Detector confidence in [0, 1]")
    model_version: str = Field(
        default=MODEL_VERSION_UNKNOWN, description="Model that produced this detection")
    provenance: Optional[Provenance] = Field(
        default=None, description="How/where this detection was produced")


class RunOutput(BaseModel):
    """Result of processing a :class:`RunInput`."""

    run_id: str = Field(description="Stable id for this run")
    agent_id: Optional[str] = Field(default=None, description="Agent that produced it")
    start_time: str = Field(description="ISO-8601 run start")
    end_time: str = Field(description="ISO-8601 run end")
    model_version: str = Field(
        default=MODEL_VERSION_UNKNOWN, description="Primary model version used")
    detections: List[Detection] = Field(
        default_factory=list, description="All detections, stably ordered")
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Aggregate counts / timeline summary")
    provenance: Optional[Provenance] = Field(
        default=None, description="Run-level provenance")


def export_schemas() -> Dict[str, Dict[str, Any]]:
    """Return JSON schemas for the public contracts, keyed by model name."""
    return {
        "RunInput": RunInput.model_json_schema(),
        "Detection": Detection.model_json_schema(),
        "RunOutput": RunOutput.model_json_schema(),
    }


__all__ = [
    "TimeWindow",
    "Provenance",
    "RunInput",
    "Detection",
    "RunOutput",
    "export_schemas",
    "MODEL_VERSION_UNKNOWN",
]
