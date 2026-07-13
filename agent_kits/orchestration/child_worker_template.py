"""Child worker template — process one partition of a drone video.

A child receives a :class:`RunInput` describing a *partition* (a time window and/or
a spatial tile carried in ``processing_flags['tile']``) and returns a
:class:`RunOutput` for just that partition. Parent orchestrators fan these out to a
cloud runner or a local queue; this template is the reference implementation used by
the in-process executor and by tests.

Determinism: identical partitions always yield identical outputs, which is what lets
the parent merge overlapping partitions into a single canonical result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..common import (
    Detection,
    MockInferenceAdapter,
    Provenance,
    RunInput,
    RunOutput,
)
from ..common.pipeline import run_pipeline


def _center_in_tile(bbox: List[float], tile: Dict[str, Any]) -> bool:
    """True when the box centre falls inside ``tile`` = {x, y, w, h}."""
    cx = bbox[0] + bbox[2] / 2.0
    cy = bbox[1] + bbox[3] / 2.0
    return (tile["x"] <= cx <= tile["x"] + tile["w"]
            and tile["y"] <= cy <= tile["y"] + tile["h"])


def process_partition(
    run_input: RunInput,
    *,
    inference: Optional[MockInferenceAdapter] = None,
) -> RunOutput:
    """Run the pipeline for a single partition, applying any spatial-tile filter."""
    inference = inference or MockInferenceAdapter()
    output = run_pipeline(run_input, inference=inference)

    tile = run_input.processing_flags.get("tile")
    if tile:
        kept: List[Detection] = [
            d for d in output.detections if _center_in_tile(d.bbox, tile)]
        output.detections = kept
        output.summary["total_detections"] = len(kept)

    # Stamp partition provenance so the merge step is auditable.
    partition_id = run_input.processing_flags.get("partition_id", run_input.run_id)
    output.provenance = Provenance(
        source="child_worker",
        method="process_partition",
        trace_id=output.provenance.trace_id if output.provenance else None,
        extra={
            "partition_id": partition_id,
            "time_window": run_input.time_window.model_dump()
            if run_input.time_window else None,
            "tile": tile,
        },
    )
    output.summary["partition_id"] = partition_id
    return output


__all__ = ["process_partition"]
