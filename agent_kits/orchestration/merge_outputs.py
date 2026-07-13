"""Deterministic merge of child :class:`RunOutput`s into one canonical result.

Partitioning with overlap means the same object can be detected by more than one
child. :func:`merge_run_outputs` reconciles those into a single, reproducible
timeline:

* **overlap removal** via IoU (boxes overlapping ≥ threshold at the same timestamp
  are duplicates),
* **conflict resolution** by confidence (the highest-confidence detection wins),
* **canonical timeline** via a total ordering on (timestamp, class, confidence, box).

Given the same set of child outputs the merged *detection data* is byte-for-byte
identical — see :func:`canonical_timeline` for the reproducible, provenance-free
view (per-run ``trace_id`` lineage on individual detections is intentionally
preserved and is the only thing that varies between runs).
"""

from __future__ import annotations

from typing import List, Sequence

from ..common import (
    Provenance,
    RunOutput,
    deterministic_run_id,
    merge_detection_sets,
    summarize_detections,
)


def merge_run_outputs(
    outputs: Sequence[RunOutput],
    *,
    iou_threshold: float = 0.5,
    run_id: str = "",
) -> RunOutput:
    """Merge child ``outputs`` into one canonical :class:`RunOutput`."""
    if not outputs:
        raise ValueError("merge_run_outputs requires at least one RunOutput")

    child_ids = [o.run_id for o in outputs]
    merged_detections = merge_detection_sets(
        [o.detections for o in outputs], iou_threshold=iou_threshold)

    frames_processed = sum(
        int(o.summary.get("frames_processed", 0)) for o in outputs)
    starts = [o.start_time for o in outputs if o.start_time]
    ends = [o.end_time for o in outputs if o.end_time]

    model_versions = sorted({o.model_version for o in outputs})
    merged_id = run_id or deterministic_run_id(*sorted(child_ids), prefix="merge")

    summary = summarize_detections(merged_detections)
    summary.update({
        "frames_processed": frames_processed,
        "partitions_merged": len(outputs),
        "child_run_ids": sorted(child_ids),
        "pre_merge_detections": sum(len(o.detections) for o in outputs),
    })

    return RunOutput(
        run_id=merged_id,
        agent_id="orchestrator",
        start_time=min(starts) if starts else "",
        end_time=max(ends) if ends else "",
        model_version=model_versions[0] if len(model_versions) == 1
        else ",".join(model_versions),
        detections=merged_detections,
        summary=summary,
        provenance=Provenance(
            source="merge_outputs", method="iou-dedupe+confidence",
            extra={"iou_threshold": iou_threshold,
                   "child_run_ids": sorted(child_ids)}),
    )


def canonical_timeline(output: RunOutput) -> List[dict]:
    """Return a compact, ordered timeline view of a merged output's detections."""
    return [
        {"timestamp": d.timestamp, "class_name": d.class_name,
         "confidence": d.confidence, "bbox": list(d.bbox)}
        for d in output.detections
    ]


__all__ = ["merge_run_outputs", "canonical_timeline"]
