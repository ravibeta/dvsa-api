"""The core fetch → extract → infer → store pipeline shared by every kit.

:func:`run_pipeline` turns a :class:`RunInput` into a schema-valid
:class:`RunOutput`, wiring together whichever adapters the caller supplies (or
sensible offline defaults). Local, cloud, and child-worker kits all call this so a
run behaves identically regardless of how it was triggered.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .adapters import (
    FrameExtractor,
    InferenceAdapter,
    LocalFileStorageAdapter,
    LocalVideoFetcher,
    MockInferenceAdapter,
    StorageAdapter,
    SyntheticFrameExtractor,
    VideoFetcher,
)
from .logging import RunLogger
from .schemas import Provenance, RunInput, RunOutput
from .utils import deterministic_run_id, stable_sort_detections, summarize_detections


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_pipeline(
    run_input: RunInput,
    *,
    fetcher: Optional[VideoFetcher] = None,
    extractor: Optional[FrameExtractor] = None,
    inference: Optional[InferenceAdapter] = None,
    storage: Optional[StorageAdapter] = None,
    store_dest: Optional[str] = None,
    logger: Optional[RunLogger] = None,
) -> RunOutput:
    """Execute the full pipeline for ``run_input`` and return a :class:`RunOutput`.

    Adapters default to the offline reference implementations. When ``store_dest``
    is provided the output is also persisted via ``storage``.
    """
    fetcher = fetcher or LocalVideoFetcher()
    extractor = extractor or SyntheticFrameExtractor()
    inference = inference or MockInferenceAdapter()
    storage = storage or LocalFileStorageAdapter()

    run_id = run_input.run_id or deterministic_run_id(
        run_input.video_uri, run_input.sensor_id, run_input.processing_flags)
    log = logger or RunLogger(run_id=run_id, agent_id=run_input.agent_id,
                              model_version=inference.model_version)
    trace_id = log.context["trace_id"]

    start_time = _now_iso()
    log.info("pipeline_start", video_uri=run_input.video_uri)

    fps = float(run_input.processing_flags.get("fps", 1.0))
    local_path = fetcher.fetch_video(run_input.video_uri)
    frames = extractor.extract_frames(local_path, fps=fps,
                                      time_window=run_input.time_window)
    log.info("frames_extracted", count=len(frames))

    detections = stable_sort_detections(inference.run_inference_on_frames(frames))
    for det in detections:
        if det.provenance is None:
            det.provenance = Provenance(source="run_pipeline", trace_id=trace_id)
        else:
            det.provenance.trace_id = trace_id
    log.info("inference_complete", detections=len(detections))

    output = RunOutput(
        run_id=run_id,
        agent_id=run_input.agent_id,
        start_time=start_time,
        end_time=_now_iso(),
        model_version=inference.model_version,
        detections=detections,
        summary={**summarize_detections(detections), "frames_processed": len(frames)},
        provenance=Provenance(source="run_pipeline", method="fetch-extract-infer",
                              trace_id=trace_id,
                              extra={"video_uri": run_input.video_uri}),
    )

    if store_dest:
        uri = storage.store_output(output, store_dest)
        output.summary["stored_uri"] = uri
        log.info("output_stored", uri=uri)

    log.info("pipeline_done", detections=len(detections))
    return output


__all__ = ["run_pipeline"]
