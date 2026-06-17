"""Celery tasks for running vision routines against uploaded videos."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger("apps.analytics")


def _run_routines_on_video(
    video_path: str,
    routine_names: List[str],
    params: Optional[Dict[str, dict]] = None,
    frame_step: int = 30,
    max_frames: int = 300,
) -> dict:
    """Execute the requested routines over a video and aggregate results.

    Returns a JSON-serialisable dict keyed by routine name. Frame-level routines
    are run on sampled frames; video-level routines (e.g. background
    subtraction) consume the frame stream directly.

    This helper is deliberately Django-free so it can be unit-tested directly.
    """

    # Imported lazily so importing this module never requires the CV stack.
    from .routines import get_routine, iter_frames, run_frame_routine

    params = params or {}
    frame_results: Dict[str, list] = {}
    video_results: Dict[str, dict] = {}

    # Split requested routines into frame-level and video-level.
    frame_routines, video_routines = [], []
    for name in routine_names:
        spec = get_routine(name)
        (video_routines if spec.level == "video" else frame_routines).append(name)

    # Frame-level routines: sample frames and run each routine per frame.
    if frame_routines:
        prev_frame = None
        for idx, frame in iter_frames(video_path, step=frame_step, max_frames=max_frames):
            for name in frame_routines:
                spec = get_routine(name)
                kwargs = dict(params.get(name, {}))
                if spec.level == "frame_pair":
                    if prev_frame is None:
                        continue
                    kwargs["prev_frame"] = prev_frame
                try:
                    out = run_frame_routine(name, frame, **kwargs)
                except Exception as exc:  # keep one bad frame from failing the run
                    logger.warning("Routine %s failed on frame %s: %s", name, idx, exc)
                    out = {"routine": name, "error": str(exc)}
                frame_results.setdefault(name, []).append({"frame": idx, "result": out})
            prev_frame = frame

    # Video-level routines: stream the frames through once each.
    for name in video_routines:
        spec = get_routine(name)
        kwargs = dict(params.get(name, {}))
        frames = (f for _, f in iter_frames(video_path, step=frame_step, max_frames=max_frames))
        video_results[name] = spec.func(frames, **kwargs)

    return {"frame_routines": frame_results, "video_routines": video_results}


def _summarise(results: dict) -> Dict[str, int]:
    """Derive the Analysis model counters from aggregated routine results."""

    objects_detected = 0
    for _name, frames in results.get("frame_routines", {}).items():
        for entry in frames:
            res = entry.get("result", {})
            objects_detected += len(res.get("detections", []))
    for _name, vres in results.get("video_routines", {}).items():
        for fr in vres.get("per_frame", []):
            objects_detected += len(fr.get("moving_objects", []))
    return {"objects_detected": objects_detected}


@shared_task(bind=True)
def run_video_analysis(
    self,
    analysis_id: int,
    routine_names: List[str],
    params: Optional[Dict[str, dict]] = None,
    frame_step: int = 30,
    max_frames: int = 300,
) -> dict:
    """Run the requested routines for an :class:`Analysis` and persist results."""

    from .models import Analysis

    analysis = Analysis.objects.select_related("video").get(pk=analysis_id)
    analysis.status = "processing"
    analysis.save(update_fields=["status", "updated_at"])

    try:
        video_path = analysis.video.file.path
        results = _run_routines_on_video(
            video_path,
            routine_names,
            params=params,
            frame_step=frame_step,
            max_frames=max_frames,
        )
        counters = _summarise(results)

        analysis.results = {"routines": routine_names, "data": results}
        analysis.objects_detected = counters["objects_detected"]
        analysis.status = "completed"
        analysis.completed_at = timezone.now()
        analysis.error_message = None
        analysis.save()

        # Parallel, orthogonal observability layer: turn the (unchanged) routine
        # output into wide commentary events. Guarded by a setting and wrapped so
        # commentary never affects the vision run's success.
        from django.conf import settings

        if getattr(settings, "COMMENTARY_ENABLED", False):
            try:
                from apps.observability.emit import emit_analysis_commentary
                from apps.observability.schema import new_trace_id

                # Resolve fps so commentary events carry real temporal segments
                # (segment_start/segment_end). Best-effort: a missing/zero fps
                # just leaves segments unset rather than failing the run.
                fps = None
                try:
                    from .routines import video_metadata

                    fps = video_metadata(video_path).get("fps") or None
                except Exception:  # noqa: BLE001 - fps is optional
                    logger.debug("Could not read fps for analysis %s", analysis_id)

                run_trace_id = new_trace_id()
                summary = emit_analysis_commentary(
                    results,
                    video_id=analysis.video_id,
                    analysis_id=analysis.id,
                    fps=fps,
                    frame_step=frame_step,
                    trace_id=run_trace_id,
                )
                # Persist the trace id so the run's commentary is correlatable
                # from the Analysis row (e.g. run_semantic_agent(trace_id=...)).
                analysis.trace_id = summary["trace_id"]
                analysis.save(update_fields=["trace_id", "updated_at"])
                logger.info(
                    "Commentary emitted for analysis %s: %s events (trace %s)",
                    analysis_id, summary["emitted"], summary["trace_id"],
                )
            except Exception:  # noqa: BLE001 - observability must not break analysis
                logger.exception("Commentary emission failed for analysis %s", analysis_id)
    except Exception as exc:  # noqa: BLE001 - record failure on the model
        logger.exception("Analysis %s failed", analysis_id)
        analysis.status = "failed"
        analysis.error_message = str(exc)
        analysis.save(update_fields=["status", "error_message", "updated_at"])
        raise

    return {"analysis_id": analysis_id, "status": analysis.status}
