"""Optional critic/review layer for orchestrated runs.

Two independent, opt-in review strategies over a merged :class:`RunOutput`:

* **Human review sampling** — deterministically select a fraction of detections for
  a human to spot-check (seeded, reproducible sampling).
* **Secondary-model validation** — re-run a *different* inference adapter over the
  same source and report agreement/disagreement with the primary detections.

The layer is entirely optional: orchestration works without it, and it never mutates
the run output — it only produces a review report.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ..common import (
    Detection,
    InferenceAdapter,
    MockInferenceAdapter,
    RunInput,
    RunOutput,
    SyntheticFrameExtractor,
    iou,
)
from ..common import LocalVideoFetcher


def sample_for_human_review(
    output: RunOutput, *, fraction: float = 0.1, seed: int = 0,
) -> List[Detection]:
    """Deterministically sample a fraction of detections for human review."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    detections = list(output.detections)
    if not detections or fraction == 0.0:
        return []
    k = max(1, round(len(detections) * fraction))
    rng = random.Random(seed)
    # Sort first for determinism, then sample stable indices.
    indexed = sorted(range(len(detections)),
                     key=lambda i: (detections[i].timestamp,
                                    detections[i].class_name,
                                    -detections[i].confidence))
    chosen = sorted(rng.sample(indexed, k))
    return [detections[i] for i in chosen]


def secondary_model_validation(
    run_input: RunInput,
    primary: RunOutput,
    *,
    secondary: Optional[InferenceAdapter] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Validate primary detections against a secondary model's detections.

    Returns a report with agreement counts and the specific detections only one
    model found (potential false positives / missed objects).
    """
    secondary = secondary or MockInferenceAdapter(model_version="secondary-1.0.0")
    path = LocalVideoFetcher().fetch_video(run_input.video_uri)
    frames = SyntheticFrameExtractor().extract_frames(
        path, fps=float(run_input.processing_flags.get("fps", 1.0)),
        time_window=run_input.time_window)
    secondary_dets = secondary.run_inference_on_frames(frames)

    agreed = 0
    primary_only: List[Detection] = []
    matched_secondary = set()
    for pdet in primary.detections:
        match = _find_match(pdet, secondary_dets, iou_threshold, matched_secondary)
        if match is None:
            primary_only.append(pdet)
        else:
            agreed += 1
            matched_secondary.add(match)
    secondary_only = [d for i, d in enumerate(secondary_dets)
                      if i not in matched_secondary]

    total = len(primary.detections)
    return {
        "primary_model": primary.model_version,
        "secondary_model": secondary.model_version,
        "primary_count": total,
        "secondary_count": len(secondary_dets),
        "agreed": agreed,
        "agreement_ratio": round(agreed / total, 4) if total else 0.0,
        "primary_only": [d.model_dump() for d in primary_only],
        "secondary_only": [d.model_dump() for d in secondary_only],
    }


def _find_match(
    det: Detection, candidates: List[Detection], threshold: float, used: set,
) -> Optional[int]:
    for i, cand in enumerate(candidates):
        if i in used:
            continue
        if cand.class_name != det.class_name:
            continue
        if abs(cand.timestamp - det.timestamp) > 1e-9:
            continue
        if iou(cand.bbox, det.bbox) >= threshold:
            return i
    return None


def review(
    run_input: RunInput,
    output: RunOutput,
    *,
    human_sample_fraction: float = 0.0,
    run_secondary: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the enabled review strategies and return a combined report (optional)."""
    report: Dict[str, Any] = {"run_id": output.run_id, "reviewed_detections":
                              len(output.detections)}
    if human_sample_fraction > 0.0:
        sampled = sample_for_human_review(
            output, fraction=human_sample_fraction, seed=seed)
        report["human_review_sample"] = [d.model_dump() for d in sampled]
    if run_secondary:
        report["secondary_validation"] = secondary_model_validation(run_input, output)
    return report


__all__ = ["sample_for_human_review", "secondary_model_validation", "review"]
