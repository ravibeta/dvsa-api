"""Model-backed commentator (Phase 4).

:class:`AzureVLMCommentator` plugs into the exact same :class:`Commentator`
interface as the deterministic :class:`~apps.observability.commentator.TemplateCommentator`,
so it can be selected via ``COMMENTARY_COMMENTATOR=vlm`` with no other change.

It turns a structured routine result into a richer natural-language description
by prompting an :class:`~apps.observability.llm.LLMClient`. The model client is
injectable (tests pass a fake/echo client) and **any** model error falls back to
the template commentary — model commentary must never break a vision run.

Note on "VLM": this commentator reasons over the *structured* detection output
(labels, counts, metrics) the pipeline produces per frame; an optional raw frame
can be threaded through ``metadata`` later without changing this interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .commentator import CommentaryContext, Commentator, TemplateCommentator
from .llm import LLMClient, get_llm_client
from .schema import CommentaryEvent, derive_metrics, new_span_id

DEFAULT_SYSTEM_PROMPT = (
    "You are a drone video analytics commentator. Given structured detections "
    "from a single frame, write one concise, factual sentence describing the "
    "scene. Do not invent objects that are not listed."
)


def _build_prompt(routine: str, result: Dict[str, Any], ctx: CommentaryContext) -> str:
    """Compose a compact, deterministic prompt from a routine result."""

    dets = result.get("detections", []) or []
    labels: Dict[str, int] = {}
    for d in dets:
        lbl = str(d.get("label", "object"))
        labels[lbl] = labels.get(lbl, 0) + 1
    label_line = ", ".join("%d %s" % (n, l) for l, n in sorted(labels.items())) or "none"
    summary = result.get("summary", {})
    lines = [
        "Routine: %s" % routine,
        "Frame: %s" % (ctx.frame_index if ctx.frame_index is not None else "n/a"),
        "Detections by label: %s" % label_line,
        "Total detections: %d" % len(dets),
    ]
    if summary:
        lines.append("Summary: %s" % summary)
    return "\n".join(lines)


class AzureVLMCommentator(Commentator):
    """Generates commentary with an LLM, falling back to templates on any error."""

    name = "vlm"

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        *,
        fallback: Optional[Commentator] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.fallback = fallback or TemplateCommentator()
        self.system_prompt = system_prompt

    def comment_on_result(
        self, routine: str, result: Dict[str, Any], ctx: CommentaryContext
    ) -> List[CommentaryEvent]:
        if "error" in result:
            return self.fallback.comment_on_result(routine, result, ctx)

        prompt = _build_prompt(routine, result, ctx)
        try:
            text = self.llm.complete(prompt, system=self.system_prompt)
        except Exception:  # noqa: BLE001 - model failure must not break the run
            return self.fallback.comment_on_result(routine, result, ctx)

        metrics = derive_metrics(result.get("detections", []))
        event = CommentaryEvent(
            commentary=text,
            attributes={
                "generated_by": self.llm.name,
                "labels": sorted({str(d.get("label", "object"))
                                  for d in result.get("detections", [])}),
            },
            metrics=metrics,
            metadata={"routine": routine, "commentator": "vlm"},
            trace_id=ctx.trace_id,
            span_id=new_span_id(),
            parent_span_id=ctx.parent_span_id,
            correlation_key=ctx.correlation_key(),
            source="agent:vlm",
            video_id=ctx.video_id,
            analysis_id=ctx.analysis_id,
            frame_index=ctx.frame_index,
            segment_start=ctx.segment_seconds(),
        )
        return [event]
