"""Unit tests for the model-backed commentator and semantic agent (Phase 4).

All model interaction goes through an injected client (echo or a fake), so these
run offline with no network or credentials — same pure-Python style as the other
observability tests.
"""

from apps.observability.schema import CommentaryEvent, new_trace_id
from apps.observability.commentator import CommentaryContext, get_commentator
from apps.observability.llm import (
    AzureOpenAIChatClient,
    EchoLLMClient,
    LLMClient,
    get_llm_client,
)
from apps.observability.vlm import AzureVLMCommentator
from apps.observability.agents import SemanticAggregatorAgent, digest_events


def _color_result(n=2):
    dets = [
        {"label": "car", "area": 100.0 * (i + 1), "score": 0.9, "bbox": [0, 0, 1, 1], "centroid": [0, 0]}
        for i in range(n)
    ]
    return {"routine": "color_detection", "summary": {"count": n}, "detections": dets}


class _FakeLLM(LLMClient):
    name = "fake"

    def __init__(self, reply="a concise scene description", raise_exc=False):
        self.reply = reply
        self.raise_exc = raise_exc
        self.calls = []

    def complete(self, prompt, *, system=None):
        self.calls.append((prompt, system))
        if self.raise_exc:
            raise RuntimeError("model down")
        return self.reply


# --------------------------------------------------------------------------- #
# LLM clients
# --------------------------------------------------------------------------- #

def test_echo_client_is_deterministic():
    c = EchoLLMClient()
    out = c.complete("first line\nsecond line")
    assert out == c.complete("first line\nsecond line")
    assert out.startswith("[echo]")


def test_get_llm_client_default_is_echo():
    assert isinstance(get_llm_client("echo"), EchoLLMClient)
    try:
        get_llm_client("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_azure_client_requires_credentials():
    try:
        AzureOpenAIChatClient(None, None, None)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_azure_client_build_request_is_pure_and_correct():
    c = AzureOpenAIChatClient("https://x.openai.azure.com/", "KEY", "gpt4o", api_version="2024-06-01")
    url, headers, body = c.build_request("hello", system="be brief")
    assert url == ("https://x.openai.azure.com/openai/deployments/gpt4o/"
                   "chat/completions?api-version=2024-06-01")
    assert headers["api-key"] == "KEY"
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}


# --------------------------------------------------------------------------- #
# VLM commentator
# --------------------------------------------------------------------------- #

def test_vlm_commentator_emits_model_event():
    fake = _FakeLLM(reply="Two cars are visible on the road.")
    ctx = CommentaryContext(new_trace_id(), video_id=7, frame_index=30, fps=30.0)
    events = AzureVLMCommentator(fake).comment_on_result("color_detection", _color_result(2), ctx)
    assert len(events) == 1
    ev = events[0]
    assert ev.commentary == "Two cars are visible on the road."
    assert ev.source == "agent:vlm"
    assert ev.attributes["generated_by"] == "fake"
    assert ev.metrics["count"] == 2.0
    assert ev.correlation_key == "video:7|frame:30"
    # The prompt actually carried the structured detections.
    assert "color_detection" in fake.calls[0][0]


def test_vlm_commentator_falls_back_on_model_error():
    fake = _FakeLLM(raise_exc=True)
    ctx = CommentaryContext(new_trace_id(), frame_index=0)
    events = AzureVLMCommentator(fake).comment_on_result("color_detection", _color_result(1), ctx)
    assert len(events) == 1
    # Fell back to the deterministic template (different source, real text).
    assert events[0].source == "routine:color_detection"
    assert "car" in events[0].commentary


def test_vlm_commentator_passes_through_routine_errors():
    fake = _FakeLLM()
    ctx = CommentaryContext(new_trace_id(), frame_index=0)
    events = AzureVLMCommentator(fake).comment_on_result(
        "color_detection", {"routine": "color_detection", "error": "boom"}, ctx
    )
    assert "failed" in events[0].commentary
    assert fake.calls == []  # never called the model on an upstream error


def test_get_commentator_vlm():
    assert isinstance(get_commentator("vlm"), AzureVLMCommentator)


# --------------------------------------------------------------------------- #
# Semantic aggregation agent
# --------------------------------------------------------------------------- #

def _low_level_events(trace):
    return [
        CommentaryEvent(
            commentary="1 car", metrics={"count": 1.0}, source="routine:color_detection",
            trace_id=trace, frame_index=0, video_id=7,
        ),
        CommentaryEvent(
            commentary="2 car", metrics={"count": 2.0}, source="routine:color_detection",
            trace_id=trace, frame_index=30, video_id=7,
        ),
        CommentaryEvent(
            commentary="motion", metrics={"count": 3.0}, source="routine:background_subtraction",
            trace_id=trace, frame_index=30, video_id=7,
        ),
    ]


def test_digest_events_aggregates_sources_and_metrics():
    d = digest_events(_low_level_events(new_trace_id()))
    assert d["event_count"] == 3
    assert d["sources"]["routine:color_detection"] == 2
    assert d["frames_covered"] == 2  # frames {0, 30}
    assert d["metric_totals"]["count"] == 6.0


def test_semantic_agent_emits_correlated_higher_level_event():
    trace = new_trace_id()
    events = _low_level_events(trace)
    fake = _FakeLLM(reply="Traffic builds across the segment.")
    out = SemanticAggregatorAgent(fake).summarize(events, video_id=7, scope="video")
    assert len(out) == 1
    summary = out[0]
    assert summary.source == "agent:semantic"
    assert summary.commentary == "Traffic builds across the segment."
    assert summary.trace_id == trace                       # shares the trace
    assert summary.correlation_key == "video:7|scope:video"
    assert summary.attributes["derived_from"] == 3
    assert len(summary.attributes["derived_from_spans"]) == 3
    assert summary.metrics["events_summarized"] == 3.0
    assert summary.metrics["total_count"] == 6.0


def test_semantic_agent_falls_back_on_model_error():
    trace = new_trace_id()
    out = SemanticAggregatorAgent(_FakeLLM(raise_exc=True)).summarize(
        _low_level_events(trace), video_id=7
    )
    assert out[0].commentary.startswith("Scene summary —")


def test_semantic_agent_empty_input():
    assert SemanticAggregatorAgent(EchoLLMClient()).summarize([]) == []
