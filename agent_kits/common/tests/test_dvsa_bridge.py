"""Tests for the DVSA bridge adapters (VideoUploadAPIView / ChatAPIView).

The adapters invoke the real DRF views in-process, but here we inject a fake
``view_caller`` and a sentinel ``view`` so the tests exercise argument construction,
result mapping, and the optional ``run_pipeline`` hooks **without** importing Django,
DRF or Azure — keeping the suite offline.
"""

import os

import pytest

from agent_kits.common import (
    DvsaChatAnalyzer,
    DvsaVideoUploadAdapter,
    RunInput,
    run_pipeline,
)
from agent_kits.test_assets.generate_synthetic_video import ensure_manifest

_ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "test_assets", "sample_short.json")


@pytest.fixture(scope="module")
def video() -> str:
    return ensure_manifest(_ASSET)


class _FakeCaller:
    """Records the last invocation and returns a canned ``(status, data)``."""

    def __init__(self, status=200, data=None):
        self.status = status
        self.data = data if data is not None else {}
        self.calls = []

    def __call__(self, view_cls, method, data, *, user=None, path="/"):
        # Snapshot data (the file handle would be closed after the call returns).
        snapshot = {k: ("<file>" if hasattr(v, "read") else v) for k, v in data.items()}
        self.calls.append({"view": view_cls, "method": method, "data": snapshot,
                           "user": user, "path": path})
        return self.status, self.data


# --------------------------------------------------------------------------- #
# DvsaVideoUploadAdapter
# --------------------------------------------------------------------------- #
def test_upload_adapter_ingests_via_view(video):
    caller = _FakeCaller(status=201, data={"id": 7, "status": "INITIALIZED"})
    adapter = DvsaVideoUploadAdapter(account_id="acct-1", user="u1",
                                     view=object(), view_caller=caller)
    result = adapter.ingest_video(video)
    assert result["status_code"] == 201
    assert result["video_entity"]["id"] == 7
    call = caller.calls[-1]
    assert call["method"] == "post"
    assert call["data"]["account_id"] == "acct-1"
    assert call["data"]["file"] == "<file>"   # a file handle was passed
    assert call["user"] == "u1"


def test_upload_adapter_requires_account_id(video, monkeypatch):
    monkeypatch.delenv("DVSA_ACCOUNT_ID", raising=False)
    adapter = DvsaVideoUploadAdapter(view=object(), view_caller=_FakeCaller())
    with pytest.raises(ValueError):
        adapter.ingest_video(video)


def test_upload_adapter_reads_account_from_env(video, monkeypatch):
    monkeypatch.setenv("DVSA_ACCOUNT_ID", "env-acct")
    caller = _FakeCaller(status=201)
    adapter = DvsaVideoUploadAdapter(view=object(), view_caller=caller)
    adapter.ingest_video(video)
    assert caller.calls[-1]["data"]["account_id"] == "env-acct"


def test_upload_adapter_raises_on_view_error(video):
    caller = _FakeCaller(status=500, data={"error": "azure down"})
    adapter = DvsaVideoUploadAdapter(account_id="a", view=object(), view_caller=caller)
    with pytest.raises(RuntimeError, match="VideoUploadAPIView failed"):
        adapter.ingest_video(video)


def test_upload_adapter_is_a_video_fetcher(video):
    caller = _FakeCaller(status=201)
    adapter = DvsaVideoUploadAdapter(account_id="a", view=object(),
                                     view_caller=caller, ingest_on_fetch=True)
    path = adapter.fetch_video(video)
    assert path.endswith("sample_short.json")
    assert caller.calls, "ingest_on_fetch should have invoked the upload view"


# --------------------------------------------------------------------------- #
# DvsaChatAnalyzer
# --------------------------------------------------------------------------- #
def test_chat_analyzer_asks_via_view():
    caller = _FakeCaller(status=200, data={"text": "Two vehicles collided at 0:02."})
    analyzer = DvsaChatAnalyzer(account_id="acct-1", user="u1",
                                view=object(), view_caller=caller)
    result = analyzer.ask("what happened?")
    assert result["answer"] == "Two vehicles collided at 0:02."
    call = caller.calls[-1]
    assert call["method"] == "put"
    assert call["data"] == {"account_id": "acct-1", "query": "what happened?"}
    assert call["user"] == "u1"


def test_chat_analyzer_requires_query():
    analyzer = DvsaChatAnalyzer(account_id="a", view=object(), view_caller=_FakeCaller())
    with pytest.raises(ValueError):
        analyzer.ask("")


def test_chat_analyzer_raises_on_view_error():
    caller = _FakeCaller(status=500, data={"error": "chat failed"})
    analyzer = DvsaChatAnalyzer(account_id="a", view=object(), view_caller=caller)
    with pytest.raises(RuntimeError, match="ChatAPIView failed"):
        analyzer.ask("q")


# --------------------------------------------------------------------------- #
# run_pipeline optional hooks
# --------------------------------------------------------------------------- #
def test_pipeline_ingestor_and_analyzer_hooks(video):
    upload_caller = _FakeCaller(status=201, data={"id": 3})
    chat_caller = _FakeCaller(status=200, data={"text": "accident confirmed"})
    ingestor = DvsaVideoUploadAdapter(account_id="a", view=object(),
                                      view_caller=upload_caller)
    analyzer = DvsaChatAnalyzer(account_id="a", view=object(), view_caller=chat_caller)

    out = run_pipeline(
        RunInput(video_uri=video, processing_flags={"fps": 5}),
        ingestor=ingestor, analyzer=analyzer, query="what happened?")

    # Offline detection pipeline still ran ...
    assert out.detections
    # ... and both existing-view hooks were leveraged and recorded.
    assert out.summary["ingestion"]["video_entity"]["id"] == 3
    assert out.summary["agentic_answer"] == "accident confirmed"
    assert out.summary["agentic_query"] == "what happened?"
    assert out.provenance.extra["analyzer"] == "ChatAPIView"
    assert upload_caller.calls and chat_caller.calls


def test_pipeline_without_hooks_is_unchanged(video):
    out = run_pipeline(RunInput(video_uri=video, processing_flags={"fps": 5}))
    assert "ingestion" not in out.summary
    assert "agentic_answer" not in out.summary
