"""Tests for the Qwen VLM function-tool (``core.azure.analyzer.ask_qwen_vlm``).

Fully offline: the network is mocked (``requests_mock``) and the global on/off is
driven by the ``DVSA_QWEN_ENABLED`` setting, so no Azure credentials are needed.
The suite covers the request/response contract, graceful fallbacks, and — most
importantly — that the global flag toggles whether Qwen joins the agent tool set
(the pre-Qwen behaviour is preserved byte-for-byte when disabled).
"""

from __future__ import annotations

from django.test import override_settings

from core.azure import analyzer

QWEN_URL = "https://found-vision-1.services.ai.azure.com/openai/v1/chat/completions"


# ----- offline / unconfigured fallbacks ------------------------------------
@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_API_KEY=None)
def test_returns_no_comment_when_unconfigured(requests_mock):
    # No key -> benign fallback, and no HTTP call is attempted.
    assert analyzer.ask_qwen_vlm("what is in the scene?") == "No comment."
    assert requests_mock.call_count == 0


@override_settings(DVSA_QWEN_ENABLED=False, DVSA_QWEN_API_KEY="secret-key")
def test_returns_no_comment_when_globally_disabled(requests_mock):
    # Global kill-switch: even with a key, Qwen stays silent and makes no call.
    assert analyzer.ask_qwen_vlm("what is in the scene?") == "No comment."
    assert requests_mock.call_count == 0


# ----- request/response contract -------------------------------------------
@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_API_KEY="secret-key",
                   DVSA_QWEN_ENDPOINT=QWEN_URL, DVSA_QWEN_MODEL="qwen--qwen3.5-0.8b")
def test_calls_endpoint_with_openai_chat_shape(requests_mock):
    requests_mock.post(QWEN_URL, json={
        "choices": [{"message": {"role": "assistant",
                                 "content": "two cars and a truck"}}]
    })

    out = analyzer.ask_qwen_vlm("count vehicles")
    assert out == "two cars and a truck"

    req = requests_mock.request_history[0]
    assert req.url == QWEN_URL
    assert req.headers["Authorization"] == "Bearer secret-key"
    body = req.json()
    assert body["model"] == "qwen--qwen3.5-0.8b"
    assert body["temperature"] == 0.6
    assert body["top_p"] == 0.95
    assert body["messages"][-1] == {"role": "user", "content": "count vehicles"}


@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_API_KEY="secret-key",
                   DVSA_QWEN_ENDPOINT=QWEN_URL)
def test_http_error_falls_back_gracefully(requests_mock):
    requests_mock.post(QWEN_URL, status_code=500)
    assert analyzer.ask_qwen_vlm("count vehicles") == "No comment."


@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_API_KEY="secret-key",
                   DVSA_QWEN_ENDPOINT=QWEN_URL)
def test_network_timeout_falls_back_gracefully(requests_mock):
    import requests  # noqa: PLC0415

    requests_mock.post(QWEN_URL, exc=requests.exceptions.ConnectTimeout)
    assert analyzer.ask_qwen_vlm("count vehicles") == "No comment."


# ----- global on/off drives tool registration (backward compat) ------------
@override_settings(DVSA_QWEN_ENABLED=True)
def test_qwen_registered_as_tool_when_enabled():
    assert analyzer.ask_qwen_vlm in analyzer.analyzer_functions()
    assert analyzer.ask_qwen_vlm in analyzer.image_user_functions()


@override_settings(DVSA_QWEN_ENABLED=False)
def test_tool_sets_unchanged_when_disabled():
    # Backward compatible: identical to the pre-Qwen tool sets.
    assert analyzer.ask_qwen_vlm not in analyzer.analyzer_functions()
    assert analyzer.ask_qwen_vlm not in analyzer.image_user_functions()
    # The existing peer tools remain registered.
    assert analyzer.ask_perplexity in analyzer.analyzer_functions()
    assert analyzer.agentic_retrieval in analyzer.image_user_functions()
