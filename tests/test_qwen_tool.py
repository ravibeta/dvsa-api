"""Tests for the Qwen VLM function-tool (``core.azure.analyzer.ask_qwen_vlm``).

Fully offline: the network is patched with :mod:`unittest.mock` (no extra test
dependency) and the global on/off is driven by the ``DVSA_QWEN_ENABLED`` setting,
so no Azure credentials are needed. The suite covers the request/response
contract, graceful fallbacks, and — most importantly — that the global flag
toggles whether Qwen joins the agent tool set (the pre-Qwen behaviour is
preserved byte-for-byte when disabled).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import override_settings

from core.azure import analyzer

QWEN_URL = "https://found-vision-1.services.ai.azure.com/openai/v1/chat/completions"


def _resp(json_body):
    """A stand-in ``requests.Response`` with a no-op ``raise_for_status``."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_body
    return resp


# ----- offline / unconfigured fallbacks ------------------------------------
@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_API_KEY=None)
def test_returns_no_comment_when_unconfigured():
    # No key -> benign fallback, and no HTTP call is attempted.
    with patch("requests.post") as post:
        assert analyzer.ask_qwen_vlm("what is in the scene?") == "No comment."
        post.assert_not_called()


@override_settings(DVSA_QWEN_ENABLED=False, DVSA_QWEN_API_KEY="secret-key")
def test_returns_no_comment_when_globally_disabled():
    # Global kill-switch: even with a key, Qwen stays silent and makes no call.
    with patch("requests.post") as post:
        assert analyzer.ask_qwen_vlm("what is in the scene?") == "No comment."
        post.assert_not_called()


# ----- request/response contract -------------------------------------------
@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_API_KEY="secret-key",
                   DVSA_QWEN_ENDPOINT=QWEN_URL, DVSA_QWEN_MODEL="qwen--qwen3.5-0.8b")
def test_calls_endpoint_with_openai_chat_shape():
    body = {"choices": [{"message": {"role": "assistant",
                                     "content": "two cars and a truck"}}]}
    with patch("requests.post", return_value=_resp(body)) as post:
        out = analyzer.ask_qwen_vlm("count vehicles")

    assert out == "two cars and a truck"
    args, kwargs = post.call_args
    assert args[0] == QWEN_URL
    assert kwargs["headers"]["Authorization"] == "Bearer secret-key"
    sent = kwargs["json"]
    assert sent["model"] == "qwen--qwen3.5-0.8b"
    assert sent["temperature"] == 0.6
    assert sent["top_p"] == 0.95
    assert sent["messages"][-1] == {"role": "user", "content": "count vehicles"}


@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_API_KEY="secret-key",
                   DVSA_QWEN_ENDPOINT=QWEN_URL)
def test_http_error_falls_back_gracefully():
    import requests  # noqa: PLC0415

    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    with patch("requests.post", return_value=resp):
        assert analyzer.ask_qwen_vlm("count vehicles") == "No comment."


@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_API_KEY="secret-key",
                   DVSA_QWEN_ENDPOINT=QWEN_URL)
def test_network_timeout_falls_back_gracefully():
    import requests  # noqa: PLC0415

    with patch("requests.post", side_effect=requests.exceptions.ConnectTimeout):
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
