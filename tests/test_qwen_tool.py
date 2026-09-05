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
from core.azure import qwen_onnx

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


# ----- local ONNX backend (standalone, Azure-free) -------------------------
@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_BACKEND="onnx",
                   DVSA_QWEN_ONNX_MODEL_PATH="/models/qwen3.5-0.8b-onnx",
                   DVSA_QWEN_API_KEY=None)
def test_onnx_backend_routes_to_local_runner_not_azure():
    # ONNX backend answers locally and makes no HTTP call, even with no API key.
    with patch("requests.post") as post, \
            patch("core.azure.qwen_onnx.run_local_qwen",
                  return_value="local answer") as run_local:
        out = analyzer.ask_qwen_vlm("what is in the scene?")

    assert out == "local answer"
    post.assert_not_called()
    run_local.assert_called_once_with("/models/qwen3.5-0.8b-onnx",
                                      "what is in the scene?")


@override_settings(DVSA_QWEN_ENABLED=True, DVSA_QWEN_BACKEND="onnx",
                   DVSA_QWEN_ONNX_MODEL_PATH="")
def test_onnx_backend_no_comment_when_model_path_unset():
    with patch("requests.post") as post:
        assert analyzer.ask_qwen_vlm("count vehicles") == "No comment."
        post.assert_not_called()


@override_settings(DVSA_QWEN_ENABLED=False, DVSA_QWEN_BACKEND="onnx",
                   DVSA_QWEN_ONNX_MODEL_PATH="/models/qwen3.5-0.8b-onnx")
def test_onnx_backend_respects_global_disable():
    # The global kill-switch wins regardless of backend.
    with patch("core.azure.qwen_onnx.run_local_qwen") as run_local:
        assert analyzer.ask_qwen_vlm("count vehicles") == "No comment."
        run_local.assert_not_called()


def test_run_local_qwen_uses_injected_generator_factory():
    calls = {}

    def fake_factory(model_path, system_prompt, user_prompt):
        calls["args"] = (model_path, system_prompt, user_prompt)
        return "two cars and a truck"

    out = qwen_onnx.run_local_qwen(
        "/models/qwen", "count vehicles", generator_factory=fake_factory
    )
    assert out == "two cars and a truck"
    model_path, system_prompt, user_prompt = calls["args"]
    assert model_path == "/models/qwen"
    assert user_prompt == "count vehicles"
    assert "analyst" in system_prompt


def test_run_local_qwen_no_comment_without_path():
    assert qwen_onnx.run_local_qwen("", "count vehicles") == "No comment."


def test_run_local_qwen_falls_back_when_generation_raises():
    def boom(*_args):
        raise RuntimeError("model failed to load")

    assert qwen_onnx.run_local_qwen(
        "/models/qwen", "count vehicles", generator_factory=boom
    ) == "No comment."


def test_run_local_qwen_falls_back_when_runtime_missing():
    # Default factory lazily imports onnxruntime_genai; absence degrades quietly.
    with patch.dict("sys.modules", {"onnxruntime_genai": None}):
        assert qwen_onnx.run_local_qwen(
            "/models/qwen", "count vehicles"
        ) == "No comment."


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
