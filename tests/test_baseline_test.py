"""Tests for the ``baseline-test`` endpoint and its Ollama runner.

Fully offline: the Ollama interaction is driven through an injected
``client_factory`` (no ``ollama`` package or server needed), and the endpoint
tests patch :func:`core.azure.qwen_ollama.run_ollama_qwen`. The suite covers the
runner contract (question/image passthrough, generation options, graceful
fallbacks) and that the endpoint mirrors ``chat/`` — same request/response shape,
same ``IsAuthenticated`` guard — while returning only the raw Qwen answer.
"""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.test import APIClient

from apps.users.models import User
from core.azure import qwen_ollama


def _recording_factory(recorder, answer="two red cars and a truck"):
    """A ``client_factory`` whose client records the chat call and replies."""

    def factory(host):
        recorder["host"] = host

        class _Client:
            def chat(self, model, messages, options):
                recorder["model"] = model
                recorder["messages"] = messages
                recorder["options"] = options
                return {"message": {"content": answer}}

        return _Client()

    return factory


# ----- run_ollama_qwen contract / fallbacks --------------------------------
def test_run_ollama_qwen_uses_injected_client_factory():
    rec = {}
    out = qwen_ollama.run_ollama_qwen(
        "http://localhost:8848", "qwen2.5vl:7b", "how many red cars?",
        client_factory=_recording_factory(rec),
    )
    assert out == "two red cars and a truck"
    assert rec["host"] == "http://localhost:8848"
    assert rec["model"] == "qwen2.5vl:7b"
    assert rec["messages"] == [{"role": "user", "content": "how many red cars?"}]
    # Deterministic sampling, matching the demonstrated local-serve script.
    assert rec["options"]["temperature"] == 0.0
    assert rec["options"]["num_ctx"] == 8192


def test_run_ollama_qwen_passes_image_bytes():
    rec = {}
    qwen_ollama.run_ollama_qwen(
        "http://localhost:8848", "qwen2.5vl:7b", "what is in the scene?",
        image_bytes=b"\xff\xd8jpegbytes",
        client_factory=_recording_factory(rec),
    )
    assert rec["messages"][0]["images"] == [b"\xff\xd8jpegbytes"]


def test_run_ollama_qwen_no_comment_without_host_or_model():
    assert qwen_ollama.run_ollama_qwen("", "qwen2.5vl:7b", "q") == "No comment."
    assert qwen_ollama.run_ollama_qwen("http://h", "", "q") == "No comment."


def test_run_ollama_qwen_falls_back_when_chat_raises():
    def boom(_host):
        class _Client:
            def chat(self, *a, **k):
                raise RuntimeError("server unreachable")

        return _Client()

    assert qwen_ollama.run_ollama_qwen(
        "http://h", "qwen2.5vl:7b", "q", client_factory=boom
    ) == "No comment."


def test_run_ollama_qwen_falls_back_when_runtime_missing():
    # Default factory lazily imports ``ollama``; absence degrades quietly.
    with patch.dict("sys.modules", {"ollama": None}):
        assert qwen_ollama.run_ollama_qwen(
            "http://h", "qwen2.5vl:7b", "q"
        ) == "No comment."


# ----- endpoint: peer of chat/, returns only the raw Qwen answer -----------
class BaselineTestEndpoint(TestCase):
    url = "/api/v1/videos/baseline-test/"

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="baseline@example.com", password="testpass123",
            first_name="Base", last_name="Line",
        )

    def test_returns_raw_qwen_answer(self):
        self.client.force_authenticate(user=self.user)
        with patch("core.azure.qwen_ollama.run_ollama_qwen",
                   return_value="two red cars") as run:
            resp = self.client.put(
                self.url, {"account_id": "2", "query": "how many red cars?"}
            )
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json() == {"text": "two red cars",
                               "imageUrl": None, "downloadUrl": None}
        run.assert_called_once()
        args = run.call_args.args
        assert args[1] == "qwen2.5vl:7b"          # configured model
        assert args[2] == "how many red cars?"     # user query passed through

    @override_settings(DVSA_OLLAMA_MODEL="ignored")
    def test_requires_query_and_account(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.put(self.url, {"account_id": "2"})
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_requires_authentication(self):
        resp = self.client.put(
            self.url, {"account_id": "2", "query": "how many red cars?"}
        )
        assert resp.status_code in (status.HTTP_401_UNAUTHORIZED,
                                    status.HTTP_403_FORBIDDEN)
