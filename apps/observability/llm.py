"""LLM client abstraction for higher-level semantic commentary (Phase 4).

Phase 4 introduces model-generated commentary. To stay consistent with the rest
of the layer it is built around a small, provider-agnostic :class:`LLMClient`
interface with two implementations:

* :class:`EchoLLMClient` — deterministic, offline, no network. Used in tests and
  whenever no model is configured, so the model-backed code paths are always
  exercisable without credentials.
* :class:`AzureOpenAIChatClient` — talks to **Azure OpenAI** (the provider the
  repo already targets via ``AZURE_OPENAI_*`` settings) over its REST API using
  only the standard library — no new dependency, same approach as the OTLP
  exporter in :mod:`apps.observability.otel`.

The request building is a pure method (:meth:`AzureOpenAIChatClient.build_request`)
so it is unit-testable without a live endpoint; only :meth:`complete` does I/O.
"""

from __future__ import annotations

import abc
import json
import urllib.request
from typing import Any, Dict, Optional, Tuple


class LLMClient(abc.ABC):
    """Minimal chat-completion interface."""

    name = "base"

    @abc.abstractmethod
    def complete(self, prompt: str, *, system: Optional[str] = None) -> str:
        """Return the model's text completion for ``prompt``."""


class EchoLLMClient(LLMClient):
    """Deterministic offline client — echoes a compact digest of the prompt.

    Not a real model; it exists so the VLM/agent code paths run reproducibly in
    tests and as a safe default when no model is configured.
    """

    name = "echo"

    def complete(self, prompt: str, *, system: Optional[str] = None) -> str:
        lines = [ln.strip() for ln in (prompt or "").splitlines() if ln.strip()]
        head = lines[0] if lines else ""
        return ("[echo] " + head)[:280]


class AzureOpenAIChatClient(LLMClient):
    """Azure OpenAI chat-completions client over the REST API (stdlib only)."""

    name = "azure"

    def __init__(
        self,
        endpoint: Optional[str],
        api_key: Optional[str],
        deployment: Optional[str],
        *,
        api_version: str = "2024-06-01",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 30.0,
    ) -> None:
        if not (endpoint and api_key and deployment):
            raise ValueError(
                "AzureOpenAIChatClient requires AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_GPT_DEPLOYMENT"
            )
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def build_request(
        self, prompt: str, system: Optional[str] = None
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Return ``(url, headers, body)`` for a chat completion (pure)."""

        url = "%s/openai/deployments/%s/chat/completions?api-version=%s" % (
            self.endpoint, self.deployment, self.api_version,
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        return url, headers, body

    def complete(self, prompt: str, *, system: Optional[str] = None) -> str:
        url, headers, body = self.build_request(prompt, system)
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def get_llm_client(name: str = None) -> LLMClient:
    """Return an LLM client by name, reading Django settings when available.

    Resolution: explicit ``name`` → ``COMMENTARY_LLM`` setting → ``"echo"``.
    Valid names: ``"echo"``, ``"azure"``.
    """

    if name is None:
        try:
            from django.conf import settings

            name = getattr(settings, "COMMENTARY_LLM", "echo")
        except Exception:  # noqa: BLE001 - no Django context
            name = "echo"

    name = (name or "echo").lower()
    if name in ("echo", "none", "off"):
        return EchoLLMClient()
    if name in ("azure", "azure_openai", "openai"):
        from django.conf import settings

        return AzureOpenAIChatClient(
            getattr(settings, "AZURE_OPENAI_ENDPOINT", None),
            getattr(settings, "AZURE_OPENAI_API_KEY", None),
            getattr(settings, "AZURE_OPENAI_GPT_DEPLOYMENT", None),
            api_version=getattr(settings, "AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )
    raise ValueError("Unknown LLM client '%s'" % name)
