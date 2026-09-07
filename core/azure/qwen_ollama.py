"""Local Ollama baseline for ``qwen2.5vl:7b`` — a raw VLM answer, no agent.

This backs the ``/api/v1/videos/baseline-test/`` endpoint, which mirrors the
regular chat endpoint's request but bypasses all DVSA agentic/RAG synthesis:
it sends the user's question (and optional image) straight to a locally hosted
``qwen2.5vl:7b`` via `Ollama <https://ollama.com>`_ and returns only the model's
reply. That makes it a like-for-like baseline you can compare against the
agentic path's answer.

The call reproduces ``local-serve-and-query-qwen.py``: an ``ollama.Client``
pointed at the local server, ``client.chat(model=..., messages=..., options=...)``,
and ``response["message"]["content"]`` as the answer. The whole Ollama
interaction lives inside :func:`_default_client_factory`, and
:func:`run_ollama_qwen` accepts a ``client_factory`` override, so tests exercise
the routing/fallback logic fully offline — no Ollama server or ``ollama`` package
required — mirroring the ``generator_factory`` seam in :mod:`core.azure.qwen_onnx`.

Any missing runtime, unreachable server, or error degrades to a benign
``"No comment."`` so the endpoint stays callable on a box without Ollama.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("apps.azure")

# Generation options mirror the demonstrated local-serve-and-query script:
# deterministic sampling, room to elaborate, and a wide context window.
OLLAMA_OPTIONS = {
    "temperature": 0.0,   # eliminate random sampling path bias
    "num_predict": 100,   # give the model room to elaborate
    "num_ctx": 8192,      # expand context to avoid aggressive clipping
}

# ``(host) -> client`` where ``client`` exposes ``.chat(model, messages, options)``.
ClientFactory = Callable[[str], Any]


def _default_client_factory(host: str) -> Any:
    """Build a real Ollama client pointed at ``host``.

    Lazily imports ``ollama`` (an optional, opt-in dependency) so the module
    stays importable without it.
    """
    import ollama  # noqa: PLC0415

    return ollama.Client(host=host)


def run_ollama_qwen(
    host: str,
    model: str,
    query_text: str,
    image_bytes: Optional[bytes] = None,
    *,
    client_factory: Optional[ClientFactory] = None,
) -> str:
    """Answer ``query_text`` with a local Ollama ``qwen2.5vl:7b`` model.

    Sends the question (and ``image_bytes`` when supplied, as the demonstrated
    script does) to the model and returns only its reply. Degrades to
    ``"No comment."`` when the host/model is unset, the ``ollama`` package is
    missing, or the server call fails — keeping the endpoint safe to hit on a
    box where Ollama has not (yet) been provisioned.
    """
    if not host or not model:
        logger.info("Ollama baseline missing host/model; no comment")
        return "No comment."

    message: dict = {"role": "user", "content": query_text}
    if image_bytes:
        message["images"] = [image_bytes]

    factory = client_factory or _default_client_factory
    try:
        client = factory(host)
        response = client.chat(
            model=model, messages=[message], options=OLLAMA_OPTIONS
        )
        text = response["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        logger.info("Ollama baseline inference failed: %s", exc)
        return "No comment."
    return (text or "").strip() or "No comment."
