"""Generic Azure Foundry / OpenAI reasoning adapter (env-configured).

This is the *simple* remote adapter used when a model's ``manifest.json`` declares
``type: "remote"`` with ``provider: "azure"``, or when ``AZURE_REASONING_DEFAULT``
selects Azure as the default reasoning backend. It reads only environment
variables — **no secrets are embedded in code**:

* ``AZURE_REASONING_ENDPOINT`` — chat/completions URL of the reasoning deployment.
* ``AZURE_REASONING_KEY``      — API key (sent as ``api-key`` + bearer header).
* ``AZURE_REASONING_DEFAULT``  — model/deployment name to use by default.

For the richer *ephemeral session* lifecycle (dynamic provisioning, Key Vault,
cost caps) use :class:`~dvsa_api.reasoning.azure_foundry_adapter.AzureFoundryAdapter`
instead. This adapter is intentionally stateless and offline-safe: with no
endpoint configured (or ``AZURE_REASONING_OFFLINE=1``) it returns a deterministic
synthetic result so CI never touches the network.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .adapter_base import (
    ReasoningModelAdapter,
    build_metadata,
    ensure_response_shape,
    new_request_id,
    now_ms,
)
from .errors import ModelUnavailableError

logger = logging.getLogger("dvsa_api.reasoning")

ADAPTER_VERSION = "1.0.0"
DEFAULT_TIMEOUT_MS = int(os.environ.get("AZURE_REASONING_TIMEOUT_MS", "30000"))
DEFAULT_RETRIES = int(os.environ.get("AZURE_REASONING_RETRIES", "2"))


def _offline() -> bool:
    return os.environ.get("AZURE_REASONING_OFFLINE", "").strip().lower() in (
        "1", "true", "yes", "on")


class AzureReasoningAdapter(ReasoningModelAdapter):
    """Stateless adapter over an Azure-hosted reasoning endpoint."""

    name = "azure_reasoning"

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        config = config or {}
        self.endpoint = endpoint or config.get("endpoint") or os.environ.get(
            "AZURE_REASONING_ENDPOINT")
        # Key is read from env only; never persisted or logged.
        self.api_key = api_key or os.environ.get("AZURE_REASONING_KEY")
        self.model_name = (
            model_name or config.get("model")
            or os.environ.get("AZURE_REASONING_DEFAULT", "azure-reasoning"))
        self.timeout_ms = timeout_ms
        self.retries = retries

    # ----- contract -----------------------------------------------------
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        request_id = new_request_id()
        started = now_ms()
        raw = self._call(context)
        latency_ms = max(now_ms() - started, 0)
        metadata = build_metadata(
            model=self.model_name,
            version=ADAPTER_VERSION,
            latency_ms=latency_ms,
            tokens=raw.get("tokens"),
            request_id=request_id,
            provider="azure",
            mode="offline" if (self._offline_effective()) else "remote",
            reasoning_effort=raw.get("reasoning_effort")
            or context.get("reasoning_effort"),
        )
        return ensure_response_shape({
            "actions": raw.get("actions", []),
            "reasoning_trace": raw.get("reasoning_trace", []),
            "metadata": metadata,
        })

    def health_check(self) -> Dict[str, Any]:
        if self._offline_effective():
            return {"status": "ok", "mode": "offline", "model": self.model_name}
        return {"status": "ok", "mode": "remote", "endpoint": self.endpoint,
                "model": self.model_name}

    # ----- internals ----------------------------------------------------
    def _offline_effective(self) -> bool:
        return _offline() or not self.endpoint

    def _call(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._offline_effective():
            return _synthetic(context, self.model_name)

        body = json.dumps(_build_request(context, self.model_name)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout_s = max(self.timeout_ms / 1000.0, 0.1)

        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.endpoint, data=body, method="POST", headers=headers)
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return _map_response(data, context)
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    raise ModelUnavailableError(
                        f"azure reasoning returned HTTP {exc.code}",
                        details={"status": exc.code}) from exc
                last_exc = exc
            except Exception as exc:  # noqa: BLE001 - network/timeout, retry
                last_exc = exc
            logger.warning("azure reasoning attempt %d failed: %s",
                           attempt + 1, last_exc)
        raise ModelUnavailableError(
            f"azure reasoning endpoint unreachable after {self.retries + 1} attempts",
            details={"error": str(last_exc)})


def _build_request(context: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Wrap the dvsa ``context`` into an OpenAI-style chat request."""
    system = (
        "You are an aerial drone video reasoning model. Identify anomalies "
        "(e.g. accidents) from tracks/frames metadata and return structured "
        "actions with an explainable reasoning trace."
    )
    return {
        "model": model_name,
        "reasoning_effort": context.get("reasoning_effort", "medium"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "query": context.get("query", "Analyse the aerial scene."),
                "tracks": context.get("tracks", []),
                "frames": context.get("frames", []),
                "sensor_meta": context.get("sensor_meta", {}),
            })},
        ],
    }


def _map_response(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Map native or OpenAI-style responses to the internal raw shape."""
    if "actions" in data or "reasoning_trace" in data:
        return {
            "actions": data.get("actions", []),
            "reasoning_trace": data.get("reasoning_trace", []),
            "tokens": (data.get("usage") or {}).get("total_tokens"),
            "reasoning_effort": data.get("reasoning_effort"),
        }
    content = ""
    choices = data.get("choices") or []
    if choices:
        content = (choices[0].get("message") or {}).get("content", "") or ""
    try:
        parsed = json.loads(content) if content else {}
    except (ValueError, TypeError):
        parsed = {}
    return {
        "actions": parsed.get("actions", []),
        "reasoning_trace": parsed.get("reasoning_trace",
                                      [content] if content else []),
        "tokens": (data.get("usage") or {}).get("total_tokens"),
        "reasoning_effort": data.get("reasoning_effort"),
    }


def _synthetic(context: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Deterministic offline response (no network, no ML)."""
    tracks = context.get("tracks") or []
    trace = [
        "azure reasoning (offline synthetic) engaged",
        f"observed {len(tracks)} object track(s)",
    ]
    if len(tracks) >= 2:
        actions = [{"type": "anomaly", "label": "accident", "confidence": 0.90,
                    "bbox": context.get("bbox", [0, 0, 0, 0])}]
        trace.append("converging tracks -> potential collision")
    else:
        actions = [{"type": "label", "label": "nominal", "confidence": 0.55}]
        trace.append("no converging tracks -> nominal scene")
    return {
        "actions": actions,
        "reasoning_trace": trace,
        "tokens": 20 + 5 * len(tracks),
        "reasoning_effort": context.get("reasoning_effort", "medium"),
    }
