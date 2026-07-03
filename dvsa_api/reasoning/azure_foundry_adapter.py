"""Azure Foundry reasoning adapter — the ``ReasoningModelAdapter`` for Foundry.

Two modes, selected automatically:

* **Managed session mode** (default) — bound to a ``session_id`` created via
  :class:`~dvsa_api.reasoning.azure_session_manager.AzureFoundrySessionManager`.
  ``predict`` validates the session, resolves its endpoint + Key Vault key, runs
  inference and accrues usage/cost back onto the session.
* **Direct endpoint mode** — when ``AZURE_FOUNDRY_ENDPOINT`` /
  ``AZURE_FOUNDRY_KEY`` are set (the spec's ``AZURE_FOUNDY_*`` spelling is also
  accepted), the adapter talks to that endpoint with no provisioning.

The network call is isolated in :meth:`_call_foundry` and is **never made** for
synthetic (dry-run) endpoints or when ``AZURE_FOUNDRY_OFFLINE`` is set, so tests
and fresh checkouts get a deterministic response with zero network access. Every
response carries telemetry (``request_id``, ``session_id``, ``latency_ms``,
``tokens``, ``cost_estimate_ms``, ``foundry_reasoning_effort``) in ``metadata``.
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
from .azure_session_manager import (
    AzureFoundrySessionManager,
    get_session_manager,
    sku_hourly_rate,
)
from .errors import ModelUnavailableError

logger = logging.getLogger("dvsa_api.reasoning")

ADAPTER_VERSION = "1.0.0"
DEFAULT_TIMEOUT_MS = int(os.environ.get("FOUNDRY_REQUEST_TIMEOUT_MS", "30000"))
DEFAULT_RETRIES = int(os.environ.get("FOUNDRY_REQUEST_RETRIES", "2"))


def _first_env(*names: str) -> Optional[str]:
    """Return the first set env var among ``names`` (tolerates the spec typo)."""
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return None


def _is_synthetic_endpoint(url: str) -> bool:
    return (not url) or ".example-foundry.local" in url or url.endswith(".local")


def _offline_forced() -> bool:
    return os.environ.get("AZURE_FOUNDRY_OFFLINE", "").strip().lower() in (
        "1", "true", "yes", "on")


class AzureFoundryAdapter(ReasoningModelAdapter):
    """Adapter that runs reasoning against an Azure Foundry endpoint."""

    name = "azure_foundry"

    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None,
        manager: Optional[AzureFoundrySessionManager] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        self.session_id = session_id
        self._explicit_model = model_name
        self.manager = manager or get_session_manager()
        self.timeout_ms = timeout_ms
        self.retries = retries
        # Direct-endpoint config (spec uses AZURE_FOUNDY_*; accept both spellings).
        self.direct_endpoint = _first_env(
            "AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDY_ENDPOINT")
        self.direct_key = _first_env("AZURE_FOUNDRY_KEY", "AZURE_FOUNDY_KEY")

    # ----- mode resolution ----------------------------------------------
    @property
    def is_direct(self) -> bool:
        """Direct mode when no managed session is bound but a direct endpoint is."""
        return self.session_id is None and bool(self.direct_endpoint)

    def _resolve_target(self, *, force: bool) -> Dict[str, Any]:
        """Return ``{endpoint, key, model_name, session}`` for the active mode."""
        if self.session_id is not None:
            session = self.manager.check_can_infer(self.session_id, force=force)
            key = self.manager.sdk.get_secret(session.key_vault_secret_name)
            return {
                "endpoint": session.endpoint_url,
                "key": key,
                "model_name": session.model_name,
                "session": session,
            }
        if self.direct_endpoint:
            return {
                "endpoint": self.direct_endpoint,
                "key": self.direct_key or "",
                "model_name": self._explicit_model or os.environ.get(
                    "AZURE_FOUNDRY_MODEL", "foundry-reasoning"),
                "session": None,
            }
        raise ModelUnavailableError(
            "adapter has neither a managed session nor a direct endpoint configured")

    # ----- contract methods ---------------------------------------------
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run one reasoning request and return the contract response shape."""
        request_id = new_request_id()
        force = bool(context.get("force"))
        started = now_ms()
        target = self._resolve_target(force=force)
        endpoint = target["endpoint"]
        model_name = target["model_name"]

        try:
            raw = self._call_foundry(endpoint, target["key"], context, model_name)
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"foundry inference failed: {exc}",
                details={"request_id": request_id, "endpoint_synthetic":
                         _is_synthetic_endpoint(endpoint)},
            ) from exc

        latency_ms = max(now_ms() - started, 0)
        tokens = raw.get("tokens")
        reasoning_effort = (
            raw.get("reasoning_effort")
            or context.get("reasoning_effort")
        )
        # Approx marginal cost of this call, in USD, from SKU hourly rate.
        cost_estimate_ms = round(sku_hourly_rate(model_name) * (latency_ms / 3_600_000.0), 8)

        metadata = build_metadata(
            model=model_name,
            version=ADAPTER_VERSION,
            latency_ms=latency_ms,
            tokens=tokens,
            request_id=request_id,
            session_id=self.session_id,
            provider="azure_foundry",
            mode="direct" if self.session_id is None else "managed",
            cost_estimate_ms=cost_estimate_ms,
            foundry_reasoning_effort=reasoning_effort,
        )
        response = ensure_response_shape({
            "actions": raw.get("actions", []),
            "reasoning_trace": raw.get("reasoning_trace", []),
            "metadata": metadata,
        })

        # Accrue usage/cost back onto the managed session (may hard-stop it).
        if self.session_id is not None:
            self.manager.record_usage(
                self.session_id, tokens=tokens or 0, latency_ms=latency_ms)
        return response

    def health_check(self) -> Dict[str, Any]:
        """Report session/endpoint liveness without running inference."""
        try:
            if self.session_id is not None:
                session = self.manager.get_session(self.session_id)
                if session.is_expired():
                    return {"status": "error", "reason": "session_expired",
                            "session_id": self.session_id}
                return {"status": "ok", "mode": "managed",
                        "session_id": self.session_id, "endpoint":
                        session.endpoint_url, "model": session.model_name}
            if self.direct_endpoint:
                return {"status": "ok", "mode": "direct",
                        "endpoint": self.direct_endpoint}
            return {"status": "error", "reason": "not_configured"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": str(exc)}

    # ----- inference plumbing -------------------------------------------
    def _call_foundry(
        self, endpoint: str, key: str, context: Dict[str, Any], model_name: str,
    ) -> Dict[str, Any]:
        """POST ``context`` to the Foundry reasoning endpoint (or synthesize).

        Synthetic/offline endpoints short-circuit to a deterministic response so
        the pipeline is exercisable without any network access.
        """
        if _offline_forced() or _is_synthetic_endpoint(endpoint):
            return _synthetic_reasoning(context, model_name)

        payload = json.dumps(_build_foundry_request(context, model_name)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["api-key"] = key
            headers["Authorization"] = f"Bearer {key}"
        timeout_s = max(self.timeout_ms / 1000.0, 0.1)

        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    endpoint, data=payload, method="POST", headers=headers)
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return _map_foundry_response(data, context, model_name)
            except urllib.error.HTTPError as exc:
                # 4xx are not retryable; surface immediately.
                if 400 <= exc.code < 500:
                    raise ModelUnavailableError(
                        f"foundry returned HTTP {exc.code}",
                        details={"status": exc.code}) from exc
                last_exc = exc
            except Exception as exc:  # noqa: BLE001 - network/timeout, retry
                last_exc = exc
            logger.warning("foundry call attempt %d failed: %s", attempt + 1, last_exc)
        raise ModelUnavailableError(
            f"foundry endpoint unreachable after {self.retries + 1} attempts",
            details={"error": str(last_exc)})


# --- request/response mapping ----------------------------------------------
def _build_foundry_request(context: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Wrap the dvsa ``context`` into a Foundry reasoning request body."""
    query = context.get("query", "Analyse the aerial scene for anomalies.")
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
                "query": query,
                "tracks": context.get("tracks", []),
                "frames": context.get("frames", []),
                "sensor_meta": context.get("sensor_meta", {}),
            })},
        ],
    }


def _map_foundry_response(
    data: Dict[str, Any], context: Dict[str, Any], model_name: str,
) -> Dict[str, Any]:
    """Map a raw Foundry response to ``{actions, reasoning_trace, tokens, ...}``.

    Tolerates both a native ``{actions, reasoning_trace}`` body and an
    OpenAI-style ``{choices:[{message:{content}}], usage:{total_tokens}}`` body.
    """
    if "actions" in data or "reasoning_trace" in data:
        return {
            "actions": data.get("actions", []),
            "reasoning_trace": data.get("reasoning_trace", []),
            "tokens": (data.get("usage") or {}).get("total_tokens"),
            "reasoning_effort": data.get("reasoning_effort"),
        }
    # OpenAI-style: try to parse structured JSON out of the message content.
    content = ""
    choices = data.get("choices") or []
    if choices:
        content = (choices[0].get("message") or {}).get("content", "") or ""
    parsed: Dict[str, Any] = {}
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


def _synthetic_reasoning(context: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Deterministic offline reasoning result (no ML, no network).

    Flags an ``accident`` anomaly when two or more tracks are present (a stand-in
    for the intersection/deceleration heuristic), otherwise reports ``nominal``.
    """
    tracks = context.get("tracks") or []
    query = context.get("query", "")
    trace = [
        f"received query: {query!r}",
        f"observed {len(tracks)} object track(s)",
    ]
    if len(tracks) >= 2:
        trace.append("two or more tracks converge -> potential collision")
        actions = [{
            "type": "anomaly",
            "label": "accident",
            "confidence": 0.95,
            "bbox": context.get("bbox", [0, 0, 0, 0]),
        }]
    else:
        trace.append("insufficient converging tracks -> nominal scene")
        actions = [{"type": "label", "label": "nominal", "confidence": 0.60}]
    # Cheap deterministic token estimate so cost accrual is exercised offline.
    tokens = 24 + 6 * len(tracks) + len(str(query))
    return {
        "actions": actions,
        "reasoning_trace": trace,
        "tokens": tokens,
        "reasoning_effort": context.get("reasoning_effort", "medium"),
    }


def build_adapter_for_session(session_id: str) -> AzureFoundryAdapter:
    """Convenience factory used by the registry / router for a managed session."""
    return AzureFoundryAdapter(session_id=session_id)
