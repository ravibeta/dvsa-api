"""Generic reasoning API — route an inference request to any registered model.

Routes (included under ``/api/reasoning/`` by ``config/urls.py``):

* ``POST /api/reasoning/<model_name>/infer`` — run a named model.
* ``POST /api/reasoning/infer``              — model auto-selected via policy
  (default ``latency_optimized``; override with ``?policy=`` or body ``policy``).
* ``GET  /api/reasoning/models``             — list discovered/registered models.

The request body is the adapter ``context`` (``frames``/``tracks``/``sensor_meta``
/``precomputed_features``/``query``). The response is the adapter's ``predict``
output verbatim, plus server telemetry (``request_id``, ``selected_model``,
``server_latency_ms``). Local adapters run under a timeout; failures return a
structured ``{error_code, message, fallback_models}`` envelope, with optional
automatic fallback to the Azure adapter when ``AZURE_REASONING_FALLBACK=true``.

This router is additive and never shadows the more specific
``/api/reasoning/foundry/`` session routes (which are included first).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from django.urls import path
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from dvsa_api.reasoning import registry
from dvsa_api.reasoning.adapter_base import new_request_id, now_ms
from dvsa_api.reasoning.errors import ModelUnavailableError, ReasoningError, error_payload
from dvsa_api.reasoning.metrics import (
    CallRecord,
    ReasoningTimeout,
    record_call,
    run_with_timeout,
)

AZURE_FALLBACK_MODEL = "azure_reasoning_fallback"


def _fallback_enabled() -> bool:
    return os.environ.get("AZURE_REASONING_FALLBACK", "").strip().lower() in (
        "1", "true", "yes", "on")


def _run_adapter(model_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Run one model under the timeout guard and emit telemetry."""
    adapter = registry.get_model(model_name)
    started = now_ms()
    request_id = new_request_id()
    status_str = "ok"
    result: Dict[str, Any] = {}
    try:
        result = run_with_timeout(lambda: adapter.predict(context))
    except ReasoningTimeout as exc:
        status_str = "timeout"
        raise ModelUnavailableError(
            f"model '{model_name}' timed out", details={"request_id": request_id}
        ) from exc
    finally:
        meta = (result or {}).get("metadata", {}) if isinstance(result, dict) else {}
        record_call(CallRecord(
            request_id=request_id,
            model_name=model_name,
            model_version=meta.get("version"),
            latency_ms=max(now_ms() - started, 0),
            tokens=meta.get("tokens"),
            reasoning_effort=meta.get("reasoning_effort")
            or meta.get("foundry_reasoning_effort"),
            status=status_str,
        ))
    return result


class _ReasoningBaseView(APIView):
    # Open by default with an in-memory registry; tighten to IsAuthenticated for
    # multi-tenant deployments.
    permission_classes = [permissions.AllowAny]

    def _telemetry(self, request_id: str, model_name: str, started: int) -> Dict[str, Any]:
        return {
            "request_id": request_id,
            "selected_model": model_name,
            "server_latency_ms": max(now_ms() - started, 0),
        }

    def _infer(self, request, model_name: str) -> Response:
        started = now_ms()
        request_id = new_request_id()
        context = dict(request.data or {})
        fallbacks = self._fallback_chain(model_name)
        try:
            result = _run_adapter(model_name, context)
        except ReasoningError as exc:
            return self._maybe_fallback(request, model_name, context, exc, started,
                                        request_id, fallbacks)
        result = dict(result)
        result.update(self._telemetry(request_id, model_name, started))
        return Response(result, status=status.HTTP_200_OK)

    def _fallback_chain(self, model_name: str) -> List[str]:
        chain: List[str] = []
        if _fallback_enabled() and model_name != AZURE_FALLBACK_MODEL:
            chain.append(AZURE_FALLBACK_MODEL)
        return chain

    def _maybe_fallback(self, request, model_name, context, exc, started,
                        request_id, fallbacks) -> Response:
        # Automatic fallback to the Azure adapter on model unavailability.
        if isinstance(exc, ModelUnavailableError) and _fallback_enabled() \
                and model_name != AZURE_FALLBACK_MODEL:
            try:
                from dvsa_api.reasoning.azure_adapter import (  # noqa: PLC0415
                    AzureReasoningAdapter)
                registry.register_instance(
                    AZURE_FALLBACK_MODEL, AzureReasoningAdapter())
                result = _run_adapter(AZURE_FALLBACK_MODEL, context)
                result = dict(result)
                result.update(self._telemetry(
                    request_id, AZURE_FALLBACK_MODEL, started))
                result["fell_back_from"] = model_name
                return Response(result, status=status.HTTP_200_OK)
            except ReasoningError:
                pass
        payload = error_payload(exc)
        payload["fallback_models"] = fallbacks
        code = getattr(exc, "http_status", status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(payload, status=code)


class ReasoningModelsView(_ReasoningBaseView):
    """``GET /models`` — list registered/discovered reasoning models."""

    def get(self, request):
        manifests = {m.name: {
            "version": m.version, "type": m.type,
            "capabilities": m.capabilities, "provider": m.provider,
        } for m in registry.list_manifests()}
        return Response(
            {"models": registry.list_models(), "manifests": manifests},
            status=status.HTTP_200_OK)


class ReasoningInferByNameView(_ReasoningBaseView):
    """``POST /<model_name>/infer`` — run a specific model."""

    def post(self, request, model_name: str):
        return self._infer(request, model_name)


class ReasoningInferAutoView(_ReasoningBaseView):
    """``POST /infer`` — auto-select a model via policy, then run it."""

    def post(self, request):
        context = dict(request.data or {})
        policy = (request.query_params.get("policy")
                  or context.get("policy") or "latency_optimized")
        try:
            model_name = registry.select_model(policy, context)
        except ReasoningError as exc:
            payload = error_payload(exc)
            payload["fallback_models"] = []
            return Response(payload, status=getattr(exc, "http_status", 500))
        return self._infer(request, model_name)


app_name = "reasoning"

urlpatterns = [
    path("models", ReasoningModelsView.as_view(), name="models"),
    path("infer", ReasoningInferAutoView.as_view(), name="infer_auto"),
    path("<str:model_name>/infer", ReasoningInferByNameView.as_view(),
         name="infer_by_name"),
]
