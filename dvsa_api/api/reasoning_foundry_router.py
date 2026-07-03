"""DRF endpoints for the Azure Foundry reasoning session lifecycle.

Routes (included under ``/api/reasoning/foundry/`` by ``config/urls.py``):

* ``POST /session``                     — create a session.
* ``GET  /session/<session_id>``        — session status + telemetry.
* ``POST /session/<session_id>/infer``  — run inference against the session.
* ``POST /session/<session_id>/teardown`` — release resources immediately.

All errors map to a structured ``{error_code, message, details}`` body via
:func:`~dvsa_api.reasoning.errors.error_payload`. A light in-memory rate limiter
guards against per-session / per-user abuse. The manager runs offline (dry-run)
out of the box, so these endpoints are fully exercisable with no cloud creds.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from django.urls import path
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from dvsa_api.reasoning.azure_foundry_adapter import AzureFoundryAdapter
from dvsa_api.reasoning.azure_session_manager import get_session_manager
from dvsa_api.reasoning.errors import ReasoningError, error_payload

# --- simple in-memory sliding-window rate limiter --------------------------
_RATE_MAX = 60          # max requests ...
_RATE_WINDOW_S = 60.0   # ... per this many seconds, per key
_hits: Dict[str, Deque[float]] = defaultdict(deque)
_hits_lock = threading.Lock()


def _rate_limited(key: str) -> bool:
    """Return True when ``key`` has exceeded the request budget."""
    now = time.monotonic()
    with _hits_lock:
        q = _hits[key]
        while q and (now - q[0]) > _RATE_WINDOW_S:
            q.popleft()
        if len(q) >= _RATE_MAX:
            return True
        q.append(now)
        return False


def _user_key(request) -> str:
    user = getattr(request, "user", None)
    if user is not None and getattr(user, "is_authenticated", False):
        return f"user:{user.pk}"
    return f"ip:{request.META.get('REMOTE_ADDR', 'anon')}"


def _error_response(exc: Exception) -> Response:
    code = getattr(exc, "http_status", status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Response(error_payload(exc), status=code)


class _FoundryBaseView(APIView):
    # Demo/default: open access with an in-memory session store. Tighten to
    # IsAuthenticated (and a persistent store) for multi-tenant deployments.
    permission_classes = [permissions.AllowAny]

    def _guard_rate(self, request, extra: Optional[str] = None) -> Optional[Response]:
        key = _user_key(request) + (f":{extra}" if extra else "")
        if _rate_limited(key):
            return Response(
                {"error_code": "rate_limited",
                 "message": "too many requests; slow down", "details":
                 {"limit": _RATE_MAX, "window_s": _RATE_WINDOW_S}},
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        return None


class FoundrySessionCreateView(_FoundryBaseView):
    """``POST /session`` — provision (or dry-run) a reasoning session."""

    def post(self, request):
        limited = self._guard_rate(request)
        if limited:
            return limited
        data = request.data or {}
        model_name = data.get("model_name")
        if not model_name:
            return Response(
                {"error_code": "invalid_request",
                 "message": "'model_name' is required", "details": {}},
                status=status.HTTP_400_BAD_REQUEST)
        user = getattr(request, "user", None)
        user_id = user.pk if (user and getattr(user, "is_authenticated", False)) else None
        try:
            session = get_session_manager().create_session(
                model_name=model_name,
                user_id=user_id,
                max_duration_minutes=data.get("max_duration_minutes"),
                max_cost_usd=data.get("max_cost_usd"),
                private_networking=bool(data.get("private_networking", False)),
                tags=data.get("tags") or {},
            )
        except ReasoningError as exc:
            return _error_response(exc)
        body = {
            "session_id": session.session_id,
            "endpoint": session.endpoint_url,
            "expires_at": session.expires_at.isoformat(),
            "cost_estimate": session.cost_estimate_usd,
            "status": session.status,
            "provisioner": session.provisioner,
        }
        return Response(body, status=status.HTTP_201_CREATED)


class FoundrySessionDetailView(_FoundryBaseView):
    """``GET /session/<id>`` — status + telemetry for a session."""

    def get(self, request, session_id: str):
        try:
            session = get_session_manager().get_session(session_id)
        except ReasoningError as exc:
            return _error_response(exc)
        return Response(session.to_public_dict(), status=status.HTTP_200_OK)


class FoundrySessionInferView(_FoundryBaseView):
    """``POST /session/<id>/infer`` — run inference against the session."""

    def post(self, request, session_id: str):
        limited = self._guard_rate(request, extra=session_id)
        if limited:
            return limited
        context = dict(request.data or {})
        adapter = AzureFoundryAdapter(session_id=session_id)
        try:
            result = adapter.predict(context)
        except ReasoningError as exc:
            return _error_response(exc)
        return Response(result, status=status.HTTP_200_OK)


class FoundrySessionTeardownView(_FoundryBaseView):
    """``POST /session/<id>/teardown`` — idempotent immediate teardown."""

    def post(self, request, session_id: str):
        try:
            result = get_session_manager().teardown_session(
                session_id, reason="api")
        except ReasoningError as exc:
            return _error_response(exc)
        return Response(result, status=status.HTTP_200_OK)


app_name = "reasoning_foundry"

urlpatterns = [
    path("session", FoundrySessionCreateView.as_view(), name="session_create"),
    path("session/<str:session_id>", FoundrySessionDetailView.as_view(),
         name="session_detail"),
    path("session/<str:session_id>/infer", FoundrySessionInferView.as_view(),
         name="session_infer"),
    path("session/<str:session_id>/teardown", FoundrySessionTeardownView.as_view(),
         name="session_teardown"),
]
