"""REST endpoints for the MCP (Multi-Agent Control Plane) subsystem.

Routes (included under ``/api/mcp/`` by ``config/urls.py``):

* ``POST /api/mcp/missions``                     — create + start a mission.
* ``GET  /api/mcp/missions/<mission_id>``        — mission status + task graph.
* ``POST /api/mcp/missions/<mission_id>/commands`` — pause/resume/cancel/escalate.
* ``GET  /api/mcp/agents``                       — list agents + health.
* ``POST /api/mcp/agents/<agent_name>/invoke``   — direct-invoke an agent (testing).

Every endpoint is gated by ``ENABLE_MCP`` (returns ``503 mcp_disabled`` when off).
Errors map to a structured ``{error_code, message, details, fallback_actions}``
body. A light in-memory rate limiter guards against abuse. The runtime is fully
in-memory/offline by default, so these endpoints work with no external services.
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

from dvsa_api.mcp import is_mcp_enabled, registry
from dvsa_api.mcp.adapter_base import new_id, now_ms
from dvsa_api.mcp.errors import MCPError, error_payload
from dvsa_api.mcp.metrics import get_metrics
from dvsa_api.mcp.session_manager import get_session_manager

# --- simple in-memory sliding-window rate limiter --------------------------
_RATE_MAX = 120
_RATE_WINDOW_S = 60.0
_hits: Dict[str, Deque[float]] = defaultdict(deque)
_hits_lock = threading.Lock()


def _rate_limited(key: str) -> bool:
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


def _error(exc: Exception) -> Response:
    code = getattr(exc, "http_status", status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Response(error_payload(exc), status=code)


class _MCPBaseView(APIView):
    permission_classes = [permissions.AllowAny]

    def _guard(self, request, extra: Optional[str] = None) -> Optional[Response]:
        """Return a Response to short-circuit (disabled / rate-limited), else None."""
        if not is_mcp_enabled():
            return Response(
                {"error_code": "mcp_disabled",
                 "message": "MCP is disabled; set ENABLE_MCP=true to enable it",
                 "details": {}, "fallback_actions": []},
                status=status.HTTP_503_SERVICE_UNAVAILABLE)
        key = _user_key(request) + (f":{extra}" if extra else "")
        if _rate_limited(key):
            return Response(
                {"error_code": "rate_limited", "message": "too many requests",
                 "details": {"limit": _RATE_MAX, "window_s": _RATE_WINDOW_S},
                 "fallback_actions": []},
                status=status.HTTP_429_TOO_MANY_REQUESTS)
        return None


class MissionCreateView(_MCPBaseView):
    """``POST /missions`` — create + start a mission from a template or inline spec."""

    def post(self, request):
        guard = self._guard(request)
        if guard:
            return guard
        body = request.data or {}
        template = body.get("mission") or body.get("template") or body.get("mission_id")
        if template is None and "tasks" in body:
            template = body  # inline mission spec
        if template is None:
            return Response(
                {"error_code": "invalid_request",
                 "message": "provide 'mission' (name/spec) or an inline 'tasks' list",
                 "details": {}, "fallback_actions": []},
                status=status.HTTP_400_BAD_REQUEST)
        started = now_ms()
        request_id = new_id("req-")
        try:
            session = get_session_manager().create_mission(
                template, context=body.get("context") or {},
                run=bool(body.get("run", True)))
        except MCPError as exc:
            return _error(exc)
        payload = session.public_dict()
        payload.update({"request_id": request_id,
                        "server_latency_ms": now_ms() - started})
        return Response(payload, status=status.HTTP_201_CREATED)


class MissionDetailView(_MCPBaseView):
    """``GET /missions/<id>`` — mission status + task graph."""

    def get(self, request, mission_id: str):
        guard = self._guard(request)
        if guard:
            return guard
        try:
            session = get_session_manager().get_mission(mission_id)
        except MCPError as exc:
            return _error(exc)
        return Response(session.public_dict(), status=status.HTTP_200_OK)


class MissionCommandView(_MCPBaseView):
    """``POST /missions/<id>/commands`` — pause/resume/cancel/escalate."""

    def post(self, request, mission_id: str):
        guard = self._guard(request, extra=mission_id)
        if guard:
            return guard
        command = (request.data or {}).get("command")
        if not command:
            return Response(
                {"error_code": "invalid_request", "message": "'command' is required",
                 "details": {}, "fallback_actions": ["pause", "resume", "cancel",
                                                     "escalate"]},
                status=status.HTTP_400_BAD_REQUEST)
        try:
            result = get_session_manager().send_command(
                mission_id, command, (request.data or {}).get("args") or {})
        except MCPError as exc:
            return _error(exc)
        return Response(result, status=status.HTTP_200_OK)


class AgentListView(_MCPBaseView):
    """``GET /agents`` — list discovered agents + health + metrics."""

    def get(self, request):
        guard = self._guard(request)
        if guard:
            return guard
        metrics = get_metrics()
        agents = []
        for manifest in registry.list_manifests():
            agents.append({
                **manifest.public_dict(),
                "avg_latency_ms": metrics.avg_latency_ms(manifest.name),
            })
        return Response(
            {"agents": agents, "metrics": metrics.snapshot()},
            status=status.HTTP_200_OK)


class AgentInvokeView(_MCPBaseView):
    """``POST /agents/<name>/invoke`` — direct-invoke one agent (testing)."""

    def post(self, request, agent_name: str):
        guard = self._guard(request, extra=agent_name)
        if guard:
            return guard
        body = request.data or {}
        started = now_ms()
        request_id = new_id("req-")
        task = body.get("task") or {"task_id": new_id("task-"),
                                    "type": body.get("type", "invoke"),
                                    "payload": body.get("payload") or {}}
        context = body.get("context") or {}
        try:
            agent = registry.get_agent(agent_name)
            result = agent.handle_task(task, context)
        except MCPError as exc:
            return _error(exc)
        except Exception as exc:  # noqa: BLE001 - agent raised
            return Response(
                {"error_code": "agent_error", "message": str(exc), "details": {},
                 "fallback_actions": []},
                status=status.HTTP_502_BAD_GATEWAY)
        return Response(
            {"request_id": request_id, "selected_agent": agent_name,
             "server_latency_ms": now_ms() - started, "result": result},
            status=status.HTTP_200_OK)


app_name = "mcp"

urlpatterns = [
    path("missions", MissionCreateView.as_view(), name="mission_create"),
    path("missions/<str:mission_id>", MissionDetailView.as_view(),
         name="mission_detail"),
    path("missions/<str:mission_id>/commands", MissionCommandView.as_view(),
         name="mission_commands"),
    path("agents", AgentListView.as_view(), name="agents"),
    path("agents/<str:agent_name>/invoke", AgentInvokeView.as_view(),
         name="agent_invoke"),
]
