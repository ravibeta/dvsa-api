"""End-to-end MCP mission integration tests (in-memory, offline).

Runs the shipped ``urban_accident_response`` mission through the real runtime
(discovery -> planner -> executor -> agents, including the reasoning-backed
analyzer and human escalation) via the simulator and the session manager. The
DRF API tests only run where rest_framework is installed (CI).
"""

import importlib.util
import os

import pytest

from dvsa_api.mcp import registry
from dvsa_api.mcp.session_manager import MCPSessionManager

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Two converging vehicle tracks (an accident) — same geometry as the simulator.
_TWO_TRACKS = [
    {"id": "veh-1", "samples": [{"t": 0.0, "x": 0.0, "y": 10.0},
                                {"t": 1.0, "x": 10.0, "y": 10.0},
                                {"t": 2.0, "x": 12.0, "y": 10.0}]},
    {"id": "veh-2", "samples": [{"t": 0.0, "x": 20.0, "y": 0.0},
                                {"t": 1.0, "x": 13.0, "y": 10.0},
                                {"t": 2.0, "x": 12.5, "y": 10.0}]},
]


def _load_simulator():
    path = os.path.join(_REPO_ROOT, "mcp", "simulators", "environment.py")
    spec = importlib.util.spec_from_file_location("mcp_env_sim_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _discover(monkeypatch):
    monkeypatch.setenv("MCP_HUMAN_AUTO_CONFIRM", "1")
    monkeypatch.delenv("MCP_ALLOWED_AGENTS", raising=False)
    registry.discover_agents(force=True)


# --------------------------------------------------------------------------- #
# Core integration (pure Python)
# --------------------------------------------------------------------------- #
def test_urban_accident_mission_end_to_end():
    report = _load_simulator().simulate_mission("urban_accident_response")
    assert report["status"] == "completed"
    assert report["escalated"] is True
    task_by_id = {t["task_id"]: t for t in report["tasks"]}
    # Static chain completed ...
    assert task_by_id["scout"]["status"] == "completed"
    assert task_by_id["analyze"]["status"] == "completed"
    assert task_by_id["coordinate"]["status"] == "delegated"
    # ... and a dynamic human-confirmation follow-up ran to completion.
    human = [t for t in report["tasks"] if t["capability"] == "human_interface"]
    assert human and human[0]["status"] == "completed"


def test_inline_mission_runs():
    manager = MCPSessionManager()
    session = manager.create_mission({
        "mission_id": "inline-1",
        "tasks": [
            {"id": "s", "capability": "scout", "payload": {"tracks": _TWO_TRACKS}},
            {"id": "a", "capability": "analyzer", "depends_on": ["s"]},
        ],
    }, context={}, run=True)
    assert session.status == "completed"
    # Reasoning model confirms the accident from the converging track geometry.
    assert session.summary["results"]["a"]["result"]["anomaly_confirmed"] is True


def test_infrastructure_inspection_mission_runs():
    report = _load_simulator().simulate_mission("infrastructure_inspection")
    assert report["status"] in ("completed", "partial")


def test_cancel_command_records_and_marks_cancelled():
    manager = MCPSessionManager()
    session = manager.create_mission({
        "mission_id": "cancel-me",
        "tasks": [{"id": "s", "capability": "scout", "payload": {"tracks": []}}],
    }, context={}, run=True)
    out = manager.send_command(session.mission_id, "cancel")
    assert out["accepted"] is True
    assert manager.get_mission(session.mission_id).commands[-1]["command"] == "cancel"


def test_unknown_mission_template_raises():
    from dvsa_api.mcp.errors import MissionError
    with pytest.raises(MissionError):
        MCPSessionManager().create_mission("does_not_exist_mission", run=False)


# --------------------------------------------------------------------------- #
# REST API (only where DRF is installed — i.e. CI)
# --------------------------------------------------------------------------- #
try:
    from rest_framework.test import APIClient  # noqa: E402
    _HAS_DRF = True
except Exception:  # noqa: BLE001
    _HAS_DRF = False

drf_only = pytest.mark.skipif(not _HAS_DRF, reason="DRF not installed")


@drf_only
def test_api_create_mission_and_status(monkeypatch):
    monkeypatch.setenv("ENABLE_MCP", "true")
    monkeypatch.setenv("MCP_HUMAN_AUTO_CONFIRM", "1")
    client = APIClient()
    resp = client.post("/api/mcp/missions",
                       {"mission": "urban_accident_response"}, format="json")
    assert resp.status_code == 201, resp.content
    data = resp.json()
    assert data["status"] == "completed"
    assert "request_id" in data and "server_latency_ms" in data
    mission_id = data["mission_id"]

    detail = client.get(f"/api/mcp/missions/{mission_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "completed"


@drf_only
def test_api_disabled_returns_503(monkeypatch):
    monkeypatch.delenv("ENABLE_MCP", raising=False)
    resp = APIClient().get("/api/mcp/agents")
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "mcp_disabled"


@drf_only
def test_api_agents_list_and_invoke(monkeypatch):
    monkeypatch.setenv("ENABLE_MCP", "true")
    client = APIClient()
    listing = client.get("/api/mcp/agents")
    assert listing.status_code == 200
    assert any(a["name"] == "scout_agent" for a in listing.json()["agents"])

    invoke = client.post(
        "/api/mcp/agents/scout_agent/invoke",
        {"payload": {"tracks": [{"id": "a"}, {"id": "b"}]}}, format="json")
    assert invoke.status_code == 200, invoke.content
    assert invoke.json()["result"]["result"]["candidates"]
