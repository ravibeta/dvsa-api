"""Unit tests for the example MCP agents' ``handle_task`` logic.

Loads each agent via the registry (folder discovery) and drives it directly. No
Django / network; the analyzer's reasoning call resolves to a deterministic
offline synthetic result.
"""

import pytest

from dvsa_api.mcp import registry

_TWO_TRACKS = [
    {"id": "veh-1", "samples": [{"t": 0.0, "x": 0.0, "y": 10.0},
                                {"t": 1.0, "x": 10.0, "y": 10.0},
                                {"t": 2.0, "x": 12.0, "y": 10.0}]},
    {"id": "veh-2", "samples": [{"t": 0.0, "x": 20.0, "y": 0.0},
                                {"t": 1.0, "x": 13.0, "y": 10.0},
                                {"t": 2.0, "x": 12.5, "y": 10.0}]},
]


@pytest.fixture(autouse=True)
def _discover():
    registry.discover_agents(force=True)


def _agent(name):
    return registry.get_agent(name)


def test_scout_flags_candidate_with_two_tracks():
    out = _agent("scout_agent").handle_task(
        {"task_id": "t", "payload": {"tracks": _TWO_TRACKS}}, {})
    assert out["status"] == "completed"
    assert out["result"]["candidates"]
    assert out["result"]["observed_tracks"] == 2


def test_scout_no_candidate_with_single_track():
    out = _agent("scout_agent").handle_task(
        {"task_id": "t", "payload": {"tracks": _TWO_TRACKS[:1]}}, {})
    assert out["result"]["candidates"] == []


def test_analyzer_confirms_accident_via_reasoning():
    context = {"upstream": {"scout": {"result": {"tracks": _TWO_TRACKS}}}}
    out = _agent("analyzer_agent").handle_task({"task_id": "t", "payload": {}}, context)
    assert out["status"] == "completed"
    assert out["result"]["anomaly_confirmed"] is True
    assert out["result"]["confidence"] >= 0.9
    assert out["result"]["reasoning_trace"]


def test_coordinator_escalates_high_confidence_anomaly():
    context = {
        "escalation_rules": {"require_human": True, "min_confidence": 0.8},
        "upstream": {"analyze": {"result": {
            "anomaly_confirmed": True, "confidence": 0.95,
            "actions": [{"label": "accident"}]}}},
    }
    out = _agent("coordinator_agent").handle_task({"task_id": "t"}, context)
    assert out["status"] == "delegated"
    assert out["result"]["escalated"] is True
    followups = out["result"]["followups"]
    assert followups and followups[0]["capability"] == "human_interface"


def test_coordinator_no_escalation_below_threshold():
    context = {
        "escalation_rules": {"require_human": True, "min_confidence": 0.8},
        "upstream": {"analyze": {"result": {
            "anomaly_confirmed": True, "confidence": 0.4}}},
    }
    out = _agent("coordinator_agent").handle_task({"task_id": "t"}, context)
    assert out["result"]["escalated"] is False
    assert out["result"]["followups"] == []


def test_human_interface_auto_confirms(monkeypatch):
    monkeypatch.setenv("MCP_HUMAN_AUTO_CONFIRM", "1")
    out = _agent("human_interface_agent").handle_task(
        {"task_id": "t", "payload": {"summary": "confirm?"}}, {})
    assert out["status"] == "completed"
    assert out["result"]["confirmed"] is True


def test_template_agent_echoes():
    out = _agent("_template_agent").handle_task(
        {"task_id": "t", "payload": {"x": 1}}, {})
    assert out["status"] == "completed"
    assert out["result"]["echo"] == {"x": 1}


def test_all_agents_report_healthy():
    for name in ("scout_agent", "analyzer_agent", "coordinator_agent",
                 "human_interface_agent", "_template_agent"):
        assert _agent(name).health_check()["status"] == "ok"
