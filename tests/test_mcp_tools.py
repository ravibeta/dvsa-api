"""Tool-invocation tests for the DVSA MCP server (with mocked DVSA calls).

Pure-Python: no Django, no network. Exercises ``tools/call`` for every DVSA tool,
including a mocked reasoning path to show tools delegate to DVSA internals.
"""

import json

import pytest

from dvsa_api.mcp import tool_registry
from dvsa_api.mcp.server import build_default_server

_TWO_TRACKS = [
    {"id": "veh-1", "samples": [{"t": 0.0, "x": 0.0, "y": 10.0},
                                {"t": 1.0, "x": 10.0, "y": 10.0},
                                {"t": 2.0, "x": 12.0, "y": 10.0}]},
    {"id": "veh-2", "samples": [{"t": 0.0, "x": 20.0, "y": 0.0},
                                {"t": 1.0, "x": 13.0, "y": 10.0},
                                {"t": 2.0, "x": 12.5, "y": 10.0}]},
]


@pytest.fixture()
def server():
    return build_default_server()


def _call(server, name, arguments):
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})
    return resp


def test_detect_anomalies_flags_accident(server):
    resp = _call(server, "dvsa.detect_anomalies", {"tracks": _TWO_TRACKS})
    result = resp["result"]
    assert result["isError"] is False
    sc = result["structuredContent"]
    assert sc["anomaly_detected"] is True
    assert sc["actions"][0]["label"] == "accident"
    # Content mirrors the structured result as JSON text.
    assert json.loads(result["content"][0]["text"])["anomaly_detected"] is True


def test_detect_anomalies_uses_mocked_reasoning(server, monkeypatch):
    # Prove the tool delegates to the DVSA reasoning layer.
    calls = {}

    def fake_call_model(model, context):
        calls["model"] = model
        return {"actions": [{"type": "anomaly", "label": "accident", "confidence": 1.0}],
                "reasoning_trace": ["mocked"], "metadata": {"model": model}}

    import dvsa_api.reasoning as reasoning
    monkeypatch.setattr(reasoning, "call_model", fake_call_model)
    resp = _call(server, "dvsa.detect_anomalies",
                 {"tracks": _TWO_TRACKS, "model": "my_model"})
    assert calls["model"] == "my_model"
    assert resp["result"]["structuredContent"]["reasoning_trace"] == ["mocked"]


def test_run_reasoning_policy_selection(server):
    resp = _call(server, "dvsa.run_reasoning",
                 {"tracks": _TWO_TRACKS, "policy": "latency_optimized"})
    sc = resp["result"]["structuredContent"]
    assert sc["model"]  # a model was selected
    assert "reasoning_trace" in sc


def test_extract_features_summary(server):
    resp = _call(server, "dvsa.extract_features", {"tracks": _TWO_TRACKS,
                                                   "frames": [{"frame_id": 0}]})
    sc = resp["result"]["structuredContent"]
    assert sc["num_tracks"] == 2
    assert sc["num_frames"] == 1
    assert sc["has_multiple_tracks"] is True


def test_get_tracks_and_frames_resource_backed(server):
    tracks = _call(server, "dvsa.get_tracks", {"id": "demo"})
    assert tracks["result"]["structuredContent"]["tracks"][0]["id"] == "veh-1"
    frames = _call(server, "dvsa.get_frames", {"id": "demo"})
    assert frames["result"]["structuredContent"]["frames"][0]["frame_id"] == 0


def test_unknown_tool_is_invalid_params(server):
    resp = _call(server, "dvsa.nope", {})
    assert resp["error"]["code"] == -32602


def test_missing_required_argument_is_invalid_params(server):
    resp = _call(server, "dvsa.detect_anomalies", {})  # 'tracks' required
    assert resp["error"]["code"] == -32602


def test_registry_direct_call():
    registry = tool_registry.build_default_registry()
    out = registry.call("dvsa.extract_features", {"tracks": _TWO_TRACKS})
    assert out["num_tracks"] == 2
