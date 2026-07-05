"""JSON-RPC / MCP protocol compliance tests for the DVSA MCP server.

Pure-Python: no Django, no network. Drives :class:`MCPServer.handle` directly.
"""

import pytest

from dvsa_api.mcp.server import PROTOCOL_VERSION, build_default_server


@pytest.fixture()
def server():
    return build_default_server()


def _req(method, params=None, rid=1):
    req = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if rid is not None:
        req["id"] = rid
    return req


# --------------------------------------------------------------------------- #
# Envelope + handshake
# --------------------------------------------------------------------------- #
def test_initialize_returns_capabilities_and_serverinfo(server):
    resp = server.handle(_req("initialize"))
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert "tools" in result["capabilities"]
    assert "resources" in result["capabilities"]
    assert result["serverInfo"]["name"] == "dvsa-mcp-server"


def test_response_echoes_request_id(server):
    resp = server.handle(_req("ping", rid=42))
    assert resp["id"] == 42
    assert resp["result"] == {}


def test_notification_returns_no_response(server):
    # No "id" -> a notification -> no response, even for a valid method.
    assert server.handle(_req("notifications/initialized", rid=None)) is None


def test_notification_unknown_method_still_silent(server):
    assert server.handle(_req("does/not/exist", rid=None)) is None


# --------------------------------------------------------------------------- #
# Error handling (JSON-RPC codes)
# --------------------------------------------------------------------------- #
def test_unknown_method_returns_method_not_found(server):
    resp = server.handle(_req("bogus/method"))
    assert resp["error"]["code"] == -32601


def test_bad_jsonrpc_version_is_invalid_request(server):
    resp = server.handle({"jsonrpc": "1.0", "id": 1, "method": "ping"})
    assert resp["error"]["code"] == -32600


def test_missing_method_is_invalid_request(server):
    resp = server.handle({"jsonrpc": "2.0", "id": 1})
    assert resp["error"]["code"] == -32600


# --------------------------------------------------------------------------- #
# Listings
# --------------------------------------------------------------------------- #
def test_tools_list_shape(server):
    tools = server.handle(_req("tools/list"))["result"]["tools"]
    names = {t["name"] for t in tools}
    assert "dvsa.detect_anomalies" in names
    for tool in tools:
        assert "description" in tool and "inputSchema" in tool


def test_resources_list_and_templates(server):
    resources = server.handle(_req("resources/list"))["result"]["resources"]
    assert any(r["uri"] == "dvsa://tracks/demo" for r in resources)
    templates = server.handle(_req("resources/templates/list"))["result"]
    assert templates["resourceTemplates"]


def test_batch_drops_notifications(server):
    batch = [_req("ping", rid=1), _req("notifications/initialized", rid=None),
             _req("ping", rid=2)]
    out = server.handle_batch(batch)
    assert [r["id"] for r in out] == [1, 2]
