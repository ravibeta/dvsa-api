"""Resource retrieval tests for the DVSA MCP server.

Pure-Python: no Django, no network. Reads the bundled sample datasets via the
resource adapter and the ``resources/read`` JSON-RPC method.
"""

import json

import pytest

from dvsa_api.mcp.protocol_errors import ResourceNotFoundError
from dvsa_api.mcp.resource_adapter import ResourceAdapter, parse_uri
from dvsa_api.mcp.server import build_default_server


@pytest.fixture()
def server():
    return build_default_server()


# --------------------------------------------------------------------------- #
# URI parsing
# --------------------------------------------------------------------------- #
def test_parse_uri_valid():
    assert parse_uri("dvsa://tracks/demo") == {"kind": "tracks", "id": "demo"}


@pytest.mark.parametrize("uri", ["http://x/y", "dvsa://tracks", "dvsa://bogus/demo"])
def test_parse_uri_invalid_raises(uri):
    with pytest.raises(ResourceNotFoundError):
        parse_uri(uri)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
def test_list_resources_includes_demo():
    uris = {r.uri for r in ResourceAdapter().list_resources()}
    assert {"dvsa://frames/demo", "dvsa://tracks/demo", "dvsa://sensor/demo"} <= uris


def test_read_tracks_frames_sensor():
    adapter = ResourceAdapter()
    assert adapter.read("dvsa://tracks/demo")[0]["id"] == "veh-1"
    assert adapter.read("dvsa://frames/demo")[0]["frame_id"] == 0
    assert adapter.read("dvsa://sensor/demo")["platform"] == "quadrotor"


def test_read_unknown_id_raises():
    with pytest.raises(ResourceNotFoundError):
        ResourceAdapter().read("dvsa://tracks/nonexistent")


# --------------------------------------------------------------------------- #
# JSON-RPC resources/read
# --------------------------------------------------------------------------- #
def test_resources_read_over_jsonrpc(server):
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "resources/read",
                          "params": {"uri": "dvsa://sensor/demo"}})
    contents = resp["result"]["contents"]
    assert contents[0]["uri"] == "dvsa://sensor/demo"
    assert contents[0]["mimeType"] == "application/json"
    assert json.loads(contents[0]["text"])["session_id"] == "demo"


def test_resources_read_unknown_uri_is_error(server):
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "resources/read",
                          "params": {"uri": "dvsa://tracks/ghost"}})
    assert resp["error"]["code"] == -32602


def test_resources_read_requires_uri(server):
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "resources/read",
                          "params": {}})
    assert resp["error"]["code"] == -32600
