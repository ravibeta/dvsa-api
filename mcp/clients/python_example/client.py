#!/usr/bin/env python
"""Example Python MCP client for the DVSA MCP server.

Two ways to connect:

1. **Zero-dependency (this file)** — launch the server over stdio using only the
   standard library. Good for CI and quick checks.
2. **Official MCP SDK** — install ``mcp`` (``pip install mcp``) and use
   ``mcp.client.stdio``; see ``README.md`` for the snippet. The DVSA server speaks
   standard MCP, so any compliant client works unchanged.

Run:

    python mcp/clients/python_example/client.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_SERVER = os.path.join(_REPO_ROOT, "scripts", "mcp_server_run.py")


def _rpc(proc, _id, method, params=None):
    proc.stdin.write(json.dumps(
        {"jsonrpc": "2.0", "id": _id, "method": method, "params": params or {}}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, _SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1)
    try:
        print(_rpc(proc, 1, "initialize")["result"]["serverInfo"])
        tools = _rpc(proc, 2, "tools/list")["result"]["tools"]
        print("tools:", [t["name"] for t in tools])
        result = _rpc(proc, 3, "tools/call", {
            "name": "dvsa.run_reasoning",
            "arguments": {"tracks": [{"id": "a", "samples": [{"t": 0, "x": 0, "y": 0}]},
                                     {"id": "b", "samples": [{"t": 0, "x": 1, "y": 0}]}],
                          "query": "Any accidents?"}})
        print("run_reasoning:", json.dumps(result["result"]["structuredContent"]))
    finally:
        proc.stdin.close()
        proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
