#!/usr/bin/env python
"""Exercise the DVSA MCP server end-to-end over stdio (no external MCP SDK).

Spawns ``scripts/mcp_server_run.py`` as a subprocess, performs the MCP handshake,
lists tools/resources, calls a couple of tools, and reads a resource — printing
each result. Fully offline; useful as a smoke test and a copy-paste client example.

Usage::

    python scripts/mcp_client_test.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StdioMCPClient:
    """Minimal JSON-RPC-over-stdio MCP client using only the standard library."""

    def __init__(self, server_cmd=None) -> None:
        self._cmd = server_cmd or [sys.executable,
                                   os.path.join(_REPO_ROOT, "scripts", "mcp_server_run.py")]
        self._proc: Optional[subprocess.Popen] = None
        self._id = 0

    def __enter__(self) -> "StdioMCPClient":
        self._proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        return self

    def __exit__(self, *exc) -> None:
        if self._proc:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method,
                    "params": params or {}})
        return self._read()

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _send(self, obj: Dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _read(self) -> Dict[str, Any]:
        assert self._proc and self._proc.stdout
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed the connection")
        return json.loads(line)


def main() -> int:
    with StdioMCPClient() as client:
        init = client.request("initialize", {"protocolVersion": "2024-11-05"})
        print("initialize ->", json.dumps(init["result"]["serverInfo"]))
        client.notify("notifications/initialized")

        tools = client.request("tools/list")["result"]["tools"]
        print("tools ->", [t["name"] for t in tools])

        resources = client.request("resources/list")["result"]["resources"]
        print("resources ->", [r["uri"] for r in resources])

        tracks = client.request("tools/call", {
            "name": "dvsa.get_tracks", "arguments": {"id": "demo"}})
        detect = client.request("tools/call", {
            "name": "dvsa.detect_anomalies",
            "arguments": {"tracks": tracks["result"]["structuredContent"]["tracks"]}})
        sc = detect["result"]["structuredContent"]
        print("detect_anomalies -> anomaly_detected =", sc["anomaly_detected"],
              "| label =", (sc["actions"] or [{}])[0].get("label"))

        sensor = client.request("resources/read", {"uri": "dvsa://sensor/demo"})
        print("resources/read sensor ->",
              json.loads(sensor["result"]["contents"][0]["text"])["platform"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
