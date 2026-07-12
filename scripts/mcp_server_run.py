#!/usr/bin/env python
"""Run the DVSA Model-Context-Protocol server over the stdio transport.

Reads newline-delimited JSON-RPC messages from stdin and writes JSON-RPC
responses to stdout (the transport Claude Desktop / the Claude CLI use to launch
a local MCP server). Runs fully offline.

Register with an MCP client by pointing its ``command`` at::

    python /abs/path/to/scripts/mcp_server_run.py

Manual smoke test::

    printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize"}' \
        | python scripts/mcp_server_run.py
"""

from __future__ import annotations

import json
import os
import sys

# Allow running as a standalone script from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dvsa_api.mcp.protocol_errors import PARSE_ERROR  # noqa: E402
from dvsa_api.mcp.server import build_default_server  # noqa: E402


def _write(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def serve(stdin=None, stdout=None) -> int:
    """Newline-delimited JSON-RPC stdio loop over the default DVSA server."""
    stdin = stdin or sys.stdin
    server = build_default_server()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            _write({"jsonrpc": "2.0", "id": None,
                    "error": {"code": PARSE_ERROR, "message": "parse error"}})
            continue
        if isinstance(request, list):  # JSON-RPC batch
            responses = server.handle_batch(request)
            for resp in responses:
                _write(resp)
            continue
        response = server.handle(request)
        if response is not None:
            _write(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
