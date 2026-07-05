# Using the DVSA MCP server from a client

The DVSA MCP server is a standard stdio Model-Context-Protocol server, so it
works with Claude Desktop, the Claude CLI, the official `mcp` Python SDK, or the
zero-dependency example client — all offline, no API keys.

## Quick check

```bash
python scripts/mcp_client_test.py
# initialize -> {"name": "dvsa-mcp-server", "version": "1.0.0"}
# tools -> ['dvsa.detect_anomalies', 'dvsa.run_reasoning', ...]
# detect_anomalies -> anomaly_detected = True | label = accident
```

## Claude Desktop

Merge this into your `claude_desktop_config.json` (see
`mcp/clients/claude_example/`), replacing the path with your checkout:

```json
{
  "mcpServers": {
    "dvsa": {
      "command": "python",
      "args": ["/ABSOLUTE/PATH/TO/dvsa-api/scripts/mcp_server_run.py"]
    }
  }
}
```

Config location — macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`;
Windows: `%APPDATA%\Claude\claude_desktop_config.json`. Restart Claude Desktop and
the **dvsa** tools appear. Try: *"Read dvsa://tracks/demo and detect anomalies."*

## Claude CLI

```bash
claude mcp add dvsa -- python /ABSOLUTE/PATH/TO/dvsa-api/scripts/mcp_server_run.py
claude mcp list
```

## Official MCP Python SDK

```bash
pip install mcp
```

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["scripts/mcp_server_run.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print([t.name for t in (await session.list_tools()).tools])
            res = await session.call_tool("dvsa.get_tracks", {"id": "demo"})
            print(res.structuredContent)

asyncio.run(main())
```

## Zero-dependency Python client

```bash
python mcp/clients/python_example/client.py
```

Uses only the standard library (launches the server over stdio and speaks
JSON-RPC directly) — see `mcp/clients/python_example/README.md`.

## Typical calls

```jsonc
// list what's available
{"jsonrpc":"2.0","id":1,"method":"tools/list"}
{"jsonrpc":"2.0","id":2,"method":"resources/list"}

// read a dataset resource
{"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":"dvsa://tracks/demo"}}

// call a tool
{"jsonrpc":"2.0","id":4,"method":"tools/call",
 "params":{"name":"dvsa.detect_anomalies","arguments":{"tracks":[{"id":"a"},{"id":"b"}]}}}
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Client can't start the server | Use an absolute path to `scripts/mcp_server_run.py`; ensure `python` is on PATH. |
| `-32601 method not found` | Check the method name (`tools/call`, not `tool/call`). |
| `-32602` on `tools/call` | Unknown tool name or a missing required argument. |
| `-32602` on `resources/read` | Unknown `dvsa://` URI; list with `resources/list`. |
| Want live data | Set `MCP_DVSA_API_BASE` to your DVSA-API base URL. |
