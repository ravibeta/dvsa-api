# DVSA MCP — Python client example

Two ways to talk to the DVSA Model-Context-Protocol server.

## 1. Zero-dependency (stdlib only)

```bash
python mcp/clients/python_example/client.py
```

`client.py` launches `scripts/mcp_server_run.py` over stdio and drives it with
plain JSON-RPC — no extra packages required (handy for CI).

## 2. Official MCP SDK

Install the Anthropic MCP client SDK and connect over stdio:

```bash
pip install mcp
```

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command="python", args=["scripts/mcp_server_run.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print([t.name for t in tools.tools])
            result = await session.call_tool(
                "dvsa.detect_anomalies",
                {"tracks": [{"id": "a"}, {"id": "b"}]})
            print(result.structuredContent)

asyncio.run(main())
```

Because the DVSA server implements standard MCP (JSON-RPC 2.0, `initialize`,
`tools/list`, `tools/call`, `resources/list`, `resources/read`), **any** MCP
client works without changes.

## Available tools

`dvsa.detect_anomalies` · `dvsa.run_reasoning` · `dvsa.extract_features` ·
`dvsa.get_tracks` · `dvsa.get_frames`

## Available resources

`dvsa://frames/<id>` · `dvsa://tracks/<id>` · `dvsa://sensor/<id>` (try `demo`).
