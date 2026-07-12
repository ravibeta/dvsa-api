# DVSA as a Model-Context-Protocol (MCP) server

DVSA-API ships a protocol-compliant **Model-Context-Protocol** server that
exposes DVSA drone-video analytics as MCP **tools** and DVSA datasets as MCP
**resources**. Any MCP client — Claude Desktop, the Claude CLI, or a custom
client — can call DVSA natively over JSON-RPC 2.0, with **zero changes to the
DVSA core**. Tool handlers reuse the DVSA reasoning adapters and data layers
internally.

> This is the Anthropic-MCP protocol server. It is **independent of** the DVSA
> multi-agent *control plane* (`mcp/agents`, missions, planner/executor), which
> remains available; the two subsystems coexist in `dvsa_api/mcp/`. See
> [`docs/mcp_integration.md`](mcp_integration.md) for the control plane.

## Architecture

```
   MCP client (Claude Desktop / CLI / SDK)
              │  JSON-RPC 2.0 over stdio
     ┌────────▼─────────┐
     │    MCPServer     │  initialize / tools/list / tools/call
     │  (server.py)     │  resources/list / resources/read / ping
     └───┬─────────┬────┘
         │         │
 ┌───────▼───┐ ┌───▼────────────┐
 │ToolRegistry│ │ ResourceAdapter│  dvsa://frames|tracks|sensor/<id>
 └─────┬──────┘ └───────┬────────┘
       │ reuse          │ load
 ┌─────▼───────────┐    ▼
 │ dvsa_api.reasoning │  local files / URL / DVSA-API endpoint
 │ (call_model, ...) │
 └───────────────────┘
```

## Run the server

```bash
# stdio transport (what MCP clients launch):
python scripts/mcp_server_run.py

# end-to-end smoke test (spawns the server, drives it, prints results):
python scripts/mcp_client_test.py
```

Everything runs **offline** with bundled sample data; no API keys required.

## Tools

| Tool | Input (key args) | Returns |
| --- | --- | --- |
| `dvsa.detect_anomalies` | `tracks[]`, `frames[]?`, `query?`, `model?` | `actions`, `anomalies`, `reasoning_trace`, `anomaly_detected` |
| `dvsa.run_reasoning` | `model?`/`policy?`, `context`/`tracks`/`frames`/`query` | `model`, `actions`, `reasoning_trace`, `metadata` |
| `dvsa.extract_features` | `frames[]?`, `tracks[]?` | `num_frames`, `num_tracks`, `track_features` |
| `dvsa.get_tracks` | `id` | `{uri, tracks}` |
| `dvsa.get_frames` | `id` | `{uri, frames}` |

Each tool advertises `name`, `description`, `inputSchema`, and `outputSchema`
(via `tools/list`). `dvsa.detect_anomalies` / `dvsa.run_reasoning` delegate to the
pluggable DVSA reasoning models (e.g. `urban_accident_example`).

## Resources

| URI | Content |
| --- | --- |
| `dvsa://frames/<id>` | Frame metadata records. |
| `dvsa://tracks/<id>` | Object tracks (`{t, x, y[, vx, vy]}`). |
| `dvsa://sensor/<id>` | Sensor/telemetry metadata (gps, altitude, time). |

Resolved (in order) from the local data root (`mcp/resource_data/<kind>/<id>.json`),
a DVSA-API endpoint (`MCP_DVSA_API_BASE`), or an `http(s)` id. Bundled sample id:
`demo`.

## JSON-RPC surface

`initialize` · `notifications/initialized` · `ping` · `tools/list` · `tools/call`
· `resources/list` · `resources/templates/list` · `resources/read`. Standard
JSON-RPC error codes: `-32700` parse, `-32600` invalid request, `-32601` method
not found, `-32602` invalid params, `-32603` internal.

```jsonc
// tools/call request
{"jsonrpc":"2.0","id":1,"method":"tools/call",
 "params":{"name":"dvsa.detect_anomalies","arguments":{"tracks":[...]}}}
// result: {"content":[{"type":"text","text":"..."}],"structuredContent":{...},"isError":false}
```

## Configuration

`dvsa_api/mcp/config.json` describes the server (name/version, transport, tools,
resources, auth). Environment variables:

| Var | Default | Purpose |
| --- | --- | --- |
| `MCP_RESOURCE_DATA_ROOT` | `mcp/resource_data` | Local resource data root. |
| `MCP_DVSA_API_BASE` | *(unset)* | Back resources with a live DVSA-API. |

## Connecting a client

See [`docs/mcp_client_usage.md`](mcp_client_usage.md) for Claude Desktop / CLI
setup and the Python client examples under `mcp/clients/`.
