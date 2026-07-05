# Connect Claude Desktop / Claude CLI to the DVSA MCP server

The DVSA MCP server is a standard stdio MCP server, so Claude Desktop and the
Claude CLI can launch it directly.

## Claude Desktop

1. Ensure Python can import the repo (run once from the repo root to verify):

   ```bash
   python scripts/mcp_server_run.py <<< '{"jsonrpc":"2.0","id":1,"method":"initialize"}'
   ```

2. Open **Claude Desktop → Settings → Developer → Edit Config**, and merge the
   snippet from [`claude_desktop_config.json`](claude_desktop_config.json),
   replacing `/ABSOLUTE/PATH/TO/dvsa-api` with your checkout path:

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

   The config lives at:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

3. Restart Claude Desktop. The **dvsa** server appears with a tools icon; you can
   now ask Claude to *"detect anomalies in dvsa://tracks/demo"* and it will call
   `dvsa.get_tracks` + `dvsa.detect_anomalies` natively.

## Claude CLI

```bash
claude mcp add dvsa -- python /ABSOLUTE/PATH/TO/dvsa-api/scripts/mcp_server_run.py
claude mcp list          # shows "dvsa"
```

## Tools & resources exposed

| Tool | Purpose |
| --- | --- |
| `dvsa.detect_anomalies` | Accident/anomaly detection from tracks/frames. |
| `dvsa.run_reasoning` | Run a DVSA reasoning model (named or policy-selected). |
| `dvsa.extract_features` | Deterministic frame/track feature summary. |
| `dvsa.get_tracks` / `dvsa.get_frames` | Fetch a dataset by id. |

Resources: `dvsa://frames/<id>`, `dvsa://tracks/<id>`, `dvsa://sensor/<id>`
(bundled sample id: `demo`).

## Notes

- Everything runs **locally and offline** by default; no API keys required.
- To back resources with a live DVSA-API instead of bundled files, set
  `MCP_DVSA_API_BASE=https://your-dvsa-host/api/v1`.
