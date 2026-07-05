# `_template_agent` — MCP agent template

Copy this folder to `mcp/agents/<your_agent_name>/` and edit the files to author a
new agent. With `ENABLE_MCP=true`, the agent is **auto-discovered** and usable in
missions and via `POST /api/mcp/agents/<name>/invoke` — **no other code changes**.

## Files

| File | Purpose |
| --- | --- |
| `manifest.json` | `name`, `version`, `type`, `entrypoint`, `capabilities`, `priority`, `resources`, optional `config`. |
| `agent.py` | Exports a class named `AgentAdapter` (`handle_task` + `health_check`, optional `on_start`/`on_stop`). |
| `Dockerfile` / `docker-compose.yml` | Optional — only for `type: "container"` agents (HTTP shim). |
| `README.md` | Dependencies / run notes. |

## Contract

```python
def handle_task(self, task: dict, context: dict) -> dict:
    return {
        "status": "completed",              # completed|failed|in_progress|delegated
        "result": {"...": "..."},           # actions / detections / reasoning_trace
        "metadata": {"agent": "my_agent", "version": "0.1.0", "latency_ms": 3},
    }

def health_check(self) -> dict:
    return {"status": "ok"}
```

`task` = `{task_id, type, payload, deadline_ms, priority}`.
`context` = `{mission_id, session_meta, sensor_meta, upstream}` where `upstream`
maps each dependency `task_id` to its completed result.

## Agent types

- `inprocess` — imported and run inside the server process (default; fastest).
- `container` — packaged as a Docker image exposing `GET /health` + `POST /handle_task`.
- `remote` — an external HTTP service; set `"endpoint"` (and optional
  `"auth": {"api_key_env": "MY_KEY"}`) in the manifest.

## Dynamic follow-up tasks

Return `result.followups = [{ "id", "type", "capability"|"agent", "payload", "depends_on" }]`
to have the executor schedule additional tasks at run time.
