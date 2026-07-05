# Authoring MCP agents (`mcp/agents/`)

Drop a folder here to add an agent. With `ENABLE_MCP=true` the agent is
auto-discovered at startup (or first use) — **no other code changes**.

## Step by step

1. **Copy the template**

   ```bash
   cp -r mcp/agents/_template_agent mcp/agents/my_agent
   ```

2. **Edit `manifest.json`** — set a unique `name`, the `capabilities` the agent
   offers, and its `type` (`inprocess` / `container` / `remote`):

   ```json
   {
     "name": "my_agent",
     "version": "1.0.0",
     "type": "inprocess",
     "entrypoint": "agent.py",
     "capabilities": ["scout"],
     "priority": 3,
     "latency_hint_ms": 20,
     "resources": {"cpu": "0.25", "memory": "128Mi"},
     "fallback_agents": []
   }
   ```

3. **Implement `agent.py`** — export a class named `AgentAdapter`:

   ```python
   class AgentAdapter:
       name = "my_agent"
       version = "1.0.0"
       capabilities = ["scout"]

       def handle_task(self, task: dict, context: dict) -> dict:
           return {"status": "completed", "result": {...},
                   "metadata": {"agent": "my_agent", "version": "1.0.0", "latency_ms": 1}}

       def health_check(self) -> dict:
           return {"status": "ok"}
   ```

4. **Reference it in a mission** by capability (see `mcp/missions/`) or invoke it
   directly: `POST /api/mcp/agents/my_agent/invoke`.

## Contract summary

- `handle_task(task, context) -> {status, result, metadata}` where `status` is
  one of `completed` / `failed` / `in_progress` / `delegated`.
- `context.upstream[<dep_task_id>]` holds each completed dependency's result.
- Emit `result.followups = [ {id, type, capability|agent, payload, depends_on} ]`
  to schedule dynamic follow-up tasks (e.g. escalate to `human_interface`).

## Example agents in this folder

| Agent | Capability | Role |
| --- | --- | --- |
| `scout_agent` | `scout` | Fast candidate-anomaly detection from tracks. |
| `analyzer_agent` | `analyzer` | Deeper analysis via `dvsa_api.reasoning.call_model`. |
| `coordinator_agent` | `coordinator` | Aggregates results, triggers human escalation. |
| `human_interface_agent` | `human_interface` | Human-in-the-loop confirmation (mock/webhook). |

## Security

- Restrict which agents may load with `MCP_ALLOWED_AGENTS` / `MCP_AGENT_WHITELIST`
  (comma-separated names).
- `inprocess` agents run in the server process — treat their code as trusted.
  Run untrusted agents as `container`/`remote` with resource limits.
