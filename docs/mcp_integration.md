# MCP — Multi-Agent Control Plane

The MCP subsystem lets DVSA run **coordinated multi-agent workflows** (missions)
for aerial video sensing — anomaly triage, incident response, persistent
surveillance with agent handoffs, and human-in-the-loop escalation. It is
**additive and opt-in**: enable it with `ENABLE_MCP=true`, drop agent folders
under `mcp/agents/`, and start the server. The default in-memory message bus
needs no external services, so it is safe to run in development.

## Architecture

```
                         POST /api/mcp/missions
                                  │
                          ┌───────▼────────┐
                          │ SessionManager │  mission lifecycle + telemetry
                          └───────┬────────┘
                                  │ plan_mission(template)
                          ┌───────▼────────┐
                          │    Planner     │  template -> task DAG (validated)
                          └───────┬────────┘
                                  │ TaskGraph
                          ┌───────▼────────┐     select_agent(capability, policy)
                          │    Executor    │────────────────► Registry ──► mcp/agents/*
                          │ timeouts/retry │                   (discovery + manifests)
                          │ concurrency    │◄───── handle_task(task, context)
                          └───┬────────┬───┘
                events │      │        │  metrics (counters, latency, health)
             ┌─────────▼──┐   │   ┌────▼───────┐
             │ MessageBus │   │   │  Metrics   │──► pluggable sink (console default)
             │ (in-memory)│   │   └────────────┘
             └────────────┘   ▼
                     scout ─► analyzer ─► coordinator ─► human_interface
                                (reasoning)     (escalation / delegated)
```

Runtime modules live in `dvsa_api/mcp/`; agents and missions are data under the
top-level `mcp/` folder.

## Quick start

```bash
export ENABLE_MCP=true

# 1. See the discovered agents.
python scripts/mcp_cli.py discover

# 2. Run the example mission end-to-end in the simulator (offline).
python scripts/mcp_cli.py simulate urban_accident_response

# 3. Or drive it over REST (Django dev server).
python manage.py runserver &
curl -X POST http://localhost:8000/api/mcp/missions \
     -H 'Content-Type: application/json' \
     -d '{"mission": "urban_accident_response"}'
```

`make mcp-start`, `make mcp-simulate`, `make mcp-test`, and `make mcp-lint` wrap
these for convenience.

## Adding an agent

Drop a folder under `mcp/agents/<name>/` with a `manifest.json` and an `agent.py`
exporting a class named `AgentAdapter`. It is auto-discovered — no other code
changes. See `mcp/agents/README.md` for the step-by-step guide and
`mcp/agents/_template_agent/` for a copyable starting point.

### Agent manifest

| Key | Required | Notes |
| --- | --- | --- |
| `name` | yes | Unique agent name (URL segment for direct invoke). |
| `version` | yes | Semver string. |
| `type` | yes | `inprocess` / `container` / `remote`. |
| `entrypoint` | no | Adapter module; default `agent.py`. |
| `capabilities` | yes | e.g. `["scout"]` — how missions select the agent. |
| `priority` | no | Lower = more urgent; used by the `priority` policy. |
| `latency_hint_ms` | no | Used by `latency_optimized`. |
| `resources` | no | CPU/memory hints (and container `image`). |
| `fallback_agents` | no | Agents to try if this one fails. |
| `endpoint` / `auth` | no | For `remote` agents (HTTP shim + `auth.api_key_env`). |

### Agent contract

```python
class AgentAdapter:
    name = "my_agent"; version = "1.0.0"; capabilities = ["scout"]

    def handle_task(self, task: dict, context: dict) -> dict:
        return {"status": "completed",           # completed|failed|in_progress|delegated
                "result": {...},                 # actions / detections / reasoning_trace
                "metadata": {"agent": "my_agent", "version": "1.0.0", "latency_ms": 3}}

    def health_check(self) -> dict: return {"status": "ok"}
    def on_start(self, config): ...              # optional lifecycle hooks
    def on_stop(self): ...
```

`context.upstream[<dep_task_id>]` holds each completed dependency's result. Emit
`result.followups = [ {id, type, capability|agent, payload, depends_on} ]` to
schedule **dynamic follow-up tasks** (this is how the coordinator escalates to a
`human_interface` agent).

## Mission templates

A mission is JSON (a named file under `mcp/missions/`, a path, or inline):

```json
{
  "mission_id": "urban_accident_response",
  "description": "...",
  "triggers": [{"type": "event", "on": "anomaly_candidate"}, {"type": "manual"}],
  "selection_policy": "priority",
  "timeouts": {"task_ms": 10000, "mission_ms": 60000},
  "retry_policy": {"max_retries": 1, "backoff_ms": 100},
  "escalation_rules": {"on_labels": ["accident"], "require_human": true, "min_confidence": 0.8},
  "tasks": [
    {"id": "scout", "capability": "scout", "payload": {...}},
    {"id": "analyze", "capability": "analyzer", "depends_on": ["scout"]},
    {"id": "coordinate", "capability": "coordinator", "depends_on": ["analyze"]}
  ]
}
```

The planner validates dependencies and rejects cycles. Each task names a
`capability` (resolved by the selection policy) or an explicit `agent`.

### Selection policies

`round_robin` · `load_aware` (lowest observed latency) · `latency_optimized`
(lowest `latency_hint_ms`) · `priority` (highest manifest priority).

## REST API

| Method & path | Purpose |
| --- | --- |
| `POST /api/mcp/missions` | Create + start a mission (template name/spec or inline `tasks`). |
| `GET  /api/mcp/missions/<id>` | Mission status + task graph. |
| `POST /api/mcp/missions/<id>/commands` | `pause` / `resume` / `cancel` / `escalate`. |
| `GET  /api/mcp/agents` | List agents, health, and metrics. |
| `POST /api/mcp/agents/<name>/invoke` | Direct-invoke an agent (testing). |

Responses include `request_id`, `mission_id`/`selected_agent`, and
`server_latency_ms`. Errors are `{error_code, message, details, fallback_actions}`.
All endpoints return `503 mcp_disabled` unless `ENABLE_MCP` is set.

## Telemetry & safety

- `dvsa_api.mcp.metrics` collects `tasks_scheduled`, `tasks_completed`,
  `task_failures`, per-agent `avg_latency_ms`, and `agent_health_status`, and
  emits mission/agent events to a pluggable sink (console by default).
- **Timeouts / retries / concurrency**: per-task `deadline_ms`, mission
  `retry_policy` with backoff and fallback agents, and `MCP_MAX_CONCURRENCY`.
- **Whitelisting**: `MCP_ALLOWED_AGENTS` / `MCP_AGENT_WHITELIST` (comma-separated)
  restrict which agents may load. Manifests/adapters are validated before use.
- **Human escalation**: high-impact actions require explicit confirmation via a
  `human_interface` agent before proceeding (`escalation_rules.require_human`).
- **Least privilege**: `inprocess` agents run in the server process (trusted
  code); run untrusted agents as `container`/`remote` with resource limits.

## Container & remote agents

Container/remote agents implement a tiny HTTP shim: `GET /health` and
`POST /handle_task` (returning `{status, result, metadata}`). Register a remote
agent by setting `"type": "remote"`, `"endpoint": "https://..."`, and optional
`"auth": {"api_key_env": "MY_KEY"}` in the manifest. Container runtime is opt-in
via `MCP_CONTAINER_RUNTIME=docker` (default `none` runs a local `agent.py` if
present). See `mcp/agents/_template_agent/Dockerfile` + `docker-compose.yml`.

## Configuration (environment variables)

| Var | Default | Purpose |
| --- | --- | --- |
| `ENABLE_MCP` | *(off)* | Master switch for the REST API. |
| `MCP_AGENTS_ROOT` | `mcp/agents` | Where agent folders are discovered. |
| `MCP_MISSIONS_ROOT` | `mcp/missions` | Where named mission templates resolve. |
| `MCP_MAX_CONCURRENCY` | `4` | Max tasks run concurrently per dependency wave. |
| `MCP_BUS_BACKEND` | `memory` | `memory` (default); `redis`/`kafka` need wiring. |
| `MCP_ALLOWED_AGENTS` | *(all)* | Comma-separated agent allow-list. |
| `MCP_CONTAINER_RUNTIME` | `none` | `docker` to launch container agents. |
| `MCP_HUMAN_AUTO_CONFIRM` | `1` | Auto-approve human confirmation (offline/dev). |

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `503 mcp_disabled` | Set `ENABLE_MCP=true`. |
| `agent_unavailable: no agent registered as '<name>'` | Folder missing `manifest.json`, or `name` mismatch. `GET /api/mcp/agents` lists what was found. |
| `agent_not_allowed` | Name excluded by `MCP_ALLOWED_AGENTS` / `MCP_AGENT_WHITELIST`. |
| `agent_contract_error: ... does not export 'AgentAdapter'` | Rename your class to exactly `AgentAdapter`. |
| `mission_error: ... contains a cycle` | Fix the `depends_on` graph. |
| `task_timeout` | Raise the task `deadline_ms` / mission `timeouts.task_ms`. |

## Example

`mcp/missions/urban_accident_response.json` runs
scout → analyzer (reasoning) → coordinator, and the coordinator dynamically
spawns a `human_interface` confirmation when a high-confidence accident is
detected. It runs end-to-end in the simulator and in the integration tests.
```bash
python scripts/mcp_cli.py simulate urban_accident_response
```
