"""Provision Azure AI Foundry project agents (and verify model deployments) as code.

Agents are *project data-plane* objects, not ARM resources, so Terraform can't
declare them natively. Instead the agent definitions live in
``solution-accelerator/agents/*.yaml`` and this module idempotently registers
them in the Foundry project — as part of the one-shot deploy
(:mod:`core.azure.deploy`, invoked from Terraform) — so every agent exists
*before* the first user query reaches ``/chat``.

Two things are generated from code instead of read from yaml:

- **Function tools** for the agents that call Python routines. The
  ``tool-agent-in-a-team`` / ``fn-agent-in-a-team`` tool lists are built from
  :func:`core.azure.analyzer.analyzer_functions` / ``image_user_functions`` using
  the same ``FunctionTool`` introspection the runtime uses, so the registered
  schemas can never drift from the callables that actually execute.
- **Environment ids**: ``${SUBSCRIPTION_ID}`` / ``${RESOURCE_GROUP}`` /
  ``${ACCOUNT}`` / ``${PROJECT}`` placeholders in the yaml are substituted, so
  the same definitions deploy to any subscription.

Only the REST data plane is used (``azure-ai-projects`` 1.0.0b11 predates the
versioned-agents API), authenticated with ``DefaultAzureCredential``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger("apps.azure")

AI_SCOPE = "https://ai.azure.com/.default"
API_VERSION = "v1"

# Agent -> name of the zero-arg callable in core.azure.analyzer whose functions
# become that agent's function tools.
FUNCTION_TOOL_SOURCES: Dict[str, str] = {
    "tool-agent-in-a-team": "analyzer_functions",
    "fn-agent-in-a-team": "image_user_functions",
}

_PLACEHOLDER = re.compile(r"\$\{([A-Z_]+)\}")


class FoundryProvisioningError(RuntimeError):
    """A Foundry data-plane call failed."""


# --------------------------------------------------------------------------
# Agent specs
# --------------------------------------------------------------------------
@dataclass
class AgentSpec:
    name: str
    definition: Dict[str, Any]
    description: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    source: str = ""


def _substitute(obj: Any, env: Dict[str, str]) -> Any:
    """Replace ``${NAME}`` placeholders (from ``env``) throughout a JSON-like tree."""
    if isinstance(obj, str):
        return _PLACEHOLDER.sub(lambda m: env.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, list):
        return [_substitute(i, env) for i in obj]
    if isinstance(obj, dict):
        return {k: _substitute(v, env) for k, v in obj.items()}
    return obj


def load_agent_specs(agents_dir: str | Path, env: Optional[Dict[str, str]] = None) -> List[AgentSpec]:
    """Load ``*.yaml`` agent definitions, keeping only what's needed to *create* them.

    The exported yaml also carries instance identities, blueprint ids,
    timestamps, etc. — runtime artifacts that must not be sent back.
    """
    import yaml  # noqa: PLC0415

    env = env or {}
    specs: List[AgentSpec] = []
    for path in sorted(Path(agents_dir).glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        definition = doc.get("definition")
        name = doc.get("name")
        if not name or not isinstance(definition, dict):
            logger.warning("skipping %s: no name/definition", path.name)
            continue
        meta = {k: str(v) for k, v in (doc.get("metadata") or {}).items()
                if k.startswith("microsoft.") is False and k != "modified_at"}
        specs.append(AgentSpec(
            name=name,
            definition=_substitute(definition, env),
            description=str(doc.get("description") or ""),
            metadata=meta,
            source=path.name,
        ))
    return specs


def function_tools_from(functions: Iterable[Callable[..., Any]]) -> List[Dict[str, Any]]:
    """Responses-API ``function`` tool dicts from Python callables.

    Reuses ``azure.ai.agents.models.FunctionTool`` (what the runtime registers
    with) so the schema is derived exactly the way execution derives it.
    """
    from azure.ai.agents.models import FunctionTool  # noqa: PLC0415

    tools: List[Dict[str, Any]] = []
    for d in FunctionTool(functions=set(functions)).definitions:
        fn = d.function
        tools.append({
            "type": "function",
            "name": fn.name,
            "description": fn.description or "No description",
            "parameters": json.loads(json.dumps(fn.parameters)),
            "strict": False,
        })
    return sorted(tools, key=lambda t: t["name"])


def apply_function_tools(spec: AgentSpec) -> AgentSpec:
    """Replace ``function`` tools with ones generated from the analyzer module."""
    source = FUNCTION_TOOL_SOURCES.get(spec.name)
    if not source:
        return spec
    from core.azure import analyzer  # noqa: PLC0415

    generated = function_tools_from(getattr(analyzer, source)())
    kept = [t for t in spec.definition.get("tools", []) if t.get("type") != "function"]
    spec.definition = {**spec.definition, "tools": generated + kept}
    return spec


def _canonical(definition: Dict[str, Any]) -> str:
    """Order-insensitive form used to decide whether a new version is needed."""
    d = json.loads(json.dumps(definition))
    tools = d.get("tools") or []
    d["tools"] = sorted(tools, key=lambda t: (t.get("type", ""), t.get("name") or t.get("server_label") or ""))
    return json.dumps(d, sort_keys=True)


# --------------------------------------------------------------------------
# REST client
# --------------------------------------------------------------------------
class FoundryProjectClient:
    """Minimal REST client for a Foundry project's agents + deployments."""

    def __init__(self, project_endpoint: str, credential: Any = None,
                 session: Any = None, retries: int = 3, retry_wait: float = 5.0) -> None:
        self.endpoint = project_endpoint.rstrip("/")
        self._credential = credential
        self._session = session
        self.retries = retries
        self.retry_wait = retry_wait

    # -- plumbing ----------------------------------------------------------
    def _token(self) -> str:
        if self._credential is None:
            from azure.identity import DefaultAzureCredential  # noqa: PLC0415

            self._credential = DefaultAzureCredential()
        return self._credential.get_token(AI_SCOPE).token

    def _http(self) -> Any:
        if self._session is None:
            import requests  # noqa: PLC0415

            self._session = requests.Session()
        return self._session

    def _request(self, method: str, path: str, *, body: Any = None,
                 ok_missing: bool = False) -> Optional[Dict[str, Any]]:
        url = f"{self.endpoint}{path}{'&' if '?' in path else '?'}api-version={API_VERSION}"
        last: Any = None
        for attempt in range(self.retries + 1):
            resp = self._http().request(
                method, url, timeout=60,
                headers={"Authorization": f"Bearer {self._token()}",
                         "Content-Type": "application/json"},
                data=json.dumps(body) if body is not None else None,
            )
            last = resp
            if resp.status_code == 404 and ok_missing:
                return None
            # Fresh role assignments can take minutes to propagate.
            if resp.status_code in (401, 403) and attempt < self.retries:
                logger.warning("%s %s -> %s; retrying (RBAC propagation?)",
                               method, path, resp.status_code)
                time.sleep(self.retry_wait)
                continue
            if resp.status_code >= 400:
                break
            return resp.json() if getattr(resp, "content", b"") else {}
        raise FoundryProvisioningError(
            f"{method} {path} failed: {last.status_code} {str(last.text)[:300]}")

    # -- agents ------------------------------------------------------------
    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        return self._request("GET", f"/agents/{name}", ok_missing=True)

    def create_agent_version(self, spec: AgentSpec) -> Dict[str, Any]:
        body: Dict[str, Any] = {"definition": spec.definition,
                                "description": spec.description}
        if spec.metadata:
            body["metadata"] = spec.metadata
        return self._request("POST", f"/agents/{spec.name}/versions", body=body) or {}

    # -- deployments (read-only) -------------------------------------------
    def get_deployment(self, name: str) -> Optional[Dict[str, Any]]:
        return self._request("GET", f"/deployments/{name}", ok_missing=True)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def sync_agents(client: FoundryProjectClient, specs: Iterable[AgentSpec], *,
                dry_run: bool = False, only: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Idempotently register each spec: create the agent or a new version if it differs."""
    wanted = set(only) if only else None
    results: List[Dict[str, Any]] = []
    for spec in specs:
        if wanted is not None and spec.name not in wanted:
            continue
        try:
            spec = apply_function_tools(spec)
            current = client.get_agent(spec.name)
            latest = ((current or {}).get("versions") or {}).get("latest") or {}
            if current and _canonical(latest.get("definition") or {}) == _canonical(spec.definition):
                results.append({"agent": spec.name, "action": "unchanged",
                                "version": latest.get("version")})
                continue
            action = "update" if current else "create"
            if dry_run:
                results.append({"agent": spec.name, "action": f"would-{action}",
                                "tools": len(spec.definition.get("tools") or [])})
                continue
            created = client.create_agent_version(spec)
            results.append({"agent": spec.name, "action": action,
                            "version": created.get("version"),
                            "tools": len(spec.definition.get("tools") or [])})
        except Exception as exc:  # noqa: BLE001 - report every agent, fail at the end
            logger.error("agent %s sync failed: %s", spec.name, exc)
            results.append({"agent": spec.name, "action": "error", "error": str(exc)})
    return results


def verify_deployments(client: FoundryProjectClient, names: Iterable[str]) -> List[Dict[str, Any]]:
    """Report whether each model deployment the agents depend on exists in the project."""
    out = []
    for name in names:
        try:
            dep = client.get_deployment(name)
            out.append({"deployment": name, "present": dep is not None,
                        "model": (dep or {}).get("modelName")})
        except Exception as exc:  # noqa: BLE001
            out.append({"deployment": name, "present": False, "error": str(exc)})
    return out


def env_from_project_endpoint(project_endpoint: str, *, subscription_id: str = "",
                              resource_group: str = "") -> Dict[str, str]:
    """Placeholder values (``${ACCOUNT}``/``${PROJECT}``/...) for yaml substitution."""
    m = re.match(r"https://([^./]+)\.services\.ai\.azure\.com/api/projects/([^/?]+)",
                 project_endpoint)
    account, project = (m.group(1), m.group(2)) if m else ("", "")
    return {
        "SUBSCRIPTION_ID": subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID", ""),
        "RESOURCE_GROUP": resource_group or os.environ.get("AZURE_RESOURCE_GROUP", ""),
        "ACCOUNT": account,
        "PROJECT": project,
    }
