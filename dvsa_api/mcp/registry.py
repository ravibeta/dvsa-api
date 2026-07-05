"""Agent discovery + registry + capability-based selection for the MCP runtime.

Responsibilities
----------------
* **Registry** — map an agent ``name`` to a lazy adapter factory
  (:func:`register`, :func:`get_agent`, :func:`list_agents`).
* **Discovery** — scan ``mcp/agents/*`` for folders containing a ``manifest.json``
  and register each as a lazy factory. ``inprocess`` agents are imported on first
  use; ``container``/``remote`` agents wrap an HTTP shim. See
  :func:`discover_agents` / :func:`ensure_discovered`.
* **Selection** — :func:`select_agent` picks an agent that offers a *capability*
  under a policy: ``round_robin`` / ``load_aware`` / ``latency_optimized`` /
  ``priority``, using manifest metadata + runtime metrics.

Safety: an optional allow-list (``MCP_ALLOWED_AGENTS`` / ``MCP_AGENT_WHITELIST``)
restricts which agent folders may be loaded. Everything is additive and
dependency-free.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapter_base import AgentAdapter, validate_agent
from .errors import AgentNotAllowedError, AgentUnavailableError
from .metrics import get_metrics

_FACTORIES: Dict[str, Callable[[], AgentAdapter]] = {}
_INSTANCES: Dict[str, AgentAdapter] = {}
_MANIFESTS: Dict[str, "AgentManifest"] = {}
_RR_COUNTERS: Dict[str, "itertools.count[int]"] = {}
_LOCK = threading.RLock()
_DISCOVERED = False

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_AGENTS_ROOT = os.environ.get(
    "MCP_AGENTS_ROOT", os.path.join(_REPO_ROOT, "mcp", "agents"))

# Static latency hints (ms) by agent type when a manifest omits one.
_LATENCY_HINT_BY_TYPE = {"inprocess": 20, "container": 200, "remote": 400}
VALID_POLICIES = ("round_robin", "load_aware", "latency_optimized", "priority")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentManifest:
    """Parsed ``manifest.json`` describing a discovered agent folder."""

    name: str
    version: str
    type: str  # inprocess | container | remote
    entrypoint: str
    agent_dir: str
    capabilities: List[str] = field(default_factory=list)
    priority: int = 5
    latency_hint_ms: int = 20
    resources: Dict[str, Any] = field(default_factory=dict)
    fallback_agents: List[str] = field(default_factory=list)
    endpoint: Optional[str] = None
    image: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], agent_dir: str) -> "AgentManifest":
        name = data.get("name")
        if not name:
            raise AgentUnavailableError(
                f"agent manifest in {agent_dir} missing required 'name'")
        atype = (data.get("type") or "inprocess").lower()
        return cls(
            name=name,
            version=str(data.get("version", "0.0.0")),
            type=atype,
            entrypoint=data.get("entrypoint", "agent.py"),
            agent_dir=agent_dir,
            capabilities=list(data.get("capabilities") or []),
            priority=int(data.get("priority", 5)),
            latency_hint_ms=int(data.get(
                "latency_hint_ms", _LATENCY_HINT_BY_TYPE.get(atype, 100))),
            resources=dict(data.get("resources") or {}),
            fallback_agents=list(data.get("fallback_agents") or []),
            endpoint=data.get("endpoint"),
            image=(data.get("resources") or {}).get("image") or data.get("image"),
            raw=data,
        )

    def public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "type": self.type,
            "capabilities": self.capabilities, "priority": self.priority,
            "latency_hint_ms": self.latency_hint_ms, "resources": self.resources,
            "fallback_agents": self.fallback_agents,
        }


# --------------------------------------------------------------------------- #
# Allow-list
# --------------------------------------------------------------------------- #
def _allowlist() -> Optional[set]:
    """Return the set of allowed agent names, or ``None`` when unrestricted."""
    raw = os.environ.get("MCP_ALLOWED_AGENTS") or os.environ.get("MCP_AGENT_WHITELIST")
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _is_allowed(name: str) -> bool:
    allow = _allowlist()
    return allow is None or name in allow


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def register(name: str, factory: Callable[[], AgentAdapter]) -> None:
    """Register a lazy agent ``factory`` under ``name`` (idempotent)."""
    with _LOCK:
        _FACTORIES[name] = factory
        _INSTANCES.pop(name, None)


def register_instance(name: str, agent: AgentAdapter) -> None:
    """Register an already-constructed agent instance."""
    validate_agent(agent)
    with _LOCK:
        _INSTANCES[name] = agent
        _FACTORIES.setdefault(name, lambda: agent)


def get_agent(name: str) -> AgentAdapter:
    """Return the agent registered as ``name`` (constructing it lazily)."""
    if not _is_allowed(name):
        raise AgentNotAllowedError(
            f"agent '{name}' is not in the allow-list", details={"agent": name})
    with _LOCK:
        if name in _INSTANCES:
            return _INSTANCES[name]
        factory = _FACTORIES.get(name)
        if factory is None:
            ensure_discovered()
            factory = _FACTORIES.get(name)
        if factory is None:
            raise AgentUnavailableError(
                f"no agent registered as '{name}'",
                details={"available": list(_FACTORIES)})
        agent = validate_agent(factory())
        _INSTANCES[name] = agent
    get_metrics().set_health(name, _safe_health(agent))
    return agent


def _safe_health(agent: AgentAdapter) -> str:
    try:
        return str(agent.health_check().get("status", "unknown"))
    except Exception:  # noqa: BLE001
        return "error"


def list_agents() -> List[str]:
    """Return the names of all registered agents (triggers discovery once)."""
    ensure_discovered()
    with _LOCK:
        return sorted(n for n in _FACTORIES if _is_allowed(n))


def get_manifest(name: str) -> Optional[AgentManifest]:
    ensure_discovered()
    with _LOCK:
        return _MANIFESTS.get(name)


def list_manifests() -> List[AgentManifest]:
    ensure_discovered()
    with _LOCK:
        return [m for m in _MANIFESTS.values() if _is_allowed(m.name)]


def clear() -> None:
    """Drop all registrations + discovery state (used by tests for isolation)."""
    global _DISCOVERED
    with _LOCK:
        _FACTORIES.clear()
        _INSTANCES.clear()
        _MANIFESTS.clear()
        _RR_COUNTERS.clear()
        _DISCOVERED = False


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _make_factory(manifest: AgentManifest) -> Callable[[], AgentAdapter]:
    """Build the lazy factory that instantiates an agent from its manifest."""

    def factory() -> AgentAdapter:
        agent_file = os.path.join(manifest.agent_dir, manifest.entrypoint)
        # A local agent.py always wins for in-process/dev use, even for
        # container/remote types when their runtime is disabled.
        runtime = os.environ.get("MCP_CONTAINER_RUNTIME", "none").lower()
        if manifest.type == "inprocess" or os.path.isfile(agent_file):
            if manifest.type == "inprocess" or runtime == "none":
                from .local_agent_loader import load_local_agent  # noqa: PLC0415
                return load_local_agent(manifest.agent_dir, manifest.raw)
        from .remote_agent import RemoteAgentProxy  # noqa: PLC0415
        return RemoteAgentProxy(manifest)

    return factory


def discover_agents(root: Optional[str] = None, *, force: bool = False) -> List[str]:
    """Scan ``root`` for agent folders and register each; return their names.

    An agent folder is an immediate subdirectory with a ``manifest.json``. Names
    starting with ``.`` are ignored. Malformed manifests disable just that agent.
    Allow-listed-out agents are still discovered but cannot be loaded.
    """
    global _DISCOVERED
    root = root or DEFAULT_AGENTS_ROOT
    found: List[str] = []
    if not os.path.isdir(root):
        with _LOCK:
            _DISCOVERED = True
        return found

    for entry in sorted(os.listdir(root)):
        if entry.startswith("."):
            continue
        agent_dir = os.path.join(root, entry)
        manifest_path = os.path.join(agent_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with _LOCK:
            if entry in _MANIFESTS and not force:
                found.append(entry)
                continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            manifest = AgentManifest.from_dict(data, agent_dir)
        except (ValueError, OSError, AgentUnavailableError):
            continue
        with _LOCK:
            _MANIFESTS[manifest.name] = manifest
            _FACTORIES[manifest.name] = _make_factory(manifest)
            _INSTANCES.pop(manifest.name, None)
        found.append(manifest.name)

    with _LOCK:
        _DISCOVERED = True
    return found


def ensure_discovered() -> None:
    """Run :func:`discover_agents` once (thread-safe, idempotent)."""
    if _DISCOVERED:
        return
    with _LOCK:
        if _DISCOVERED:
            return
    discover_agents()


# --------------------------------------------------------------------------- #
# Capability-based selection
# --------------------------------------------------------------------------- #
def agents_for_capability(capability: str) -> List[AgentManifest]:
    """Return allow-listed manifests advertising ``capability``."""
    return [m for m in list_manifests() if capability in m.capabilities]


def select_agent(capability: str, policy: str = "round_robin") -> str:
    """Return the name of an agent offering ``capability`` under ``policy``.

    * ``round_robin`` — rotate through the candidates.
    * ``load_aware`` — lowest observed average latency (a load proxy).
    * ``latency_optimized`` — lowest static ``latency_hint_ms``.
    * ``priority`` — highest manifest ``priority`` (ties → lowest latency hint).
    """
    candidates = agents_for_capability(capability)
    if not candidates:
        raise AgentUnavailableError(
            f"no agent offers capability '{capability}'",
            details={"capability": capability, "policy": policy})

    if policy == "latency_optimized":
        return min(candidates, key=lambda m: m.latency_hint_ms).name
    if policy == "priority":
        return max(candidates, key=lambda m: (m.priority, -m.latency_hint_ms)).name
    if policy == "load_aware":
        metrics = get_metrics()
        # Prefer the least-latent agent; unseen agents (0.0) are tried first.
        return min(candidates, key=lambda m: metrics.avg_latency_ms(m.name)).name
    if policy == "round_robin":
        names = sorted(m.name for m in candidates)
        with _LOCK:
            counter = _RR_COUNTERS.setdefault(capability, itertools.count())
            idx = next(counter)
        return names[idx % len(names)]

    raise AgentUnavailableError(
        f"unknown selection policy '{policy}'",
        details={"valid": list(VALID_POLICIES)})
