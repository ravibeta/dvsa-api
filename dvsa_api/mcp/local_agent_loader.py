"""Safe dynamic loader for an agent folder's ``agent.py``.

An in-process MCP agent is a directory under ``mcp/agents/<name>/`` containing a
``manifest.json`` and an ``agent.py`` exporting a class named ``AgentAdapter``
implementing the contract in :mod:`dvsa_api.mcp.adapter_base`.

The file is imported *by path* (the agent folder need not be an installed
package), instantiated with optional config, and contract-validated before use.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import uuid
from typing import Any, Dict, Optional

from .adapter_base import AgentAdapter, validate_agent
from .errors import AgentContractError

AGENT_CLASS_NAME = "AgentAdapter"
_IMPORT_LOCK = threading.Lock()


def load_config(agent_dir: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``manifest['config']`` with an optional ``config.json`` sidecar."""
    config: Dict[str, Any] = dict(manifest.get("config") or {})
    sidecar = os.path.join(agent_dir, "config.json")
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                config.update(json.load(fh) or {})
        except (ValueError, OSError) as exc:  # pragma: no cover - defensive
            raise AgentContractError(
                f"invalid config.json in {agent_dir}: {exc}") from exc
    return config


def _import_module_from_path(module_path: str):
    """Import a standalone ``.py`` file by path under a unique module name."""
    if not os.path.isfile(module_path):
        raise AgentContractError(f"agent entrypoint not found: {module_path}")
    mod_name = f"dvsa_mcp_agent_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AgentContractError(f"cannot build import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    with _IMPORT_LOCK:
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - surface any import error
            sys.modules.pop(mod_name, None)
            raise AgentContractError(
                f"failed importing agent {module_path}: {exc}") from exc
    return module


def load_local_agent(
    agent_dir: str,
    manifest: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> AgentAdapter:
    """Import ``entrypoint`` from ``agent_dir`` and return a started, validated agent."""
    entrypoint = manifest.get("entrypoint", "agent.py")
    module_path = os.path.join(agent_dir, entrypoint)
    module = _import_module_from_path(module_path)

    agent_cls = getattr(module, AGENT_CLASS_NAME, None)
    if agent_cls is None or not isinstance(agent_cls, type):
        raise AgentContractError(
            f"{module_path} does not export a class named '{AGENT_CLASS_NAME}'")

    resolved = config if config is not None else load_config(agent_dir, manifest)
    try:
        instance = agent_cls(config=resolved)
    except TypeError:
        instance = agent_cls()

    agent = validate_agent(instance)
    # Fire the optional lifecycle hook; failures here are contract errors.
    on_start = getattr(agent, "on_start", None)
    if callable(on_start):
        try:
            on_start(resolved)
        except Exception as exc:  # noqa: BLE001
            raise AgentContractError(
                f"agent {module_path} on_start() failed: {exc}") from exc
    return agent
