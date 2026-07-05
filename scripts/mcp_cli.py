#!/usr/bin/env python
"""Local CLI for the MCP (Multi-Agent Control Plane) runtime.

Runs fully offline against the in-memory runtime — discover agents, run a mission
in local simulator mode, or produce a simulation report, without any external
services.

Examples
--------
    python scripts/mcp_cli.py discover
    python scripts/mcp_cli.py start
    python scripts/mcp_cli.py run-mission urban_accident_response
    python scripts/mcp_cli.py simulate urban_accident_response
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

# Allow running as a standalone script from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dvsa_api.mcp import registry  # noqa: E402
from dvsa_api.mcp.errors import MCPError, error_payload  # noqa: E402
from dvsa_api.mcp.session_manager import MCPSessionManager  # noqa: E402


def _load_simulator():
    """Import the offline simulator (top-level mcp/ is not a package)."""
    path = os.path.join(_REPO_ROOT, "mcp", "simulators", "environment.py")
    spec = importlib.util.spec_from_file_location("mcp_env_sim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_discover(_args) -> int:
    names = registry.discover_agents(force=True)
    agents = []
    for manifest in registry.list_manifests():
        agents.append({**manifest.public_dict(),
                       "health": registry.get_agent(manifest.name).health_check()})
    _print({"discovered": names, "agents": agents})
    return 0


def cmd_start(_args) -> int:
    registry.ensure_discovered()
    _print({
        "status": "ready",
        "agents": registry.list_agents(),
        "note": "REST API is served by Django (manage.py runserver) with "
                "ENABLE_MCP=true; agents run in-process.",
    })
    return 0


def cmd_run_mission(args) -> int:
    manager = MCPSessionManager()
    session = manager.create_mission(args.mission, context={}, run=True)
    _print(session.public_dict())
    return 0 if session.status in ("completed", "partial") else 1


def cmd_simulate(args) -> int:
    sim = _load_simulator()
    report = sim.simulate_mission(args.mission)
    _print(report)
    return 0 if report["status"] in ("completed", "partial") else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("discover", help="List discovered agents.")
    sub.add_parser("start", help="Discover agents and report runtime readiness.")

    p_run = sub.add_parser("run-mission", help="Run a mission locally.")
    p_run.add_argument("mission", help="Mission template name, path, or id.")

    p_sim = sub.add_parser("simulate", help="Run a mission against the simulator.")
    p_sim.add_argument("mission", default="urban_accident_response", nargs="?")

    args = parser.parse_args(argv)
    handlers = {
        "discover": cmd_discover, "start": cmd_start,
        "run-mission": cmd_run_mission, "simulate": cmd_simulate,
    }
    try:
        return handlers[args.command](args)
    except MCPError as exc:
        _print(error_payload(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
