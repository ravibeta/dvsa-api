"""One-shot deploy step: register the search index + Foundry agents in one run.

Invoked by Terraform (``solution-accelerator/deploy.tf``) after the ARM resources
exist, and runnable by hand::

    python -m core.azure.deploy                       # index + all agents
    python -m core.azure.deploy --only tool-agent-in-a-team   # one-time agent sync
    python -m core.azure.deploy --dry-run

It registers, in a single pass:

1. the shared base search index (``AZURE_SEARCH_INDEX_NAME``, ``dvsa-index``)
   built from :mod:`core.azure.index_schema` — the same builder the runtime uses
   for each per-account ``dvsa-index-s-<account>``;
2. every agent in ``solution-accelerator/agents/*.yaml`` (function tools
   generated from :mod:`core.azure.analyzer`), in both the versioned store and
   the classic store the runtime executes;
3. a read-only check that the model deployments those agents use exist.

Every step is attempted even if an earlier one fails; the process exits non-zero
if anything failed so a pipeline run surfaces it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("apps.azure")

DEFAULT_AGENTS_DIR = Path(__file__).resolve().parents[2] / "solution-accelerator" / "agents"


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
    import django  # noqa: PLC0415

    django.setup()


def ensure_base_index(config: Any, name: Optional[str] = None, *, dry_run: bool = False,
                      recreate: bool = False) -> Dict[str, Any]:
    """Create/update the shared base index with the full retrieval schema.

    Azure AI Search can't change an existing field's type in place, and the base
    index may predate this schema (it was created by the old ``index007``
    Terraform). If the in-place update is rejected, an **empty** index (or any
    index with ``recreate=True``) is dropped and rebuilt; an index that holds
    documents is left alone and reported, so a deploy never destroys data.
    """
    from .provisioning import AzureSdkProvisioner, DryRunProvisioner  # noqa: PLC0415

    name = name or config.search_index_name
    if dry_run or not (config.search_endpoint and config.search_admin_key):
        op = DryRunProvisioner(config).ensure_search_index(name)
        return {"index": name, "action": "would-ensure" if dry_run else "skipped-no-search-config", **{
            k: op[k] for k in ("vector_profiles", "semantic") if k in op}}
    prov = AzureSdkProvisioner(config)
    op = prov.ensure_search_index(name)
    if not op.get("error"):
        return {"index": name, "action": "ensured"}
    try:
        docs = prov._index_client().get_search_client(name).get_document_count()
    except Exception as exc:  # noqa: BLE001 - index missing/unreadable: surface the original error
        return {"index": name, "action": "error", "error": op["error"], "detail": str(exc)}
    if docs and not recreate:
        return {"index": name, "action": "error", "documents": docs,
                "error": f"index has {docs} documents and its schema can't be updated in place; "
                         "re-run with --recreate-index to drop and rebuild it",
                "detail": op["error"]}
    try:
        prov._index_client().delete_index(name)
    except Exception as exc:  # noqa: BLE001
        if "referencing" in str(exc):
            # A Search knowledge agent / knowledge base targets this index (the
            # portal's MCP knowledge-base connections point at those). Dropping
            # it would break them, so leave the index and say how to proceed.
            return {"index": name, "action": "blocked",
                    "error": "index can't be rebuilt: an AI Search knowledge agent still references it",
                    "remediation": "delete or retarget the referencing knowledge agent(s) (named in "
                                   "detail), re-run, then recreate them (core.azure agents recreate "
                                   "search-agent-in-a-team on first use)",
                    "detail": str(exc).splitlines()[0]}
        return {"index": name, "action": "error", "error": str(exc)}
    op = prov.ensure_search_index(name)
    if op.get("error"):
        return {"index": name, "action": "error", "error": op["error"]}
    return {"index": name, "action": "recreated", "dropped_documents": docs}


def deployments_used(specs: List[Any]) -> List[str]:
    return sorted({s.definition.get("model") for s in specs
                   if s.definition.get("model") and "/" not in s.definition["model"]})


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from .config import AzureEnvironmentConfig  # noqa: PLC0415
    from . import foundry_provisioning as fp  # noqa: PLC0415

    config = AzureEnvironmentConfig.from_settings()
    report: Dict[str, Any] = {"index": None, "agents": [], "classic_agents": [], "deployments": [], "ok": True}

    if not args.skip_index:
        report["index"] = ensure_base_index(config, args.index, dry_run=args.dry_run,
                                            recreate=args.recreate_index)
        if report["index"].get("action") == "error":
            report["ok"] = False

    if not args.skip_agents:
        endpoint = config.project_endpoint
        if not endpoint:
            report["agents"] = [{"action": "error", "error": "AZURE_PROJECT_ENDPOINT is not set"}]
            report["ok"] = False
            return report
        env = fp.env_from_project_endpoint(
            endpoint, subscription_id=config.subscription_id or "",
            resource_group=config.resource_group,
            search_service=config.search_service_name or "")
        specs = fp.load_agent_specs(args.agents_dir, env)
        client = fp.FoundryProjectClient(endpoint, retries=args.retries)
        report["agents"] = fp.sync_agents(client, specs, dry_run=args.dry_run,
                                          only=args.only.split(",") if args.only else None)
        report["deployments"] = fp.verify_deployments(client, deployments_used(specs))
        # The runtime executes the classic (asst_*) agents; keep those in sync too.
        try:
            from azure.ai.agents import AgentsClient  # noqa: PLC0415
            from azure.identity import DefaultAzureCredential  # noqa: PLC0415

            with AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential()) as ac:
                report["classic_agents"] = fp.sync_classic_agents(
                    ac, specs, dry_run=args.dry_run,
                    only=args.only.split(",") if args.only else None)
        except Exception as exc:  # noqa: BLE001
            report["classic_agents"] = [{"action": "error", "error": str(exc)}]
        if any(a.get("action") == "error" for a in report.get("classic_agents", [])):
            report["ok"] = False
        if any(a.get("action") == "error" for a in report["agents"]):
            report["ok"] = False
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Register the DVSA search index + Foundry agents.")
    p.add_argument("--agents-dir", default=str(DEFAULT_AGENTS_DIR))
    p.add_argument("--index", help="base index name (default AZURE_SEARCH_INDEX_NAME)")
    p.add_argument("--only", help="comma-separated agent names to sync")
    p.add_argument("--skip-index", action="store_true")
    p.add_argument("--recreate-index", action="store_true",
                   help="drop and rebuild the base index even if it holds documents")
    p.add_argument("--skip-agents", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retries", type=int, default=3, help="retries on 401/403 (RBAC propagation)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _setup_django()
    report = run(args)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
