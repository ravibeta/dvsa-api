#!/usr/bin/env python
"""Local CLI to exercise the Azure Foundry reasoning session lifecycle.

Runs fully offline (dry-run) by default, so you can create a session, run an
inference call, inspect status and tear down — without any Azure credentials.

Examples
--------
    python scripts/foundry_session_cli.py create --model o1-mini --dry-run
    python scripts/foundry_session_cli.py infer  --session <id> --query "accidents?"
    python scripts/foundry_session_cli.py status --session <id>
    python scripts/foundry_session_cli.py teardown --session <id>

This is a thin wrapper over
``dvsa_api.reasoning.azure_session_manager`` and ``AzureFoundryAdapter``; the
manager itself is also runnable as
``python -m dvsa_api.reasoning.azure_session_manager --dry-run``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a standalone script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dvsa_api.reasoning.azure_foundry_adapter import AzureFoundryAdapter  # noqa: E402
from dvsa_api.reasoning.azure_session_manager import (  # noqa: E402
    AzureFoundrySessionManager,
    reset_session_manager,
)
from dvsa_api.reasoning.errors import ReasoningError, error_payload  # noqa: E402


def _manager(dry_run: bool) -> AzureFoundrySessionManager:
    # Share one manager across the CLI process so `infer/status/teardown` can
    # see the session created by `create` within the same invocation chain.
    mgr = AzureFoundrySessionManager(dry_run=True if dry_run else None)
    reset_session_manager(mgr)
    return mgr


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Force offline mode (no cloud calls).")
    # Shared parent so --dry-run is also accepted *after* the subcommand
    # (e.g. `create --dry-run`), matching the documented examples.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true",
                        help="Force offline mode (no cloud calls).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a session.", parents=[common])
    p_create.add_argument("--model", default="o1-mini")
    p_create.add_argument("--minutes", type=int, default=30)
    p_create.add_argument("--max-cost", type=float, default=1.0)
    p_create.add_argument("--private-networking", action="store_true")

    p_infer = sub.add_parser("infer", help="Run one inference call.", parents=[common])
    p_infer.add_argument("--session", required=True)
    p_infer.add_argument("--query", default="Detect anomalies in the aerial scene.")
    p_infer.add_argument("--tracks", type=int, default=2,
                         help="Number of synthetic object tracks to include.")

    p_status = sub.add_parser("status", help="Show session status.", parents=[common])
    p_status.add_argument("--session", required=True)

    p_teardown = sub.add_parser("teardown", help="Tear down a session.", parents=[common])
    p_teardown.add_argument("--session", required=True)

    args = parser.parse_args(argv)
    mgr = _manager(args.dry_run)

    try:
        if args.command == "create":
            session = mgr.create_session(
                model_name=args.model, max_duration_minutes=args.minutes,
                max_cost_usd=args.max_cost,
                private_networking=args.private_networking)
            _print(session.to_public_dict())
        elif args.command == "infer":
            adapter = AzureFoundryAdapter(session_id=args.session, manager=mgr)
            context = {
                "query": args.query,
                "tracks": [{"id": f"veh-{i}"} for i in range(args.tracks)],
            }
            _print(adapter.predict(context))
        elif args.command == "status":
            _print(mgr.get_session(args.session).to_public_dict())
        elif args.command == "teardown":
            _print(mgr.teardown_session(args.session, reason="cli"))
    except ReasoningError as exc:
        _print(error_payload(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
