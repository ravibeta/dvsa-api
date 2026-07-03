"""Lifecycle manager for ephemeral Azure Foundry reasoning sessions.

Responsibilities
----------------
* **provision** ephemeral Foundry resources for a session (Terraform ``plan`` +
  ``apply`` by default, or the Azure SDK when ``USE_SDK_PROVISION=true``),
* **track** per-session usage, cost and TTL in an in-memory store (pluggable),
* **enforce** conservative cost / quota / time limits with soft + hard stops,
* **tear down** resources — automatically on TTL/inactivity/cost breach or on an
  explicit call — *idempotently*.

Everything is **offline-safe**: with no Terraform binary, no credentials and no
``AZURE_SUBSCRIPTION_ID`` the manager records what it *would* do and hands back a
synthetic endpoint, so the API, tests and the CLI all run with zero cloud calls
(same philosophy as ``core/azure/provisioning.py``'s dry-run backend).

CLI
---
``python -m dvsa_api.reasoning.azure_session_manager --model o1-mini --dry-run``
creates a session locally and prints its metadata as JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .azure_sdk_wrapper import AzureSdkWrapper
from .errors import (
    CostLimitExceeded,
    ProvisioningError,
    QuotaExceeded,
    SessionNotFoundError,
    TeardownError,
)

logger = logging.getLogger("dvsa_api.reasoning")

# --- safe defaults (all overridable via env) -------------------------------
DEFAULT_DURATION_MINUTES = int(os.environ.get("FOUNDRY_DEFAULT_DURATION_MINUTES", "30"))
DEFAULT_MAX_COST_USD = float(os.environ.get("FOUNDRY_DEFAULT_MAX_COST_USD", "1.0"))
INACTIVITY_TIMEOUT_MINUTES = int(
    os.environ.get("FOUNDRY_INACTIVITY_TIMEOUT_MINUTES", "15")
)
MAX_ACTIVE_SESSIONS = int(os.environ.get("FOUNDRY_MAX_ACTIVE_SESSIONS", "25"))
MAX_ACTIVE_SESSIONS_PER_USER = int(
    os.environ.get("FOUNDRY_MAX_ACTIVE_SESSIONS_PER_USER", "5")
)
SOFT_STOP_FRACTION = float(os.environ.get("FOUNDRY_SOFT_STOP_FRACTION", "0.8"))

# Approximate hourly USD price per Foundry model SKU, used to estimate cost
# up-front (before apply) and to accrue per-call spend. Deliberately rough and
# conservative — the point is a guardrail, not billing accuracy.
SKU_HOURLY_USD: Dict[str, float] = {
    "o1": 2.50,
    "o1-mini": 0.60,
    "o3-mini": 0.55,
    "o4-mini": 0.55,
    "gpt-4o": 1.20,
    "gpt-4o-mini": 0.30,
    "default": 1.00,
}
# Rough marginal cost per 1K tokens, added to accrued spend as calls run.
TOKEN_COST_PER_1K_USD = float(os.environ.get("FOUNDRY_TOKEN_COST_PER_1K_USD", "0.01"))


def sku_hourly_rate(model_name: str) -> float:
    """Return the hourly USD estimate for ``model_name`` (falls back to default)."""
    return SKU_HOURLY_USD.get(model_name, SKU_HOURLY_USD["default"])


def estimate_session_cost(model_name: str, duration_minutes: int) -> float:
    """Up-front cost estimate for holding ``model_name`` for ``duration``."""
    return round(sku_hourly_rate(model_name) * (duration_minutes / 60.0), 4)


def _slug(value: str) -> str:
    """Azure-name-safe slug (lowercase alphanumerics + single dashes)."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "s"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionUsage:
    """Running counters for one session (drives soft/hard cost stops)."""

    calls: int = 0
    tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class FoundrySession:
    """Metadata + live state for one ephemeral reasoning session."""

    session_id: str
    model_name: str
    user_id: Optional[str]
    resource_group: str
    managed_identity_name: str
    key_vault_secret_name: str
    endpoint_url: str
    location: str
    private_networking: bool
    max_duration_minutes: int
    max_cost_usd: float
    cost_estimate_usd: float
    created_at: datetime
    expires_at: datetime
    last_activity_at: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    status: str = "active"  # active | over_budget | expired | torn_down | failed
    provisioner: str = "dryrun"  # terraform | sdk | dryrun
    usage: SessionUsage = field(default_factory=SessionUsage)
    provision_ops: List[Dict[str, Any]] = field(default_factory=list)

    # ----- derived state -------------------------------------------------
    def is_expired(self, now: Optional[datetime] = None) -> bool:
        return (now or _utcnow()) >= self.expires_at

    def is_inactive(self, now: Optional[datetime] = None,
                    timeout_minutes: int = INACTIVITY_TIMEOUT_MINUTES) -> bool:
        cutoff = self.last_activity_at + timedelta(minutes=timeout_minutes)
        return (now or _utcnow()) >= cutoff

    def soft_limit_usd(self) -> float:
        return round(self.max_cost_usd * SOFT_STOP_FRACTION, 4)

    def over_soft_limit(self) -> bool:
        return self.usage.estimated_cost_usd >= self.soft_limit_usd()

    def over_hard_limit(self) -> bool:
        return self.usage.estimated_cost_usd >= self.max_cost_usd

    def to_public_dict(self) -> Dict[str, Any]:
        """JSON-serializable view safe for API responses (no secrets/keys)."""
        d = asdict(self)
        d.pop("key_vault_secret_name", None)  # name is not a secret, but omit anyway
        for k in ("created_at", "expires_at", "last_activity_at"):
            d[k] = getattr(self, k).isoformat()
        d["usage"] = asdict(self.usage)
        d["soft_limit_usd"] = self.soft_limit_usd()
        return d


class SessionStore:
    """In-memory session store. Swap for Redis by matching this interface."""

    def __init__(self) -> None:
        self._data: Dict[str, FoundrySession] = {}
        self._lock = threading.RLock()

    def put(self, session: FoundrySession) -> None:
        with self._lock:
            self._data[session.session_id] = session

    def get(self, session_id: str) -> Optional[FoundrySession]:
        with self._lock:
            return self._data.get(session_id)

    def pop(self, session_id: str) -> Optional[FoundrySession]:
        with self._lock:
            return self._data.pop(session_id, None)

    def all(self) -> List[FoundrySession]:
        with self._lock:
            return list(self._data.values())

    def count(self, user_id: Optional[str] = None) -> int:
        with self._lock:
            if user_id is None:
                return len(self._data)
            return sum(1 for s in self._data.values() if s.user_id == user_id)


class AzureFoundrySessionManager:
    """Create, track and tear down ephemeral Foundry reasoning sessions."""

    def __init__(
        self,
        *,
        store: Optional[SessionStore] = None,
        sdk: Optional[AzureSdkWrapper] = None,
        terraform_dir: Optional[str] = None,
        location: Optional[str] = None,
        use_sdk_provision: Optional[bool] = None,
        dry_run: Optional[bool] = None,
    ) -> None:
        self.store = store or SessionStore()
        self.sdk = sdk or AzureSdkWrapper()
        self.terraform_dir = terraform_dir or os.environ.get(
            "AZURE_FOUNDRY_TERRAFORM_DIR", "infra/azure/foundry_module"
        )
        self.location = location or os.environ.get("AZURE_LOCATION", "eastus")
        self.subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        if use_sdk_provision is None:
            use_sdk_provision = os.environ.get(
                "USE_SDK_PROVISION", ""
            ).strip().lower() in ("1", "true", "yes", "on")
        self.use_sdk_provision = use_sdk_provision
        # Dry-run when explicitly requested, or when nothing can actually
        # provision (no terraform binary AND no subscription for the SDK path).
        if dry_run is None:
            has_tf = bool(shutil.which("terraform")) and bool(self.terraform_dir)
            dry_run = not (has_tf or (self.use_sdk_provision and self.subscription_id))
        self.dry_run = dry_run
        self._lock = threading.RLock()

    # ================= session creation =================================
    def create_session(
        self,
        *,
        model_name: str,
        user_id: Optional[str] = None,
        max_duration_minutes: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        private_networking: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ) -> FoundrySession:
        """Provision (or dry-run) a session and return its metadata.

        Raises
        ------
        QuotaExceeded
            If global or per-user active-session limits are hit.
        CostLimitExceeded
            If the up-front estimate already exceeds ``max_cost_usd``.
        ProvisioningError
            If real provisioning fails (best-effort cleanup is attempted).
        """
        duration = int(max_duration_minutes or DEFAULT_DURATION_MINUTES)
        cost_cap = float(max_cost_usd if max_cost_usd is not None else DEFAULT_MAX_COST_USD)
        tags = dict(tags or {})

        self._check_quota(user_id)

        cost_estimate = estimate_session_cost(model_name, duration)
        if cost_estimate > cost_cap:
            raise CostLimitExceeded(
                "estimated session cost exceeds max_cost_usd",
                details={"cost_estimate": cost_estimate, "max_cost_usd": cost_cap},
            )

        session_id = uuid.uuid4().hex
        short = _slug(f"{model_name}-{session_id[:8]}")
        resource_group = f"rg-foundry-{short}"
        identity_name = f"id-foundry-{short}"
        secret_name = f"foundry-key-{short}"
        vault_tags = {"session_id": session_id, "component": "dvsa-foundry", **tags}

        now = _utcnow()
        try:
            prov = self._provision(
                session_id=session_id,
                model_name=model_name,
                resource_group=resource_group,
                identity_name=identity_name,
                secret_name=secret_name,
                private_networking=private_networking,
                duration=duration,
                tags=vault_tags,
            )
        except ProvisioningError:
            # Best-effort cleanup so a failed apply leaves nothing behind.
            self._best_effort_cleanup(resource_group, secret_name)
            raise

        session = FoundrySession(
            session_id=session_id,
            model_name=model_name,
            user_id=str(user_id) if user_id is not None else None,
            resource_group=resource_group,
            managed_identity_name=identity_name,
            key_vault_secret_name=secret_name,
            endpoint_url=prov["endpoint_url"],
            location=self.location,
            private_networking=private_networking,
            max_duration_minutes=duration,
            max_cost_usd=cost_cap,
            cost_estimate_usd=cost_estimate,
            created_at=now,
            expires_at=now + timedelta(minutes=duration),
            last_activity_at=now,
            tags=tags,
            status="active",
            provisioner=prov["provisioner"],
            provision_ops=prov["operations"],
        )
        self.store.put(session)
        logger.info(
            "created foundry session %s (model=%s, provisioner=%s, est=$%.4f)",
            session_id, model_name, session.provisioner, cost_estimate,
        )
        return session

    def _check_quota(self, user_id: Optional[str]) -> None:
        if self.store.count() >= MAX_ACTIVE_SESSIONS:
            raise QuotaExceeded(
                "global active-session limit reached",
                details={"limit": MAX_ACTIVE_SESSIONS},
            )
        if user_id is not None and (
            self.store.count(str(user_id)) >= MAX_ACTIVE_SESSIONS_PER_USER
        ):
            raise QuotaExceeded(
                "per-user active-session limit reached",
                details={"limit": MAX_ACTIVE_SESSIONS_PER_USER, "user": str(user_id)},
            )

    # ================= provisioning backends ============================
    def _provision(
        self, *, session_id: str, model_name: str, resource_group: str,
        identity_name: str, secret_name: str, private_networking: bool,
        duration: int, tags: Dict[str, str],
    ) -> Dict[str, Any]:
        """Dispatch to terraform / sdk / dry-run and normalise the result."""
        tf_vars = self._tf_vars(
            resource_group=resource_group, session_name=_slug(session_id[:12]),
            model_name=model_name, identity_name=identity_name,
            secret_name=secret_name, private_networking=private_networking,
            ttl_minutes=duration, tags=tags,
        )
        if self.dry_run:
            return self._provision_dryrun(session_id, model_name, resource_group,
                                          secret_name, tf_vars)
        if self.use_sdk_provision:
            return self._provision_sdk(session_id, model_name, resource_group,
                                       secret_name, tf_vars)
        return self._provision_terraform(session_id, model_name, resource_group,
                                          secret_name, tf_vars)

    def _synthetic_endpoint(self, session_id: str, model_name: str) -> str:
        return (
            f"https://foundry-{_slug(session_id[:8])}.example-foundry.local/"
            f"reasoning/{_slug(model_name)}/score"
        )

    def _provision_dryrun(self, session_id, model_name, resource_group,
                          secret_name, tf_vars) -> Dict[str, Any]:
        endpoint = self._synthetic_endpoint(session_id, model_name)
        # Store a synthetic key so the read path is exercised offline too.
        self.sdk.set_secret(secret_name, f"dryrun-key-{session_id[:8]}")
        ops = [{"action": "plan", "resource": "foundry_module", "dryrun": True,
                "vars": tf_vars},
               {"action": "apply", "resource": "foundry_module", "dryrun": True}]
        return {"endpoint_url": endpoint, "provisioner": "dryrun", "operations": ops}

    def _provision_terraform(self, session_id, model_name, resource_group,
                             secret_name, tf_vars) -> Dict[str, Any]:
        # Plan first (cost/scope preview), then apply. Never apply if plan fails.
        plan_rc = self._terraform("plan", tf_vars)
        if plan_rc != 0:
            raise ProvisioningError(
                "terraform plan failed", details={"returncode": plan_rc})
        apply_rc = self._terraform("apply", tf_vars)
        if apply_rc != 0:
            raise ProvisioningError(
                "terraform apply failed", details={"returncode": apply_rc})
        endpoint = self._read_tf_output("foundry_endpoint_url") or (
            self._synthetic_endpoint(session_id, model_name))
        ops = [{"action": "plan", "resource": "foundry_module", "returncode": plan_rc},
               {"action": "apply", "resource": "foundry_module", "returncode": apply_rc}]
        return {"endpoint_url": endpoint, "provisioner": "terraform", "operations": ops}

    def _provision_sdk(self, session_id, model_name, resource_group,
                       secret_name, tf_vars) -> Dict[str, Any]:
        rg = self.sdk.ensure_resource_group(resource_group, self.location)
        # A real integration would create the Foundry deployment here and read
        # back its scoring endpoint + key; we synthesize the endpoint and stash
        # a placeholder key in Key Vault so downstream code is identical.
        endpoint = self._synthetic_endpoint(session_id, model_name)
        self.sdk.set_secret(secret_name, f"sdk-key-{session_id[:8]}")
        ops = [{"action": "ensure", "resource": "resource_group", **rg}]
        return {"endpoint_url": endpoint, "provisioner": "sdk", "operations": ops}

    # ----- terraform plumbing (mirrors core/azure/terraform.py) ----------
    def _tf_vars(self, **kwargs: Any) -> Dict[str, str]:
        model_name = kwargs["model_name"]
        v = {
            "subscription_id": self.subscription_id or "",
            "location": self.location,
            "resource_group_name": kwargs["resource_group"],
            "session_name": kwargs["session_name"],
            "foundry_model_sku": model_name,
            "managed_identity_name": kwargs["identity_name"],
            "key_vault_secret_name": kwargs["secret_name"],
            "private_networking": "true" if kwargs["private_networking"] else "false",
            "ttl_minutes": str(kwargs["ttl_minutes"]),
        }
        return v

    def _terraform(self, action: str, tf_vars: Dict[str, str]) -> int:
        if not shutil.which("terraform"):
            logger.warning("terraform binary absent; treating %s as no-op", action)
            return 0
        var_args: List[str] = []
        for key, val in tf_vars.items():
            var_args += ["-var", f"{key}={val}"]
        subprocess.run(["terraform", "init", "-input=false"],
                       cwd=self.terraform_dir, check=True)
        extra = ["-auto-approve"] if action in ("apply", "destroy") else []
        proc = subprocess.run(
            ["terraform", action, "-input=false", *extra, *var_args],
            cwd=self.terraform_dir, check=False,
        )
        return proc.returncode

    def _read_tf_output(self, name: str) -> Optional[str]:
        if not shutil.which("terraform"):
            return None
        try:
            proc = subprocess.run(
                ["terraform", "output", "-raw", name],
                cwd=self.terraform_dir, check=False, capture_output=True, text=True,
            )
            return proc.stdout.strip() if proc.returncode == 0 else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("terraform output %s failed: %s", name, exc)
            return None

    # ================= lookup / usage ===================================
    def get_session(self, session_id: str) -> FoundrySession:
        session = self.store.get(session_id)
        if session is None:
            raise SessionNotFoundError(
                f"session '{session_id}' not found", details={"session_id": session_id})
        return session

    def heartbeat(self, session_id: str) -> FoundrySession:
        """Refresh a session's inactivity clock."""
        session = self.get_session(session_id)
        session.last_activity_at = _utcnow()
        self.store.put(session)
        return session

    def record_usage(
        self, session_id: str, *, tokens: int = 0, latency_ms: int = 0,
    ) -> FoundrySession:
        """Accrue usage/cost after an inference call and enforce hard stop.

        Returns the updated session. If the hard cost cap is breached the
        session is torn down and marked ``over_budget`` before returning.
        """
        session = self.get_session(session_id)
        rate_per_ms = sku_hourly_rate(session.model_name) / 3_600_000.0
        call_cost = rate_per_ms * max(latency_ms, 0)
        call_cost += (tokens / 1000.0) * TOKEN_COST_PER_1K_USD
        session.usage.calls += 1
        session.usage.tokens += max(tokens, 0)
        session.usage.estimated_cost_usd = round(
            session.usage.estimated_cost_usd + call_cost, 6)
        session.last_activity_at = _utcnow()
        self.store.put(session)

        if session.over_hard_limit():
            logger.warning("session %s breached hard cost cap ($%.4f >= $%.4f); "
                           "tearing down", session_id,
                           session.usage.estimated_cost_usd, session.max_cost_usd)
            self.teardown_session(session_id, reason="cost_hard_stop")
            session.status = "over_budget"
        return session

    def check_can_infer(self, session_id: str, *, force: bool = False) -> FoundrySession:
        """Guard called before inference: TTL, status and soft cost stop.

        Raises ``CostLimitExceeded`` on soft-stop unless ``force=True``.
        """
        session = self.get_session(session_id)
        if session.is_expired():
            self.teardown_session(session_id, reason="ttl_expired")
            raise SessionNotFoundError(
                f"session '{session_id}' has expired",
                details={"session_id": session_id})
        if session.over_soft_limit() and not force:
            raise CostLimitExceeded(
                "session near cost cap; pass force=true to continue",
                details={
                    "estimated_cost": session.usage.estimated_cost_usd,
                    "soft_limit": session.soft_limit_usd(),
                    "max_cost_usd": session.max_cost_usd,
                },
            )
        return session

    # ================= teardown / sweeping ==============================
    def teardown_session(self, session_id: str, *, reason: str = "manual") -> Dict[str, Any]:
        """Idempotently release a session's resources.

        Repeated calls (or calls for unknown ids) succeed and return a record —
        never raise — as required by the contract.
        """
        with self._lock:
            session = self.store.pop(session_id)
        if session is None:
            return {"session_id": session_id, "status": "torn_down",
                    "already_gone": True, "reason": reason}
        ops: List[Dict[str, Any]] = []
        try:
            if not self.dry_run and self.use_sdk_provision:
                ops.append(self.sdk.delete_resource_group(session.resource_group))
            elif not self.dry_run:
                rc = self._terraform("destroy", self._tf_vars(
                    resource_group=session.resource_group,
                    session_name=_slug(session_id[:12]),
                    model_name=session.model_name,
                    identity_name=session.managed_identity_name,
                    secret_name=session.key_vault_secret_name,
                    private_networking=session.private_networking,
                    ttl_minutes=session.max_duration_minutes, tags=session.tags,
                ))
                ops.append({"action": "destroy", "returncode": rc})
            else:
                ops.append({"action": "destroy", "dryrun": True})
            # Always scrub the secret (best-effort, idempotent).
            self.sdk.delete_secret(session.key_vault_secret_name)
        except TeardownError:
            # Re-insert so a retry can complete the teardown.
            self.store.put(session)
            raise
        session.status = "torn_down"
        logger.info("torn down foundry session %s (reason=%s)", session_id, reason)
        return {"session_id": session_id, "status": "torn_down",
                "reason": reason, "operations": ops}

    def sweep(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Tear down every session past TTL, inactivity or hard cost limit.

        Intended to be called periodically (cron / Celery beat). Returns the
        teardown records for whatever it reaped.
        """
        now = now or _utcnow()
        reaped: List[Dict[str, Any]] = []
        for session in self.store.all():
            reason = None
            if session.is_expired(now):
                reason = "ttl_expired"
            elif session.over_hard_limit():
                reason = "cost_hard_stop"
            elif session.is_inactive(now):
                reason = "inactivity_timeout"
            if reason:
                reaped.append(self.teardown_session(session.session_id, reason=reason))
        return reaped

    def _best_effort_cleanup(self, resource_group: str, secret_name: str) -> None:
        try:
            if not self.dry_run and self.use_sdk_provision:
                self.sdk.delete_resource_group(resource_group)
            self.sdk.delete_secret(secret_name)
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask original error
            logger.warning("best-effort cleanup after failed provision: %s", exc)


# --- process-wide singleton (mirrors core.azure's module-level registry) ----
_MANAGER: Optional[AzureFoundrySessionManager] = None
_MANAGER_LOCK = threading.RLock()


def get_session_manager() -> AzureFoundrySessionManager:
    """Return the shared session manager, building it on first use."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = AzureFoundrySessionManager()
        return _MANAGER


def reset_session_manager(manager: Optional[AzureFoundrySessionManager] = None) -> None:
    """Replace the singleton (used by tests for isolation)."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = manager


# ============================== CLI ========================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dvsa_api.reasoning.azure_session_manager",
        description="Create/inspect ephemeral Azure Foundry reasoning sessions.",
    )
    p.add_argument("--model", default="o1-mini", help="Foundry model SKU to run.")
    p.add_argument("--user", default=None, help="Owning user id (for quotas).")
    p.add_argument("--minutes", type=int, default=DEFAULT_DURATION_MINUTES,
                   help="Session TTL in minutes.")
    p.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST_USD,
                   help="Soft cost cap in USD.")
    p.add_argument("--private-networking", action="store_true",
                   help="Provision inside a VNet.")
    p.add_argument("--dry-run", action="store_true",
                   help="Never touch the cloud; record intended operations only.")
    p.add_argument("--teardown", metavar="SESSION_ID", default=None,
                   help="Tear down an existing session id instead of creating one.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    manager = AzureFoundrySessionManager(dry_run=True if args.dry_run else None)

    if args.teardown:
        result = manager.teardown_session(args.teardown, reason="cli")
        print(json.dumps(result, indent=2))
        return 0

    session = manager.create_session(
        model_name=args.model, user_id=args.user,
        max_duration_minutes=args.minutes, max_cost_usd=args.max_cost,
        private_networking=args.private_networking,
    )
    print(json.dumps(session.to_public_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
