"""Configuration for the DVSA Databricks connector.

The connector needs two things to reach DVSA-APIs: an **endpoint** (for remote
mode) and an **API key**. Everything else has a safe default. Configuration can
come from three places, in ascending order of preference:

1. explicit keyword arguments to :class:`DVSAConfig`,
2. environment variables (:func:`config_from_env`),
3. Databricks Secrets (:func:`config_from_secrets`) — the recommended path on a
   real cluster because it keeps the key out of notebooks, logs and MLflow.

This module is deliberately dependency-free (stdlib only) so it can be imported
and unit-tested without ``pyspark``/``mlflow`` installed, exactly like the
routine and schema layers in the core repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Environment-variable names the connector understands. Kept as constants so the
# notebooks/docs and the code never drift.
ENV_ENDPOINT = "DVSA_ENDPOINT"
ENV_API_KEY = "DVSA_API_KEY"
ENV_MODE = "DVSA_MODE"
ENV_INFER_PATH = "DVSA_INFER_PATH"
ENV_TIMEOUT = "DVSA_TIMEOUT"
ENV_MAX_RETRIES = "DVSA_MAX_RETRIES"
ENV_BACKOFF = "DVSA_BACKOFF_FACTOR"
ENV_VERIFY_SSL = "DVSA_VERIFY_SSL"

# The two supported deployment modes (see docs/databricks_integration.md).
MODE_REMOTE = "remote"          # call a hosted DVSA-API endpoint
MODE_IN_CLUSTER = "in_cluster"  # run a DVSA adapter installed on the cluster
VALID_MODES = (MODE_REMOTE, MODE_IN_CLUSTER)


def _as_bool(value: Any, default: bool = True) -> bool:
    """Parse a truthy/falsey string ("1", "true", "no", ...) into a bool."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class DVSAConfig:
    """Validated connection settings for DVSA-APIs.

    Attributes
    ----------
    endpoint:
        Base URL of the hosted DVSA-API (remote mode), e.g.
        ``https://dvsa.example.com/api/v1``. Ignored in in-cluster mode.
    api_key:
        Bearer token / API key. Read from Databricks Secrets in production.
    mode:
        ``"remote"`` or ``"in_cluster"``.
    infer_path:
        Path appended to ``endpoint`` for the inference call. Defaults to
        ``/analytics/infer`` (mirrors the ``/api/v1/analytics/`` namespace).
    timeout:
        Per-request timeout in seconds (remote mode).
    max_retries / backoff_factor:
        Retry policy for transient inference failures (429/5xx/timeouts).
    verify_ssl:
        Whether to verify TLS certificates (leave ``True`` in production).
    extra:
        Free-form extra settings preserved from secrets/env (e.g. catalog,
        schema names) so callers can extend without a schema change.
    """

    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    mode: str = MODE_REMOTE
    infer_path: str = "/analytics/infer"
    timeout: float = 60.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    verify_ssl: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unknown mode {self.mode!r}; expected one of {VALID_MODES}"
            )
        self.timeout = float(self.timeout)
        self.max_retries = int(self.max_retries)
        self.backoff_factor = float(self.backoff_factor)
        self.verify_ssl = bool(self.verify_ssl)

    # ------------------------------------------------------------------ #
    def validate(self) -> "DVSAConfig":
        """Raise ``ValueError`` if the config cannot be used for inference.

        Remote mode needs an ``endpoint`` (and, in practice, an ``api_key``);
        in-cluster mode needs neither because inference runs locally.
        """

        if self.mode == MODE_REMOTE:
            if not self.endpoint:
                raise ValueError(
                    "Remote mode requires an endpoint. Set the "
                    f"{ENV_ENDPOINT!r} secret/env var or pass endpoint=."
                )
            if not self.api_key:
                # A missing key is usually a misconfiguration; fail loudly so it
                # is caught at setup rather than as a 401 mid-pipeline.
                raise ValueError(
                    "Remote mode requires an api_key. Set the "
                    f"{ENV_API_KEY!r} secret/env var or pass api_key=."
                )
        return self

    def infer_url(self) -> str:
        """Full inference URL for remote mode (``endpoint`` + ``infer_path``)."""

        if not self.endpoint:
            raise ValueError("endpoint is not set")
        return self.endpoint.rstrip("/") + "/" + self.infer_path.lstrip("/")

    def redacted(self) -> Dict[str, Any]:
        """Config as a dict with the api_key masked — safe to log / print."""

        return {
            "endpoint": self.endpoint,
            "api_key": "***" if self.api_key else None,
            "mode": self.mode,
            "infer_path": self.infer_path,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
            "verify_ssl": self.verify_ssl,
            "extra": self.extra,
        }


def config_from_env(env: Optional[Dict[str, str]] = None) -> DVSAConfig:
    """Build a :class:`DVSAConfig` from environment variables.

    Useful for local development and for CI, where Databricks ``dbutils`` is not
    available. ``env`` defaults to :data:`os.environ` but can be injected in
    tests.
    """

    env = dict(os.environ if env is None else env)
    kwargs: Dict[str, Any] = {
        "endpoint": env.get(ENV_ENDPOINT),
        "api_key": env.get(ENV_API_KEY),
        "mode": env.get(ENV_MODE, MODE_REMOTE),
        "infer_path": env.get(ENV_INFER_PATH, "/analytics/infer"),
    }
    if ENV_TIMEOUT in env:
        kwargs["timeout"] = float(env[ENV_TIMEOUT])
    if ENV_MAX_RETRIES in env:
        kwargs["max_retries"] = int(env[ENV_MAX_RETRIES])
    if ENV_BACKOFF in env:
        kwargs["backoff_factor"] = float(env[ENV_BACKOFF])
    if ENV_VERIFY_SSL in env:
        kwargs["verify_ssl"] = _as_bool(env[ENV_VERIFY_SSL])
    return DVSAConfig(**kwargs)


def config_from_secrets(
    dbutils: Any = None,
    *,
    scope: str = "dvsa",
    endpoint_key: str = "DVSA_ENDPOINT",
    api_key_key: str = "DVSA_API_KEY",
    mode_key: str = "DVSA_MODE",
    fallback_env: bool = True,
) -> DVSAConfig:
    """Read DVSA settings from Databricks Secrets and return a validated config.

    On a Databricks cluster ``dbutils`` is injected into the notebook globals;
    pass it in explicitly so this function stays importable/testable off-cluster.

    Parameters
    ----------
    dbutils:
        The Databricks ``dbutils`` object. If ``None`` and ``fallback_env`` is
        set, falls back to :func:`config_from_env` (handy for local dev/CI).
    scope:
        Databricks Secret scope holding the DVSA keys.
    endpoint_key / api_key_key / mode_key:
        Secret names within ``scope``.
    fallback_env:
        When ``dbutils`` is ``None``, read from environment variables instead of
        raising — so the same notebook cell works locally and on-cluster.

    Notes
    -----
    Values fetched via ``dbutils.secrets.get`` are automatically redacted by
    Databricks in notebook output; we never echo them here either.
    """

    if dbutils is None:
        if fallback_env:
            return config_from_env().validate()
        raise ValueError(
            "dbutils is None and fallback_env=False; cannot read secrets."
        )

    def _get(key: str, default: Optional[str] = None) -> Optional[str]:
        try:
            return dbutils.secrets.get(scope=scope, key=key)
        except Exception:  # noqa: BLE001 - secret may be optional (e.g. mode)
            return default

    mode = _get(mode_key, MODE_REMOTE) or MODE_REMOTE
    cfg = DVSAConfig(
        endpoint=_get(endpoint_key),
        api_key=_get(api_key_key),
        mode=mode,
    )
    return cfg.validate()
