"""DVSA inference clients — remote HTTP and in-cluster adapter.

The connector talks to DVSA-APIs through a tiny :class:`DVSAClient` interface
with a single method, :meth:`DVSAClient.infer`, that takes a DVSA ``context``
dict (see :mod:`dvsa_databricks_connector.connector`) plus a ``model_name`` and
returns a JSON-serialisable result dict. Two implementations are provided:

* :class:`RemoteDVSAClient` — POSTs the context to a hosted DVSA-API endpoint
  with a bearer token, retry/backoff on transient failures. This is the default
  for most Databricks users (**remote DVSA mode**).
* :class:`InClusterDVSAClient` — calls a Python callable that runs DVSA
  inference *inside* the cluster (**in-cluster DVSA mode**), for private
  deployments where the DVSA adapter wheel is installed on the workers.

``requests`` is imported lazily inside :class:`RemoteDVSAClient` so this module
(and the in-cluster path) import cleanly in environments without it, and so unit
tests can exercise the mapping/batching logic with a stub client.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .config import DVSAConfig, MODE_IN_CLUSTER, MODE_REMOTE

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying: 429 (rate limit) + 5xx (transient server error).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class DVSAClient:
    """Interface every DVSA client implements.

    Subclasses override :meth:`infer_one`; :meth:`infer` / :meth:`infer_batch`
    provide batching on top so callers have a uniform surface.
    """

    def infer_one(self, context: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        raise NotImplementedError

    def infer(self, context: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Run inference for a single context dict."""

        return self.infer_one(context, model_name)

    def infer_batch(
        self, contexts: List[Dict[str, Any]], model_name: str
    ) -> List[Dict[str, Any]]:
        """Run inference for a list of contexts (default: sequential)."""

        return [self.infer_one(ctx, model_name) for ctx in contexts]

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release any held resources (HTTP session, model handle)."""


class RemoteDVSAClient(DVSAClient):
    """Call a hosted DVSA-API endpoint over HTTP with retry/backoff.

    The request body is ``{"model_name": ..., "context": {...}}`` and the
    response is expected to be a JSON object (returned verbatim as the result).
    """

    def __init__(self, config: DVSAConfig, session: Any = None) -> None:
        config.validate()
        self.config = config
        # Allow a caller/test to inject a pre-built requests.Session (or a
        # requests-mock backed one). Otherwise build lazily on first use.
        self._session = session

    def _get_session(self) -> Any:
        if self._session is None:
            import requests  # lazy: keep module importable without requests

            self._session = requests.Session()
        return self._session

    def infer_one(self, context: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        import requests  # lazy import (see module docstring)

        url = self.config.infer_url()
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {"model_name": model_name, "context": context}

        session = self._get_session()
        last_exc: Optional[Exception] = None
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                resp = session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl,
                )
                if resp.status_code in _RETRYABLE_STATUS:
                    raise _RetryableStatus(resp.status_code)
                resp.raise_for_status()
                return resp.json()
            except (_RetryableStatus, requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                sleep = self.config.backoff_factor * (2 ** (attempt - 1))
                # Never log the api_key; only the URL and the retry reason.
                logger.warning(
                    "DVSA inference retry %d/%d after %s (sleeping %.2fs): %s",
                    attempt, attempts - 1, url, sleep, exc,
                )
                time.sleep(sleep)
        raise RuntimeError(
            f"DVSA inference failed after {attempts} attempt(s): {last_exc}"
        ) from last_exc

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._session = None


class _RetryableStatus(Exception):
    """Internal marker so retryable HTTP statuses share the retry path."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"retryable HTTP status {status_code}")
        self.status_code = status_code


# Type alias for an in-cluster inference callable: (context, model_name) -> dict
InferFn = Callable[[Dict[str, Any], str], Dict[str, Any]]


class InClusterDVSAClient(DVSAClient):
    """Run DVSA inference inside the cluster via an injected callable.

    ``infer_fn`` is any callable ``(context, model_name) -> dict``. This lets a
    private deployment wire the DVSA adapter (e.g. one built on
    ``custom_models.ModelSelector``) without the connector taking a hard
    dependency on it. If ``infer_fn`` is omitted, :meth:`infer_one` raises a
    clear error telling the operator to install/provide an adapter.
    """

    def __init__(self, infer_fn: Optional[InferFn] = None) -> None:
        self._infer_fn = infer_fn

    def infer_one(self, context: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        if self._infer_fn is None:
            raise RuntimeError(
                "In-cluster mode requires a DVSA adapter. Pass infer_fn=... to "
                "InClusterDVSAClient (or build_client(config, infer_fn=...)) "
                "with a callable(context, model_name) -> dict."
            )
        result = self._infer_fn(context, model_name)
        if not isinstance(result, dict):
            raise TypeError(
                f"In-cluster infer_fn must return a dict, got {type(result)!r}"
            )
        return result


def build_client(
    config: DVSAConfig,
    *,
    infer_fn: Optional[InferFn] = None,
    session: Any = None,
) -> DVSAClient:
    """Factory: return the right :class:`DVSAClient` for ``config.mode``.

    Parameters
    ----------
    config:
        Validated :class:`~dvsa_databricks_connector.config.DVSAConfig`.
    infer_fn:
        In-cluster inference callable (only used in in-cluster mode).
    session:
        Optional pre-built HTTP session (only used in remote mode; handy for
        tests using ``requests-mock``).
    """

    if config.mode == MODE_REMOTE:
        return RemoteDVSAClient(config, session=session)
    if config.mode == MODE_IN_CLUSTER:
        return InClusterDVSAClient(infer_fn=infer_fn)
    raise ValueError(f"Unsupported mode: {config.mode!r}")  # pragma: no cover
