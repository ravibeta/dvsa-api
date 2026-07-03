"""The reasoning-model adapter contract shared by every provider.

A reasoning model — local shim, Azure Foundry, OpenAI, etc. — is exposed to the
rest of dvsa-api through a single, minimal interface:

* :meth:`ReasoningModelAdapter.predict` — run inference over a ``context`` dict
  and return a structured ``{actions, reasoning_trace, metadata}`` response.
* :meth:`ReasoningModelAdapter.health_check` — report liveness.

The concrete :class:`~dvsa_api.reasoning.azure_foundry_adapter.AzureFoundryAdapter`
implements this contract on top of ephemeral Azure Foundry deployments.

The helpers here (:func:`new_request_id`, :func:`build_metadata`,
:func:`ensure_response_shape`) let adapters emit contract-compliant responses
without re-deriving the schema each time.
"""

from __future__ import annotations

import abc
import time
import uuid
from typing import Any, Dict, List, Optional

# Keys every ``predict`` response must contain.
RESPONSE_KEYS = ("actions", "reasoning_trace", "metadata")


class ReasoningModelAdapter(abc.ABC):
    """Abstract runtime interface all reasoning adapters implement."""

    #: Human-readable adapter/provider name (overridden by subclasses).
    name: str = "reasoning-adapter"

    @abc.abstractmethod
    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run one reasoning request.

        Parameters
        ----------
        context:
            Task payload. Recognised keys include ``frames``, ``tracks``,
            ``sensor_meta``, ``precomputed_features`` and ``query``; adapters
            must tolerate extra/missing keys.

        Returns
        -------
        dict
            JSON-serializable ``{actions, reasoning_trace, metadata}``.
        """

    @abc.abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return ``{"status": "ok"}`` or ``{"status": "error", ...}``."""

    # ``call_model``-style convenience so callers need not know the schema.
    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict(context)


def new_request_id() -> str:
    """Return a short, unique request id for telemetry correlation."""
    return uuid.uuid4().hex


def now_ms() -> int:
    """Monotonic-ish millisecond timestamp for latency math."""
    return int(time.time() * 1000)


def build_metadata(
    *,
    model: str,
    version: str,
    latency_ms: int,
    tokens: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Assemble the ``metadata`` block required by the response contract."""
    meta: Dict[str, Any] = {
        "model": model,
        "version": version,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }
    meta.update(extra)
    return meta


def ensure_response_shape(response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate/normalise an adapter response to the contract shape.

    Missing ``actions``/``reasoning_trace`` default to empty lists; ``metadata``
    must be present (adapters build it via :func:`build_metadata`). Raises
    ``ValueError`` when the response is not a dict or lacks ``metadata`` so
    contract violations fail loudly in tests rather than at the client.
    """
    if not isinstance(response, dict):
        raise ValueError("adapter response must be a dict")
    actions = response.get("actions", [])
    trace = response.get("reasoning_trace", [])
    if "metadata" not in response:
        raise ValueError("adapter response missing 'metadata'")
    if not isinstance(actions, list):
        raise ValueError("'actions' must be a list")
    if not isinstance(trace, list):
        raise ValueError("'reasoning_trace' must be a list")
    normalised = dict(response)
    normalised["actions"] = actions
    normalised["reasoning_trace"] = trace
    return normalised


def validate_adapter(obj: Any) -> ReasoningModelAdapter:
    """Duck-type check that ``obj`` satisfies the adapter contract.

    Used by dynamic loaders: a foreign ``adapter.py`` need not subclass
    :class:`ReasoningModelAdapter` as long as it provides callable ``predict``
    and ``health_check`` methods.
    """
    for method in ("predict", "health_check"):
        if not callable(getattr(obj, method, None)):
            raise TypeError(
                f"adapter {obj!r} does not implement required method '{method}'"
            )
    return obj  # type: ignore[return-value]


def error_action(message: str, *, code: str = "error") -> List[Dict[str, Any]]:
    """Return a one-element ``actions`` list describing a soft failure."""
    return [{"type": "error", "code": code, "message": message}]
