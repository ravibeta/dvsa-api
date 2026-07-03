"""A tiny in-process registry mapping model names to adapter instances.

This is the seam the wider dvsa-api reasoning layer uses: ``get_model(name)``
returns a live :class:`~dvsa_api.reasoning.adapter_base.ReasoningModelAdapter`
and ``call_model(name, context)`` runs it. The Azure Foundry adapter registers
itself here so it participates in the same routing/fallback machinery as any
"bring-your-own" local model.

The registry is deliberately minimal and dependency-free; a richer folder-based
discovery layer can populate it without changing this contract.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List

from .adapter_base import ReasoningModelAdapter, validate_adapter
from .errors import ModelUnavailableError

# name -> factory callable returning an adapter (lazy so remote/ephemeral
# providers are only constructed on demand).
_FACTORIES: Dict[str, Callable[[], ReasoningModelAdapter]] = {}
_INSTANCES: Dict[str, ReasoningModelAdapter] = {}
_LOCK = threading.RLock()


def register(name: str, factory: Callable[[], ReasoningModelAdapter]) -> None:
    """Register a lazy adapter ``factory`` under ``name`` (idempotent)."""
    with _LOCK:
        _FACTORIES[name] = factory
        _INSTANCES.pop(name, None)  # drop any stale cached instance


def register_instance(name: str, adapter: ReasoningModelAdapter) -> None:
    """Register an already-constructed adapter instance."""
    validate_adapter(adapter)
    with _LOCK:
        _INSTANCES[name] = adapter
        _FACTORIES.setdefault(name, lambda: adapter)


def get_model(name: str) -> ReasoningModelAdapter:
    """Return the adapter registered as ``name`` (constructing it lazily)."""
    with _LOCK:
        if name in _INSTANCES:
            return _INSTANCES[name]
        factory = _FACTORIES.get(name)
        if factory is None:
            raise ModelUnavailableError(
                f"no reasoning model registered as '{name}'",
                details={"available": list(_FACTORIES)},
            )
        adapter = validate_adapter(factory())
        _INSTANCES[name] = adapter
        return adapter


def list_models() -> List[str]:
    """Return the names of all registered models."""
    with _LOCK:
        return sorted(_FACTORIES)


def call_model(name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience: run ``get_model(name).predict(context)``."""
    return get_model(name).predict(context)


def clear() -> None:
    """Drop all registrations (used by tests for isolation)."""
    with _LOCK:
        _FACTORIES.clear()
        _INSTANCES.clear()
