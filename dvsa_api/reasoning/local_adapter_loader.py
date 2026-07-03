"""Safe dynamic loader for a model folder's ``adapter.py``.

A "bring-your-own" reasoning model is just a directory under
``custom_models/reasoning/<name>/`` containing a ``manifest.json`` and an
``adapter.py`` that exports a class named ``ReasoningModelAdapter`` implementing
the contract in :mod:`dvsa_api.reasoning.adapter_base`.

This module imports that file *by path* (without requiring the model folder to be
an installed package), instantiates the adapter with optional model-specific
config, and validates the contract before handing it back. Nothing here executes
model code at import time beyond constructing the adapter object.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import uuid
from typing import Any, Dict, Optional

from .adapter_base import ReasoningModelAdapter, validate_adapter
from .errors import ModelUnavailableError

# Name the loaded module must export.
ADAPTER_CLASS_NAME = "ReasoningModelAdapter"

_IMPORT_LOCK = threading.Lock()


def _load_config(model_dir: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``manifest['config']`` with an optional ``config.json`` sidecar."""
    config: Dict[str, Any] = dict(manifest.get("config") or {})
    sidecar = os.path.join(model_dir, "config.json")
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                config.update(json.load(fh) or {})
        except (ValueError, OSError) as exc:  # pragma: no cover - defensive
            raise ModelUnavailableError(
                f"invalid config.json in {model_dir}: {exc}") from exc
    return config


def _import_module_from_path(module_path: str):
    """Import a standalone ``.py`` file by path under a unique module name."""
    if not os.path.isfile(module_path):
        raise ModelUnavailableError(f"adapter entrypoint not found: {module_path}")
    # Unique synthetic module name so two models can both define
    # ``ReasoningModelAdapter`` without colliding in ``sys.modules``.
    mod_name = f"dvsa_reasoning_model_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ModelUnavailableError(f"cannot build import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    with _IMPORT_LOCK:
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - surface any adapter import error
            sys.modules.pop(mod_name, None)
            raise ModelUnavailableError(
                f"failed importing adapter {module_path}: {exc}") from exc
    return module


def load_local_adapter(
    model_dir: str,
    manifest: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> ReasoningModelAdapter:
    """Import ``entrypoint`` from ``model_dir`` and return a validated adapter.

    Parameters
    ----------
    model_dir:
        Absolute path to the model folder.
    manifest:
        Parsed ``manifest.json`` (uses ``entrypoint``, default ``adapter.py``).
    config:
        Optional pre-resolved config; when omitted it is read from the manifest
        and an optional ``config.json`` sidecar.
    """
    entrypoint = manifest.get("entrypoint", "adapter.py")
    module_path = os.path.join(model_dir, entrypoint)
    module = _import_module_from_path(module_path)

    adapter_cls = getattr(module, ADAPTER_CLASS_NAME, None)
    if adapter_cls is None or not isinstance(adapter_cls, type):
        raise ModelUnavailableError(
            f"{module_path} does not export a class named '{ADAPTER_CLASS_NAME}'")

    resolved = config if config is not None else _load_config(model_dir, manifest)
    # Adapters may accept a ``config`` kwarg or take no args; support both so the
    # contract stays minimal for simple models.
    try:
        instance = adapter_cls(config=resolved)
    except TypeError:
        instance = adapter_cls()

    return validate_adapter(instance)
