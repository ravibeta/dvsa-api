"""In-process registry + folder discovery + policy routing for reasoning models.

Responsibilities
----------------
* **Registry** — map a model ``name`` to a lazy adapter factory
  (:func:`register`, :func:`get_model`, :func:`list_models`, :func:`call_model`).
  The Azure Foundry provider registers itself here so it shares this machinery.
* **Discovery** — scan ``custom_models/reasoning/*`` for folders containing a
  ``manifest.json`` and register each as a lazy factory (local adapters are
  imported on first use; Azure/remote adapters wrap an endpoint). Reasoning
  models live nested under ``custom_models/`` as a clearly-separated special
  case of the wider custom-model machinery. See :func:`discover_models` /
  :func:`ensure_discovered`.
* **Policy routing** — :func:`select_model` picks a model name for a request from
  ``by_name`` / ``cost_optimized`` / ``latency_optimized`` / ``privacy_first``
  policies using manifest metadata and environment variables.

Everything is additive and dependency-free; dropping a new model folder needs no
code changes elsewhere.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapter_base import ReasoningModelAdapter, validate_adapter
from .errors import ModelUnavailableError

# name -> factory callable returning an adapter (lazy so remote/ephemeral
# providers are only constructed on demand).
_FACTORIES: Dict[str, Callable[[], ReasoningModelAdapter]] = {}
_INSTANCES: Dict[str, ReasoningModelAdapter] = {}
_MANIFESTS: Dict[str, "ModelManifest"] = {}
_LOCK = threading.RLock()
_DISCOVERED = False

# Default location scanned for model folders: <repo-root>/custom_models/reasoning.
# Reasoning models are nested under custom_models/ (their own "reasoning/"
# subtree), separate from the vision/detection custom models.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODELS_ROOT = os.environ.get(
    "REASONING_MODELS_ROOT", os.path.join(_REPO_ROOT, "custom_models", "reasoning"))

# Heuristic hints when a manifest does not specify them explicitly.
_LOCAL_LATENCY_HINT_MS = 50
_REMOTE_LATENCY_HINT_MS = 800
_LOCAL_COST_HINT = 0.0
_REMOTE_COST_HINT = 0.02


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelManifest:
    """Parsed ``manifest.json`` describing a discovered model folder."""

    name: str
    version: str
    type: str  # "local" | "remote"
    entrypoint: str
    model_dir: str
    capabilities: List[str] = field(default_factory=list)
    provider: Optional[str] = None
    cost_hint: float = _LOCAL_COST_HINT
    latency_hint_ms: int = _LOCAL_LATENCY_HINT_MS
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.type.lower() == "local"

    @property
    def is_azure(self) -> bool:
        prov = (self.provider or "").lower()
        return prov in ("azure", "azure_foundry", "openai") or "azure" in self.name.lower()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], model_dir: str) -> "ModelManifest":
        name = data.get("name")
        if not name:
            raise ModelUnavailableError(
                f"manifest in {model_dir} missing required 'name'")
        mtype = (data.get("type") or "local").lower()
        is_local = mtype == "local"
        return cls(
            name=name,
            version=str(data.get("version", "0.0.0")),
            type=mtype,
            entrypoint=data.get("entrypoint", "adapter.py"),
            model_dir=model_dir,
            capabilities=list(data.get("capabilities") or []),
            provider=data.get("provider"),
            cost_hint=float(data.get(
                "cost_hint",
                _LOCAL_COST_HINT if is_local else _REMOTE_COST_HINT)),
            latency_hint_ms=int(data.get(
                "latency_hint_ms",
                _LOCAL_LATENCY_HINT_MS if is_local else _REMOTE_LATENCY_HINT_MS)),
            raw=data,
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
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
            # A folder may have been dropped in after first discovery.
            ensure_discovered()
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
    """Return the names of all registered models (triggers discovery once)."""
    ensure_discovered()
    with _LOCK:
        return sorted(_FACTORIES)


def get_manifest(name: str) -> Optional[ModelManifest]:
    """Return the discovered manifest for ``name`` (or ``None`` if code-registered)."""
    ensure_discovered()
    with _LOCK:
        return _MANIFESTS.get(name)


def list_manifests() -> List[ModelManifest]:
    ensure_discovered()
    with _LOCK:
        return list(_MANIFESTS.values())


def call_model(name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience: run ``get_model(name).predict(context)``."""
    return get_model(name).predict(context)


def clear() -> None:
    """Drop all registrations + discovery state (used by tests for isolation)."""
    global _DISCOVERED
    with _LOCK:
        _FACTORIES.clear()
        _INSTANCES.clear()
        _MANIFESTS.clear()
        _DISCOVERED = False


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _make_factory(manifest: ModelManifest) -> Callable[[], ReasoningModelAdapter]:
    """Build the lazy factory that instantiates a model's adapter."""

    def factory() -> ReasoningModelAdapter:
        # Azure/remote-by-provider models use the generic env-configured adapter
        # unless the folder ships its own adapter.py wrapper.
        adapter_file = os.path.join(manifest.model_dir, manifest.entrypoint)
        if manifest.is_azure and not os.path.isfile(adapter_file):
            from .azure_adapter import AzureReasoningAdapter  # noqa: PLC0415
            return AzureReasoningAdapter(
                model_name=manifest.name, config=manifest.raw.get("config"))
        from .local_adapter_loader import load_local_adapter  # noqa: PLC0415
        return load_local_adapter(manifest.model_dir, manifest.raw)

    return factory


def discover_models(root: Optional[str] = None, *, force: bool = False) -> List[str]:
    """Scan ``root`` for model folders and register each; return their names.

    A model folder is any immediate subdirectory containing ``manifest.json``.
    Folder names beginning with ``.`` are ignored. Re-registration is idempotent;
    pass ``force=True`` to rescan after adding folders at runtime.
    """
    global _DISCOVERED
    root = root or DEFAULT_MODELS_ROOT
    discovered: List[str] = []
    if not os.path.isdir(root):
        with _LOCK:
            _DISCOVERED = True
        return discovered

    for entry in sorted(os.listdir(root)):
        if entry.startswith("."):
            continue
        model_dir = os.path.join(root, entry)
        manifest_path = os.path.join(model_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with _LOCK:
            if entry in _MANIFESTS and not force:
                discovered.append(entry)
                continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            manifest = ModelManifest.from_dict(data, model_dir)
        except (ValueError, OSError, ModelUnavailableError):
            # A malformed manifest disables just that model, not discovery.
            continue
        with _LOCK:
            _MANIFESTS[manifest.name] = manifest
            _FACTORIES[manifest.name] = _make_factory(manifest)
            _INSTANCES.pop(manifest.name, None)
        discovered.append(manifest.name)

    with _LOCK:
        _DISCOVERED = True
    return discovered


def ensure_discovered() -> None:
    """Run :func:`discover_models` once (thread-safe, idempotent)."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    with _LOCK:
        if _DISCOVERED:
            return
    discover_models()


# --------------------------------------------------------------------------- #
# Policy routing
# --------------------------------------------------------------------------- #
VALID_POLICIES = ("by_name", "cost_optimized", "latency_optimized", "privacy_first")


def select_model(policy: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Return the name of the model chosen for ``context`` under ``policy``.

    * ``by_name`` — use ``context['model']`` / ``context['model_name']`` or the
      ``AZURE_REASONING_DEFAULT`` env var.
    * ``cost_optimized`` — lowest ``cost_hint`` (local models are free).
    * ``latency_optimized`` — lowest ``latency_hint_ms`` (local models are fast).
    * ``privacy_first`` — prefer ``type: local`` models; never route off-box if a
      local model exists.

    Raises :class:`ModelUnavailableError` when no candidate satisfies the policy.
    """
    context = context or {}
    ensure_discovered()

    if policy == "by_name":
        name = (context.get("model") or context.get("model_name")
                or os.environ.get("AZURE_REASONING_DEFAULT"))
        if not name:
            raise ModelUnavailableError(
                "by_name policy requires a model name in context or "
                "AZURE_REASONING_DEFAULT")
        with _LOCK:
            if name not in _FACTORIES:
                ensure_discovered()
            if name not in _FACTORIES:
                raise ModelUnavailableError(
                    f"model '{name}' not found", details={"available": list(_FACTORIES)})
        return name

    manifests = list_manifests()
    if not manifests:
        raise ModelUnavailableError(
            "no discovered models to select from", details={"policy": policy})

    if policy == "privacy_first":
        locals_ = [m for m in manifests if m.is_local]
        pool = locals_ or manifests
        return min(pool, key=lambda m: m.latency_hint_ms).name
    if policy == "cost_optimized":
        return min(manifests, key=lambda m: (m.cost_hint, m.latency_hint_ms)).name
    if policy == "latency_optimized":
        return min(manifests, key=lambda m: (m.latency_hint_ms, m.cost_hint)).name

    raise ModelUnavailableError(
        f"unknown selection policy '{policy}'",
        details={"valid": list(VALID_POLICIES)})
