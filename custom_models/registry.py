"""Adapter registry + the single ``get_detector`` entry point.

Adapters register themselves by *format* name with the :func:`register`
decorator (see ``custom_models/adapters/*.py``). Callers then build a ready
detector for any model via :func:`get_detector`, passing either a
:class:`~custom_models.loader.ModelSpec` or a catalog id string.

    from custom_models import get_detector
    detector = get_detector(spec).load()        # spec: ModelSpec
    detector = get_detector("visdrone-yolov8x")  # resolved from the catalog
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type, Union

from .loader import ModelSpec

logger = logging.getLogger(__name__)

# format name -> adapter class
_ADAPTERS: Dict[str, Type] = {}


def register(format_name: str):
    """Class decorator registering an adapter under ``format_name``.

    Raises
    ------
    ValueError
        If ``format_name`` is already registered to a different class.
    """

    def _wrap(cls: Type) -> Type:
        existing = _ADAPTERS.get(format_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Adapter format {format_name!r} already registered to {existing.__name__}"
            )
        _ADAPTERS[format_name] = cls
        logger.debug("Registered adapter %s for format %r", cls.__name__, format_name)
        return cls

    return _wrap


def get_adapter_class(format_name: str) -> Type:
    """Return the adapter class registered for ``format_name``."""

    _ensure_adapters_imported()
    try:
        return _ADAPTERS[format_name]
    except KeyError as exc:
        raise KeyError(
            f"No adapter registered for format {format_name!r}. "
            f"Available: {sorted(_ADAPTERS)}"
        ) from exc


def available_formats() -> List[str]:
    """Return the sorted list of registered adapter format names."""

    _ensure_adapters_imported()
    return sorted(_ADAPTERS)


def get_detector(
    model: Union[ModelSpec, str],
    *,
    catalog=None,
    **adapter_kwargs,
):
    """Build (but do not :meth:`load`) a detector for ``model``.

    Parameters
    ----------
    model:
        A :class:`ModelSpec`, or a catalog id ``str`` to resolve against
        ``catalog`` (a :class:`custom_models.selector.ModelSelector` or the path
        to a ``models_catalog.json``; defaults to the bundled catalog).
    catalog:
        Optional catalog source used only when ``model`` is a string.
    **adapter_kwargs:
        Forwarded to the adapter constructor (e.g. ``session_factory=...``,
        ``model_loader=...``, ``score_threshold=...``).

    Returns
    -------
    object
        An *unloaded* adapter exposing ``load()`` / ``infer(frame)`` / ``close()``.
    """

    spec = _resolve_spec(model, catalog)
    adapter_cls = get_adapter_class(spec.format)
    logger.info("Building %s for model %s (format=%s)", adapter_cls.__name__, spec.id, spec.format)
    return adapter_cls(spec, **adapter_kwargs)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _resolve_spec(model: Union[ModelSpec, str], catalog) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    if isinstance(model, str):
        from .selector import ModelSelector

        selector = (
            catalog
            if isinstance(catalog, ModelSelector)
            else ModelSelector.from_file(catalog) if catalog else ModelSelector.default()
        )
        return selector.get(model)
    raise TypeError(f"model must be a ModelSpec or catalog id str, got {type(model).__name__}")


def _ensure_adapters_imported() -> None:
    """Import the built-in adapters so their ``@register`` side effects run.

    Importing :mod:`custom_models.registry` alone does not pull in the adapter
    modules (that would create an import cycle), so we import them lazily the
    first time the registry is queried.
    """

    if _ADAPTERS:
        return
    from . import adapters  # noqa: F401  (registration side effects)
