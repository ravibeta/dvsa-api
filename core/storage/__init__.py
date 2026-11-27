"""Cloud-agnostic object storage (see :mod:`core.storage.base`).

``get_object_store()`` returns the implementation selected by ``DVSA_OBJECT_STORE``
(default ``"azure"``). Only Azure exists today; an ``"aws"`` S3 store can be registered in
``_PROVIDERS`` without changing any feature code.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from .base import ObjectCreated, ObjectInfo, ObjectStore, StorageEventError, UploadTarget


def _azure() -> ObjectStore:
    from .azure_blob import AzureBlobObjectStore  # noqa: PLC0415

    return AzureBlobObjectStore.from_settings()


_PROVIDERS: Dict[str, Callable[[], ObjectStore]] = {"azure": _azure}


def get_object_store(provider: Optional[str] = None) -> ObjectStore:
    if provider is None:
        from django.conf import settings  # noqa: PLC0415

        provider = getattr(settings, "DVSA_OBJECT_STORE", "azure")
    provider = (provider or "azure").strip().lower()
    try:
        factory = _PROVIDERS[provider]
    except KeyError:
        raise ValueError(f"unsupported object store provider: {provider!r}") from None
    return factory()


__all__ = ["ObjectCreated", "ObjectInfo", "ObjectStore", "StorageEventError",
           "UploadTarget", "get_object_store"]
