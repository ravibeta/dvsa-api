"""Offline helpers shared across the runtime services.

These keep the ported runtime exercisable without Azure credentials or network
access — the same offline-first philosophy as ``apps.observability.llm``.
"""

from __future__ import annotations

import hashlib
import struct
from typing import List


def deterministic_embedding(text: str, dims: int) -> List[float]:
    """Stable pseudo-embedding from a SHA-256 stream (offline, reproducible)."""
    out: List[float] = []
    counter = 0
    while len(out) < dims:
        h = hashlib.sha256(f"{text}:{counter}".encode("utf-8")).digest()
        for i in range(0, 32, 4):
            out.append(struct.unpack("<I", h[i:i + 4])[0] / 0xFFFFFFFF)
            if len(out) >= dims:
                break
        counter += 1
    return out[:dims]


def retry(*dargs, **dkw):
    """Best-effort ``tenacity.retry`` shim.

    Uses tenacity when installed; otherwise returns the function unchanged so
    modules import cleanly without the dependency.
    """
    try:
        from tenacity import retry as _retry  # noqa: PLC0415

        return _retry(*dargs, **dkw)
    except Exception:  # noqa: BLE001 - tenacity absent
        def _decorator(fn):
            return fn

        # Support both @retry and @retry(...) usage.
        if len(dargs) == 1 and callable(dargs[0]) and not dkw:
            return dargs[0]
        return _decorator
