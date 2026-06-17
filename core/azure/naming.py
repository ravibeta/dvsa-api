"""Deterministic naming for per-session logical isolation.

Global resources (storage account, search service, Foundry account) are shared.
*Session* isolation is achieved logically and is fully deterministic from the
session id, so teardown can reconstruct every name without external state:

- ``index`` isolation  -> a dedicated search index ``{base}-s-{slug}``.
- ``filter`` isolation -> the shared base index; rows carry ``session``/``user``
  fields and are filtered at query time.

Blob artifacts are always namespaced under a per-session prefix inside the
shared ``input``/``output`` containers.
"""

from __future__ import annotations

import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def session_slug(session_id: str) -> str:
    """Normalize an arbitrary session/user id into an Azure-name-safe slug.

    Lowercased, non-alphanumerics collapsed to ``-``, trimmed to 40 chars so
    composed index names stay within Azure Search's 128-char limit.
    """
    slug = _SLUG_RE.sub("-", str(session_id).strip().lower()).strip("-")
    return (slug or "session")[:40]


def index_name(base: str, session_id: str, isolation: str) -> str:
    """Return the search index name for ``session_id`` under ``isolation``."""
    if isolation == "filter":
        return base
    return f"{base}-s-{session_slug(session_id)}"


def blob_prefix(session_id: str) -> str:
    """Per-session virtual-directory prefix used inside each container."""
    return f"sessions/{session_slug(session_id)}/"


def blob_path(session_id: str, name: str) -> str:
    """Full blob path for ``name`` within the session's prefix."""
    return blob_prefix(session_id) + name.lstrip("/")
