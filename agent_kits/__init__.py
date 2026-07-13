"""DVSA Agent Kits — Warp-style agent patterns for drone-video analytics.

Four complementary kits share a single foundation (``agent_kits.common``):

* **local_interactive** — agent-guided local execution & experimentation.
* **cloud_autonomous** — unattended containerised processing (FastAPI runner).
* **orchestration** — parent/child partitioned execution + deterministic merge.
* **integrations** — event-driven triggers (GitHub Actions, Slack, webhooks).

Everything is deterministic, offline-safe, and env-var configured; no secrets are
committed and all external services are mocked in tests. See ``agent_kits/README.md``.
"""

from __future__ import annotations

import os

__all__ = ["__version__", "version"]


def _read_version() -> str:
    path = os.path.join(os.path.dirname(__file__), "VERSION")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:  # pragma: no cover - VERSION always ships with the package
        return "0.0.0"


__version__ = _read_version()


def version() -> str:
    """Return the agent-kits package version (source of truth: ``VERSION``)."""
    return __version__
