"""Template reasoning adapter — copy this folder to add your own model.

Contract (see ``dvsa_api/reasoning/adapter_base.py``):

* ``predict(self, context: dict) -> dict`` returning
  ``{"actions": [...], "reasoning_trace": [...], "metadata": {...}}``.
* ``health_check(self) -> dict`` returning ``{"status": "ok"}`` (or error detail).

The class **must** be named ``ReasoningModelAdapter``. It may accept a ``config``
keyword (merged from ``manifest.json``'s ``config`` and an optional
``config.json`` sidecar) or take no arguments. Keep dependencies minimal — this
template is pure standard library.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

MODEL_NAME = "_template_model"
MODEL_VERSION = "0.1.0"


class ReasoningModelAdapter:
    """Minimal, dependency-free example implementing the adapter contract."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        # Model-specific config comes from manifest.json ("config") / config.json.
        self.confidence_floor = float(config.get("confidence_floor", 0.5))

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Echo a trivial, deterministic decision so the pipeline is exercised."""
        started = time.time()
        query = context.get("query", "")
        tracks = context.get("tracks") or []
        actions = [{
            "type": "label",
            "label": "nominal",
            "confidence": max(0.5, self.confidence_floor),
        }]
        reasoning_trace = [
            f"template model received query: {query!r}",
            f"observed {len(tracks)} track(s); no domain logic in template",
        ]
        return {
            "actions": actions,
            "reasoning_trace": reasoning_trace,
            "metadata": {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "tokens": None,
                "latency_ms": int((time.time() - started) * 1000),
                "request_id": uuid.uuid4().hex,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION}
