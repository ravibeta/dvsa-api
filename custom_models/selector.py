"""``ModelSelector`` — map a user query to the best-fit model in the catalog.

The catalog (``models_catalog.json``) describes the curated drone-detection
models. A :class:`SelectionQuery` expresses what the caller needs — the task,
the object classes of interest, the capture altitude and the source resolution —
and :class:`ModelSelector` ranks the catalog entries by how well they match.

    selector = ModelSelector.default()
    spec = selector.select(classes=["vehicle", "person"], altitude="high",
                           resolution=(3840, 2160))
    detector = get_detector(spec).load()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .loader import ModelSpec

logger = logging.getLogger(__name__)

_DEFAULT_CATALOG = os.path.join(os.path.dirname(__file__), "models_catalog.json")

# Scoring weights — tuned so class coverage dominates, then task, then altitude
# and a small resolution/tiling nudge.
_W_CLASSES = 3.0
_W_TASK = 1.5
_W_ALTITUDE = 1.0
_W_RESOLUTION = 1.0

_LARGE_FRAME_PX = 1920  # max(w, h) above this is treated as "large" => prefer tiling


@dataclass
class SelectionQuery:
    """What the caller needs from a model. All fields optional."""

    task: Optional[str] = None
    classes: Optional[List[str]] = None
    altitude: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None


@dataclass
class ModelSelector:
    """Rank/select catalog :class:`ModelSpec` entries against a query."""

    specs: List[ModelSpec] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str, *, base_dir: Optional[str] = None) -> "ModelSelector":
        """Load a selector from a ``models_catalog.json`` file."""

        if not path or not os.path.isfile(path):
            raise ValueError(f"Catalog not found: {path!r}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_catalog(data, base_dir=base_dir if base_dir is not None else os.path.dirname(path))

    @classmethod
    def from_catalog(cls, data, *, base_dir: str = "") -> "ModelSelector":
        """Build a selector from already-parsed catalog data.

        ``data`` may be a list of entries or a ``{"models": [...]}`` object.
        """

        entries = data["models"] if isinstance(data, dict) else data
        specs = [ModelSpec.from_catalog(e, base_dir=base_dir) for e in entries]
        return cls(specs=specs)

    @classmethod
    def default(cls) -> "ModelSelector":
        """Load the catalog bundled with the package."""

        return cls.from_file(_DEFAULT_CATALOG)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def get(self, model_id: str) -> ModelSpec:
        """Return the spec with ``id == model_id``."""

        for spec in self.specs:
            if spec.id == model_id:
                return spec
        raise KeyError(
            f"Unknown model id {model_id!r}. Available: {[s.id for s in self.specs]}"
        )

    def rank(self, query: Optional[SelectionQuery] = None, **kwargs) -> List[Tuple[float, ModelSpec]]:
        """Return ``[(score, spec), ...]`` sorted by descending score.

        ``query`` may be a :class:`SelectionQuery` or supplied as keyword
        arguments (``task=``, ``classes=``, ``altitude=``, ``resolution=``).
        Ties break on ``spec.id`` for determinism.
        """

        q = query or SelectionQuery(**kwargs)
        scored = [(self._score(spec, q), spec) for spec in self.specs]
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return scored

    def select(self, query: Optional[SelectionQuery] = None, **kwargs) -> ModelSpec:
        """Return the single best-fit spec for ``query``.

        Raises
        ------
        ValueError
            If the catalog is empty.
        """

        ranked = self.rank(query, **kwargs)
        if not ranked:
            raise ValueError("Catalog is empty; nothing to select")
        score, spec = ranked[0]
        logger.info("Selected model %s (score=%.3f)", spec.id, score)
        return spec

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _score(spec: ModelSpec, q: SelectionQuery) -> float:
        score = 0.0

        if q.classes:
            requested = {c.lower() for c in q.classes}
            covered = requested & set(spec.capabilities)
            # Fraction of the requested classes the model can detect.
            score += _W_CLASSES * (len(covered) / len(requested))

        if q.task:
            score += _W_TASK if spec.task == q.task.lower() else 0.0

        if q.altitude:
            if spec.altitude in (q.altitude.lower(), "any"):
                score += _W_ALTITUDE

        if q.resolution:
            largest = max(int(q.resolution[0]), int(q.resolution[1]))
            is_large = largest > _LARGE_FRAME_PX
            if is_large and spec.tile_recommendation is not None:
                score += _W_RESOLUTION  # large frames benefit from tiling
            elif not is_large and spec.tile_recommendation is None:
                score += 0.25 * _W_RESOLUTION  # small frames: avoid needless tiling

        return score
