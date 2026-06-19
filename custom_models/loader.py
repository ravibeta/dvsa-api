"""Unified model description + discovery for the ``custom_models`` package.

This module is intentionally lightweight — it imports only :mod:`numpy`-free
standard library and re-uses the proven label-map loader from the existing
:mod:`custom_model` package. Heavy runtimes (``onnxruntime``, ``torch``,
``ultralytics``) are imported lazily inside the individual adapters, never here,
so configuration/selection code can be imported cheaply.

Concepts
--------
``ModelSpec``
    A format-agnostic description of a single model artifact: where it lives,
    what runtime loads it, the labels it predicts and a few hints (input size,
    tile recommendation, capabilities) used by :class:`custom_models.selector.ModelSelector`.

``discover_model(path)``
    Infer a :class:`ModelSpec` from a filesystem path by looking at the file
    extension (and a light filename heuristic for YOLO artifacts).

``load_label_map(path)``
    Load ``{class_id: label}`` — delegates to the battle-tested implementation
    in :func:`custom_model.model_loader.load_label_map`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

# Re-use the existing, well-tested label loader rather than duplicating it.
from custom_model.model_loader import load_label_map  # noqa: F401  (re-exported)

__all__ = ["ModelSpec", "discover_model", "load_label_map", "EXTENSION_FORMATS"]


# Map a file extension to the canonical adapter format name understood by the
# registry. ``.pt``/``.pth`` are ambiguous (torchvision TorchScript vs an
# Ultralytics YOLO checkpoint); ``discover_model`` resolves that with a filename
# heuristic, defaulting to "torch".
EXTENSION_FORMATS: Dict[str, str] = {
    ".onnx": "onnx",
    ".pt": "torch",
    ".pth": "torch",
    ".torchscript": "torch",
    ".engine": "onnx",  # TensorRT engines are most often paired with the onnx path
}


@dataclass
class ModelSpec:
    """A format-agnostic description of one detection model.

    Attributes
    ----------
    id:
        Stable identifier (e.g. ``"visdrone-yolov8x"``). Used as the catalog key.
    name:
        Human-readable name.
    format:
        Canonical runtime format: ``"onnx"``, ``"torch"`` or ``"yolo"``. Selects
        the adapter via :mod:`custom_models.registry`.
    path:
        Local filesystem path to the artifact. May be empty when the spec comes
        from a catalog and the weights have not been downloaded yet.
    labels_file:
        Path to a ``label_map.json`` (``{class_id: label}``). Optional for YOLO
        artifacts, which usually embed class names.
    input_size:
        ``(width, height)`` the model expects, in pixels.
    tile_recommendation:
        Optional ``(width, height)`` tile size recommended for large aerial
        frames. ``None`` means tiling is not recommended.
    capabilities:
        Coarse capability tags (e.g. ``["vehicle", "person", "bicycle"]``) used
        by the selector to match a query's requested classes.
    task:
        Coarse task tag, e.g. ``"detection"`` or ``"obb"`` (oriented boxes).
    altitude:
        Coarse altitude suitability tag: ``"low"``, ``"medium"``, ``"high"`` or
        ``"any"``. Used by the selector.
    source_url, artifact_filename:
        Provenance metadata copied from the catalog (download URL + filename).
    metadata:
        Free-form extra fields preserved from the catalog entry.
    """

    id: str
    name: str = ""
    format: str = "onnx"
    path: str = ""
    labels_file: Optional[str] = None
    input_size: Tuple[int, int] = (640, 640)
    tile_recommendation: Optional[Tuple[int, int]] = None
    capabilities: List[str] = field(default_factory=list)
    task: str = "detection"
    altitude: str = "any"
    source_url: str = ""
    artifact_filename: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.format = str(self.format).lower()
        if tuple(self.input_size) and len(tuple(self.input_size)) != 2:
            raise ValueError(f"input_size must be (width, height), got {self.input_size!r}")
        self.input_size = (int(self.input_size[0]), int(self.input_size[1]))
        if self.tile_recommendation is not None:
            self.tile_recommendation = (
                int(self.tile_recommendation[0]),
                int(self.tile_recommendation[1]),
            )
        self.capabilities = [str(c).lower() for c in self.capabilities]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_catalog(cls, entry: Mapping, *, base_dir: str = "") -> "ModelSpec":
        """Build a :class:`ModelSpec` from a ``models_catalog.json`` entry.

        ``base_dir`` is prepended to ``artifact_filename``/``labels_file`` when
        they are relative, so a catalog can ship alongside a weights directory.
        """

        known = {
            "id", "name", "format", "labels_file", "input_size",
            "tile_recommendation", "capabilities", "task", "altitude",
            "source_url", "artifact_filename",
        }
        artifact = entry.get("artifact_filename", "") or ""
        labels = entry.get("labels_file")

        def _resolve(p: Optional[str]) -> Optional[str]:
            if not p or not base_dir:
                return p
            return p if os.path.isabs(p) else os.path.join(base_dir, p)

        return cls(
            id=str(entry["id"]),
            name=str(entry.get("name", entry["id"])),
            format=str(entry.get("format", "onnx")),
            path=_resolve(artifact) or "",
            labels_file=_resolve(labels),
            input_size=_coerce_size(entry.get("input_size"), default=(640, 640)),
            tile_recommendation=_coerce_size(entry.get("tile_recommendation"), default=None),
            capabilities=list(entry.get("capabilities", []) or []),
            task=str(entry.get("task", "detection")),
            altitude=str(entry.get("altitude", "any")),
            source_url=str(entry.get("source_url", "")),
            artifact_filename=artifact,
            metadata={k: v for k, v in entry.items() if k not in known},
        )


def _coerce_size(value, *, default):
    """Coerce ``[w, h]`` / ``"WxH"`` / ``None`` into a ``(w, h)`` tuple or default."""

    if value is None:
        return default
    if isinstance(value, str):
        w, h = value.lower().split("x")
        return (int(w), int(h))
    seq = list(value)
    if len(seq) != 2:
        raise ValueError(f"size must have two components, got {value!r}")
    return (int(seq[0]), int(seq[1]))


def discover_model(path: str, *, labels_file: Optional[str] = None) -> ModelSpec:
    """Infer a :class:`ModelSpec` from a filesystem ``path``.

    The runtime ``format`` is derived from the file extension (see
    :data:`EXTENSION_FORMATS`). A ``.pt``/``.pth`` whose filename contains
    ``"yolo"`` is classified as the ``"yolo"`` format so the Ultralytics adapter
    handles it; other ``.pt`` files default to the generic ``"torch"`` adapter.
    An ``.onnx`` whose filename contains ``"yolo"`` is likewise routed to the
    YOLO adapter, which can consume exported YOLO ONNX directly.

    Parameters
    ----------
    path:
        Path to the model artifact. The file need not exist yet (discovery is
        purely lexical), but ``id`` is derived from the basename.
    labels_file:
        Optional explicit label-map path to attach to the spec.

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """

    if not path:
        raise ValueError("path must be a non-empty string")

    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    ext = ext.lower()

    if ext not in EXTENSION_FORMATS:
        raise ValueError(
            f"Unrecognised model extension {ext!r} for {path!r}; "
            f"expected one of {sorted(EXTENSION_FORMATS)}"
        )

    fmt = EXTENSION_FORMATS[ext]
    if "yolo" in stem.lower():
        # YOLO checkpoints (.pt) and exported YOLO ONNX are both best handled by
        # the Ultralytics adapter, which auto-detects the underlying artifact.
        fmt = "yolo"

    return ModelSpec(
        id=stem,
        name=stem,
        format=fmt,
        path=path,
        labels_file=labels_file,
    )
