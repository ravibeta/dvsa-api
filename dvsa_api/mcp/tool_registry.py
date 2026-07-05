"""MCP **tool** registry — DVSA analytics exposed as Model-Context-Protocol tools.

Each tool carries a ``name``, ``description``, JSON-Schema ``input_schema`` and
``output_schema``, and a handler. Handlers reuse DVSA capabilities internally —
the reasoning adapters (``dvsa_api.reasoning``) and DVSA data (via the
:class:`~dvsa_api.mcp.resource_adapter.ResourceAdapter`) — so nothing about the
DVSA core changes. Everything runs offline/deterministically.

Tools registered:

* ``dvsa.detect_anomalies`` — accident/anomaly detection from tracks/frames.
* ``dvsa.run_reasoning``     — run a named/selected DVSA reasoning model.
* ``dvsa.extract_features``  — deterministic frame/track feature summary.
* ``dvsa.get_tracks``        — fetch object tracks for an id (resource-backed).
* ``dvsa.get_frames``        — fetch frame metadata for an id (resource-backed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .protocol_errors import InvalidParamsError, ToolExecutionError, ToolNotFoundError
from .resource_adapter import ResourceAdapter

Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class Tool:
    """An MCP tool definition + its Python handler."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Handler
    required: List[str] = field(default_factory=list)

    def definition(self) -> Dict[str, Any]:
        """MCP ``tools/list`` entry (``inputSchema`` is the MCP field name)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
        }


class ToolRegistry:
    """Holds tools and dispatches ``tools/call``."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.definition() for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool, validating required args; returns its structured result."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"unknown tool '{name}'", details={"available": list(self._tools)})
        arguments = arguments or {}
        missing = [k for k in tool.required if k not in arguments]
        if missing:
            raise InvalidParamsError(
                f"tool '{name}' missing required argument(s): {missing}",
                details={"missing": missing})
        try:
            return tool.handler(arguments)
        except (InvalidParamsError, ToolNotFoundError):
            raise
        except Exception as exc:  # noqa: BLE001 - surface as an MCP tool error
            raise ToolExecutionError(
                f"tool '{name}' failed: {exc}", details={"tool": name}) from exc


# --------------------------------------------------------------------------- #
# DVSA tool handlers (reuse reasoning + resource layers internally)
# --------------------------------------------------------------------------- #
def _detect_anomalies(args: Dict[str, Any]) -> Dict[str, Any]:
    from dvsa_api.reasoning import call_model  # noqa: PLC0415 - lazy, offline-safe
    context = {
        "tracks": args.get("tracks", []),
        "frames": args.get("frames", []),
        "query": args.get("query", "Detect anomalies (e.g. accidents) in the scene."),
    }
    model = args.get("model", "urban_accident_example")
    out = call_model(model, context)
    actions = out.get("actions", [])
    anomalies = [a for a in actions if a.get("type") == "anomaly"]
    return {
        "actions": actions,
        "anomalies": anomalies,
        "reasoning_trace": out.get("reasoning_trace", []),
        "anomaly_detected": bool(anomalies),
        "model": out.get("metadata", {}).get("model", model),
    }


def _run_reasoning(args: Dict[str, Any]) -> Dict[str, Any]:
    from dvsa_api.reasoning import call_model, select_model  # noqa: PLC0415
    context = args.get("context") or {
        "tracks": args.get("tracks", []),
        "frames": args.get("frames", []),
        "query": args.get("query", "Analyse the aerial scene."),
    }
    model = args.get("model")
    if not model:
        policy = args.get("policy", "latency_optimized")
        model = select_model(policy, context)
    out = call_model(model, context)
    return {
        "model": model,
        "actions": out.get("actions", []),
        "reasoning_trace": out.get("reasoning_trace", []),
        "metadata": out.get("metadata", {}),
    }


def _extract_features(args: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, dependency-free feature summary over frames/tracks."""
    tracks = args.get("tracks", []) or []
    frames = args.get("frames", []) or []
    track_features = []
    for track in tracks:
        samples = track.get("samples") or track.get("positions") or []
        track_features.append({
            "id": track.get("id"),
            "num_samples": len(samples),
            "class": track.get("class"),
        })
    return {
        "num_frames": len(frames),
        "num_tracks": len(tracks),
        "track_features": track_features,
        "has_multiple_tracks": len(tracks) >= 2,
    }


def _make_get_resource(kind: str, resources: ResourceAdapter) -> Handler:
    def _handler(args: Dict[str, Any]) -> Dict[str, Any]:
        rid = args.get("id")
        if not rid:
            raise InvalidParamsError(f"'{kind}' lookup requires an 'id'")
        uri = f"dvsa://{kind}/{rid}"
        return {"uri": uri, kind: resources.read(uri)}
    return _handler


def build_default_registry(
    resources: Optional[ResourceAdapter] = None,
) -> ToolRegistry:
    """Construct the registry pre-loaded with the DVSA tool set."""
    resources = resources or ResourceAdapter()
    registry = ToolRegistry()

    _tracks_schema = {
        "type": "array",
        "items": {"type": "object"},
        "description": "Object tracks with samples ({t, x, y[, vx, vy]}).",
    }
    _frames_schema = {
        "type": "array", "items": {"type": "object"},
        "description": "Frame metadata records.",
    }

    registry.register(Tool(
        name="dvsa.detect_anomalies",
        description="Detect anomalies (e.g. urban-street accidents) from object "
                    "tracks and frame metadata using DVSA reasoning models.",
        input_schema={
            "type": "object",
            "properties": {
                "tracks": _tracks_schema,
                "frames": _frames_schema,
                "query": {"type": "string"},
                "model": {"type": "string",
                          "description": "Reasoning model name (optional)."},
            },
            "required": ["tracks"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "actions": {"type": "array"},
                "anomalies": {"type": "array"},
                "reasoning_trace": {"type": "array"},
                "anomaly_detected": {"type": "boolean"},
            },
        },
        handler=_detect_anomalies,
        required=["tracks"],
    ))

    registry.register(Tool(
        name="dvsa.run_reasoning",
        description="Run a DVSA reasoning model over a context (explicit model or "
                    "policy-selected) and return actions + an explainable trace.",
        input_schema={
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "policy": {"type": "string",
                           "enum": ["by_name", "cost_optimized",
                                    "latency_optimized", "privacy_first"]},
                "context": {"type": "object"},
                "tracks": _tracks_schema,
                "frames": _frames_schema,
                "query": {"type": "string"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {"model": {"type": "string"}, "actions": {"type": "array"},
                           "reasoning_trace": {"type": "array"},
                           "metadata": {"type": "object"}},
        },
        handler=_run_reasoning,
    ))

    registry.register(Tool(
        name="dvsa.extract_features",
        description="Compute a deterministic feature summary of frames/tracks "
                    "(counts and per-track sample stats).",
        input_schema={
            "type": "object",
            "properties": {"frames": _frames_schema, "tracks": _tracks_schema},
        },
        output_schema={
            "type": "object",
            "properties": {"num_frames": {"type": "integer"},
                           "num_tracks": {"type": "integer"},
                           "track_features": {"type": "array"}},
        },
        handler=_extract_features,
    ))

    registry.register(Tool(
        name="dvsa.get_tracks",
        description="Fetch object tracks for a dataset id (dvsa://tracks/<id>).",
        input_schema={"type": "object",
                      "properties": {"id": {"type": "string"}},
                      "required": ["id"]},
        output_schema={"type": "object",
                       "properties": {"uri": {"type": "string"},
                                      "tracks": {"type": "array"}}},
        handler=_make_get_resource("tracks", resources),
        required=["id"],
    ))

    registry.register(Tool(
        name="dvsa.get_frames",
        description="Fetch frame metadata for a dataset id (dvsa://frames/<id>).",
        input_schema={"type": "object",
                      "properties": {"id": {"type": "string"}},
                      "required": ["id"]},
        output_schema={"type": "object",
                       "properties": {"uri": {"type": "string"},
                                      "frames": {"type": "array"}}},
        handler=_make_get_resource("frames", resources),
        required=["id"],
    ))

    return registry
