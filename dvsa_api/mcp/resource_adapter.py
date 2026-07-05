"""Expose DVSA datasets as MCP **resources** (frames, tracks, sensor metadata).

Resources are addressed by URI following the ``dvsa://<kind>/<id>`` scheme:

* ``dvsa://frames/<id>``  — frame metadata for a clip/session.
* ``dvsa://tracks/<id>``  — object tracks for a clip/session.
* ``dvsa://sensor/<id>``  — sensor/telemetry metadata (gps, altitude, time).

Data is loaded, in order of preference, from a local data root (default
``mcp/resource_data/<kind>/<id>.json``), an ``http(s)`` URL, or a DVSA-API
endpoint (``MCP_DVSA_API_BASE``). Everything is offline-safe: with no network and
the bundled sample data, ``dvsa://tracks/demo`` resolves from local files.

This is the Model-Context-Protocol resource layer and is independent of the
agent *control plane* modules that also live in this package.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .protocol_errors import ResourceNotFoundError

KINDS = ("frames", "tracks", "sensor")
SCHEME = "dvsa://"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_ROOT = os.environ.get(
    "MCP_RESOURCE_DATA_ROOT", os.path.join(_REPO_ROOT, "mcp", "resource_data"))


@dataclass(frozen=True)
class ResourceDescriptor:
    """One MCP resource entry (as returned by ``resources/list``)."""

    uri: str
    name: str
    description: str
    mime_type: str = "application/json"

    def to_mcp(self) -> Dict[str, Any]:
        return {"uri": self.uri, "name": self.name,
                "description": self.description, "mimeType": self.mime_type}


def parse_uri(uri: str) -> Dict[str, str]:
    """Split a ``dvsa://<kind>/<id>`` URI into ``{kind, id}``."""
    if not uri.startswith(SCHEME):
        raise ResourceNotFoundError(
            f"unsupported resource URI scheme: {uri!r}", details={"uri": uri})
    rest = uri[len(SCHEME):]
    parts = rest.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ResourceNotFoundError(
            f"malformed resource URI: {uri!r} (expected dvsa://<kind>/<id>)",
            details={"uri": uri})
    kind, rid = parts[0], parts[1]
    if kind not in KINDS:
        raise ResourceNotFoundError(
            f"unknown resource kind '{kind}'", details={"valid": list(KINDS)})
    return {"kind": kind, "id": rid}


class ResourceAdapter:
    """Loads DVSA datasets and presents them as MCP resources."""

    def __init__(self, *, data_root: Optional[str] = None,
                 api_base: Optional[str] = None) -> None:
        self.data_root = data_root or DEFAULT_DATA_ROOT
        self.api_base = api_base or os.environ.get("MCP_DVSA_API_BASE")

    # ----- listing ------------------------------------------------------
    def list_resources(self) -> List[ResourceDescriptor]:
        """Enumerate resources discoverable under the local data root."""
        out: List[ResourceDescriptor] = []
        for kind in KINDS:
            kind_dir = os.path.join(self.data_root, kind)
            if not os.path.isdir(kind_dir):
                continue
            for entry in sorted(os.listdir(kind_dir)):
                if not entry.endswith(".json"):
                    continue
                rid = entry[: -len(".json")]
                out.append(ResourceDescriptor(
                    uri=f"{SCHEME}{kind}/{rid}",
                    name=f"{kind}:{rid}",
                    description=f"DVSA {kind} dataset '{rid}'"))
        return out

    def resource_templates(self) -> List[Dict[str, Any]]:
        """MCP resource templates for the addressable URI patterns."""
        return [
            {"uriTemplate": f"{SCHEME}{kind}/{{id}}",
             "name": f"dvsa-{kind}",
             "description": f"DVSA {kind} by id",
             "mimeType": "application/json"}
            for kind in KINDS
        ]

    # ----- reading ------------------------------------------------------
    def read(self, uri: str) -> Dict[str, Any]:
        """Return the parsed payload for ``uri`` (dict/list)."""
        ref = parse_uri(uri)
        # 1) local file, 2) DVSA-API endpoint, 3) explicit http(s) id.
        local = os.path.join(self.data_root, ref["kind"], f"{ref['id']}.json")
        if os.path.isfile(local):
            with open(local, "r", encoding="utf-8") as fh:
                return json.load(fh)
        if self.api_base:
            payload = self._read_http(
                f"{self.api_base.rstrip('/')}/{ref['kind']}/{ref['id']}")
            if payload is not None:
                return payload
        if ref["id"].startswith("http://") or ref["id"].startswith("https://"):
            payload = self._read_http(ref["id"])
            if payload is not None:
                return payload
        raise ResourceNotFoundError(
            f"resource not found: {uri}", details={"uri": uri, "looked_in": local})

    def read_mcp(self, uri: str) -> Dict[str, Any]:
        """Return an MCP ``resources/read`` result for ``uri``."""
        payload = self.read(uri)
        return {"contents": [{
            "uri": uri,
            "mimeType": "application/json",
            "text": json.dumps(payload),
        }]}

    def _read_http(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None
