"""Search-document formation + upload (ported from ezvision myvideoindexer.py).

Maps a vectorized/analyzed aerial frame into the AI Search index schema
(:mod:`core.azure.index_schema`) and uploads it. Also ports ``geolocation`` (the
Perplexity image-geolocation call) used to populate the ``geotags`` field.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import AzureEnvironmentConfig

logger = logging.getLogger("apps.azure")


def geolocation(config: AzureEnvironmentConfig, image_url: str):
    """Estimate ``(latitude, longitude)`` for an image via Perplexity geo API."""
    if not config.perplexity_geo_api_key:
        return "", ""
    import requests  # noqa: PLC0415

    headers = {"Authorization": f"Bearer {config.perplexity_geo_api_key}",
               "Content-Type": "application/json"}
    try:
        resp = requests.post(config.perplexity_geo_api_url, headers=headers,
                             json={"image_url": image_url}, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("latitude", ""), data.get("longitude", "")
    except Exception as exc:  # noqa: BLE001
        logger.info("geolocation failed: %s", exc)
    return "", ""


def build_document(*, account_id: str, frame_number: int, vector: List[float],
                   description: str, path: str, geotags: str = "",
                   boundingbox: str = "0,0,1280,720", caption: str = "",
                   labels: Optional[List[str]] = None, tags: Optional[List[str]] = None,
                   objects: str = "", user: str = "", session: str = "") -> Dict[str, Any]:
    """Assemble an index document spanning the canonical + fidelity fields."""
    return {
        "id": f"{account_id}-{frame_number:04d}",
        "account_id": str(account_id),
        "vector": vector,
        "description": description,
        "caption": caption or description[:280],
        "labels": labels or [],
        "tags": tags or [],
        "objects": objects,
        "boundingbox": boundingbox,
        "geotags": geotags,
        "path": path,
        "user": str(user),
        "session": str(session),
        "created": datetime.now(timezone.utc).isoformat(),
    }


def form_and_upload_document(config: AzureEnvironmentConfig, search_client, account_id,
                             frame_number, vector, description, source_sas_url,
                             *, user="", session="", deep=False) -> Dict[str, Any]:
    """Build a frame document, attach geotags, and upload it to the index.

    Returns the uploaded document's id (and ``dryrun`` flag when no search
    client is available).
    """
    vec = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    geotags = str(geolocation(config, source_sas_url))
    doc = build_document(
        account_id=account_id, frame_number=frame_number, vector=vec,
        description=description, path=source_sas_url, geotags=geotags,
        user=user, session=session,
    )
    if search_client is None:
        logger.info("dryrun form_and_upload_document %s", doc["id"])
        return {"indexed": doc["id"], "dryrun": True}
    results = search_client.upload_documents([doc])
    errors = ",".join(r.error_message for r in results if getattr(r, "error_message", None)).strip(",")
    if errors:
        logger.info("upload errors: %s", errors)
    return {"indexed": doc["id"], "errors": errors or None}


def parse_description_json(description: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the analyzer description string into a dict."""
    try:
        return json.loads(description)
    except Exception:  # noqa: BLE001
        return None
