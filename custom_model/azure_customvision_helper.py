"""Optional helper for exporting/importing ONNX models with Azure Custom Vision.

This module is **optional**. For first-time setup the Azure portal or the
official ``azure-cognitiveservices-vision-customvision`` SDK is usually easier
and better supported. The helpers here are a thin, defensive ``requests``
wrapper for teams that want to script the export step in CI.

Security
--------
No secrets are hard-coded. Keys and endpoints are passed in by the caller and
should come from a secret store / environment variables, never from source.

References
----------
Custom Vision Training REST API — "ExportIteration" / "GetExports":
https://learn.microsoft.com/azure/ai-services/custom-vision-service/
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Custom Vision supports exporting trained iterations to several platforms;
# "ONNX" is the relevant one for this adapter.
_EXPORT_PLATFORM = "ONNX"


def _require_requests():
    """Import ``requests`` lazily with a friendly error if it is missing."""

    try:
        import requests  # noqa: WPS433 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The Azure Custom Vision helper requires the 'requests' package. "
            "Install it with `pip install -r custom_model/requirements.txt`."
        ) from exc
    return requests


def export_iteration_to_onnx(
    project_id: str,
    iteration_id: str,
    training_key: str,
    endpoint: str,
    timeout: int = 30,
) -> str:
    """Trigger an ONNX export of a trained Custom Vision iteration.

    Parameters
    ----------
    project_id, iteration_id:
        Identify the trained model to export.
    training_key:
        Custom Vision **training** key (not the prediction key).
    endpoint:
        Resource endpoint, e.g. ``https://<region>.api.cognitive.microsoft.com``.
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    str
        The ``downloadUri`` of the exported ONNX package once available.

    Raises
    ------
    ValueError
        If required arguments are empty.
    RuntimeError
        If the Custom Vision API returns an error or no download URI.
    """

    if not all([project_id, iteration_id, training_key, endpoint]):
        raise ValueError("project_id, iteration_id, training_key and endpoint are all required")

    requests = _require_requests()
    base = endpoint.rstrip("/")
    headers = {"Training-Key": training_key}
    export_url = (
        f"{base}/customvision/v3.3/training/projects/{project_id}"
        f"/iterations/{iteration_id}/export"
    )

    logger.info("Requesting %s export for iteration %s", _EXPORT_PLATFORM, iteration_id)
    resp = requests.post(export_url, params={"platform": _EXPORT_PLATFORM}, headers=headers, timeout=timeout)

    # 200 = export started/exists; 409 = an export is already in progress.
    if resp.status_code not in (200, 409):
        raise RuntimeError(
            f"Custom Vision export failed ({resp.status_code}): {resp.text[:500]}"
        )

    download_uri = _extract_download_uri(resp)
    if download_uri:
        return download_uri

    # Export is still flagging — poll the GetExports endpoint for the URI.
    get_url = export_url  # GET on the same path lists existing exports
    poll = requests.get(get_url, headers=headers, timeout=timeout)
    if poll.status_code != 200:
        raise RuntimeError(f"Could not list exports ({poll.status_code}): {poll.text[:500]}")
    download_uri = _extract_download_uri(poll)
    if not download_uri:
        raise RuntimeError(
            "Export request accepted but no downloadUri is available yet. "
            "Retry shortly — ONNX export is asynchronous."
        )
    return download_uri


def upload_onnx_to_customvision(
    project_id: str,
    iteration_id: str,
    onnx_path: str,
    prediction_key: str,
    endpoint: str,
    timeout: int = 30,
) -> str:
    """Obtain a hosted ONNX export URL for a Custom Vision iteration.

    Kept for API parity with the integration guide. Custom Vision does not let
    you *upload* an arbitrary ONNX file into an existing project; instead you
    train in Custom Vision and **export** the iteration as ONNX. This helper
    therefore validates the local artifact and returns the export download URI.

    Parameters
    ----------
    onnx_path:
        Local ONNX file to sanity-check before requesting the export.
    prediction_key:
        Passed through as the training key for the export call. (Custom Vision
        uses a training key for exports; supply the appropriate key.)

    Returns
    -------
    str
        The export download URI.
    """

    if onnx_path and not os.path.isfile(onnx_path):
        raise ValueError(f"Local ONNX file not found: {onnx_path!r}")
    if onnx_path and not onnx_path.lower().endswith(".onnx"):
        raise ValueError(f"Expected a .onnx file, got: {onnx_path!r}")

    logger.info("Resolving Custom Vision ONNX export for project %s", project_id)
    return export_iteration_to_onnx(
        project_id=project_id,
        iteration_id=iteration_id,
        training_key=prediction_key,
        endpoint=endpoint,
        timeout=timeout,
    )


def _extract_download_uri(resp) -> Optional[str]:
    """Best-effort extraction of a ready ONNX export ``downloadUri`` from a response."""

    try:
        data = resp.json()
    except ValueError:
        return None

    # A single export object or a list of them may be returned.
    exports = data if isinstance(data, list) else [data]
    for exp in exports:
        if not isinstance(exp, dict):
            continue
        if str(exp.get("platform", "")).upper() != _EXPORT_PLATFORM:
            continue
        if str(exp.get("status", "")).lower() == "done" and exp.get("downloadUri"):
            return exp["downloadUri"]
    return None
