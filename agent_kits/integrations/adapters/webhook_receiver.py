"""Generic webhook receiver — turn a signed HTTP event into a run.

Verifies an HMAC signature (``X-DVSA-Signature: sha256=<hex>`` computed with
``WEBHOOK_SECRET``), converts the JSON payload into a :class:`RunInput`, and
dispatches it. The core is framework-agnostic pure functions so it is trivially
testable; an optional stdlib ``http.server`` factory (:func:`make_server`) is
provided for a runnable receiver with no third-party dependency.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Dict, Optional, Tuple

from ...common import RunInput
from ..run_tracker import Dispatcher, RunTracker, make_default_dispatcher

SIGNATURE_HEADER = "X-DVSA-Signature"


def compute_signature(body: bytes, secret: str) -> str:
    """Return the ``sha256=<hex>`` signature for ``body`` under ``secret``."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_webhook_signature(
    body: bytes, signature: str, *, secret: Optional[str] = None,
) -> bool:
    """Constant-time verify the webhook HMAC signature."""
    secret = secret if secret is not None else os.environ.get("WEBHOOK_SECRET")
    if not secret:
        return False  # fail closed when no secret configured
    expected = compute_signature(body, secret)
    return hmac.compare_digest(expected, signature or "")


def webhook_payload_to_run_input(payload: Dict[str, Any]) -> RunInput:
    """Convert a webhook JSON payload into a validated :class:`RunInput`."""
    video_uri = payload.get("video_uri") or payload.get("video")
    if not video_uri:
        raise ValueError("webhook payload missing 'video_uri'")
    return RunInput(
        video_uri=video_uri,
        sensor_id=payload.get("sensor_id"),
        time_window=payload.get("time_window"),
        processing_flags=payload.get("processing_flags") or {},
        agent_id=payload.get("agent_id", "webhook-trigger"),
        run_id=payload.get("run_id"),
    )


def handle_webhook(
    headers: Dict[str, str], body: bytes, *,
    secret: Optional[str] = None,
    dispatcher: Optional[Dispatcher] = None,
    tracker: Optional[RunTracker] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Verify → parse → dispatch a webhook. Returns ``(status_code, json_body)``."""
    signature = _header(headers, SIGNATURE_HEADER)
    if not verify_webhook_signature(body, signature, secret=secret):
        return 401, {"error": "invalid signature"}
    try:
        payload = json.loads(body.decode("utf-8"))
        run_input = webhook_payload_to_run_input(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        return 400, {"error": f"bad request: {exc}"}
    dispatch = dispatcher or make_default_dispatcher(tracker)
    record = dispatch(run_input)
    return 202, {"accepted": True, "run": record}


def _header(headers: Dict[str, str], name: str) -> str:
    lowered = {k.lower(): v for k, v in headers.items()}
    return lowered.get(name.lower(), "")


def make_server(host: str = "127.0.0.1", port: int = 8090,
                *, dispatcher: Optional[Dispatcher] = None):
    """Build a stdlib ``HTTPServer`` exposing ``POST /webhook`` (no extra deps)."""
    from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: PLC0415

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            headers = {k: v for k, v in self.headers.items()}
            status, payload = handle_webhook(headers, body, dispatcher=dispatcher)
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:  # silence default logging
            return

    return HTTPServer((host, port), Handler)


__all__ = [
    "compute_signature",
    "verify_webhook_signature",
    "webhook_payload_to_run_input",
    "handle_webhook",
    "make_server",
    "SIGNATURE_HEADER",
]
