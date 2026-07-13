"""Slack slash-command trigger — ``/dvsa-run <video_uri> [key=value ...]``.

Verifies the Slack request signature (the documented ``v0`` HMAC-SHA256 scheme with
a ±5-minute replay window), parses the slash-command form body into a
:class:`RunInput`, dispatches the run, and returns a Slack-friendly message. Core
logic is pure functions so it runs under any web framework (or none) and is fully
testable offline.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs

from ....common import RunInput
from ...run_tracker import Dispatcher, RunTracker, make_default_dispatcher

TIMESTAMP_HEADER = "X-Slack-Request-Timestamp"
SIGNATURE_HEADER = "X-Slack-Signature"
REPLAY_WINDOW_S = 60 * 5


def compute_slack_signature(timestamp: str, body: bytes, secret: str) -> str:
    """Return the Slack ``v0=<hex>`` signature for a request."""
    base = b"v0:" + timestamp.encode("utf-8") + b":" + body
    digest = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return f"v0={digest}"


def verify_slack_signature(
    timestamp: str, body: bytes, signature: str, *,
    secret: Optional[str] = None, now: Optional[float] = None,
) -> bool:
    """Constant-time verify a Slack request, rejecting stale timestamps (replay)."""
    secret = secret if secret is not None else os.environ.get("SLACK_SIGNING_SECRET")
    if not secret:
        return False  # fail closed
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs((now if now is not None else time.time()) - ts) > REPLAY_WINDOW_S:
        return False
    expected = compute_slack_signature(timestamp, body, secret)
    return hmac.compare_digest(expected, signature or "")


def parse_command_text(text: str) -> Dict[str, Any]:
    """Parse ``<video_uri> [fps=5 sensor_id=cam1 ...]`` into structured fields."""
    tokens = (text or "").split()
    if not tokens:
        raise ValueError("no video URI supplied")
    video_uri = tokens[0]
    flags: Dict[str, Any] = {}
    fields: Dict[str, Any] = {}
    for tok in tokens[1:]:
        if "=" not in tok:
            continue
        key, _, value = tok.partition("=")
        if key in ("sensor_id", "agent_id", "run_id"):
            fields[key] = value
        elif key == "fps":
            flags["fps"] = float(value)
        else:
            flags[key] = value
    return {"video_uri": video_uri, "processing_flags": flags, **fields}


def slack_payload_to_run_input(form: Dict[str, Any]) -> RunInput:
    """Convert a decoded Slack slash-command form into a :class:`RunInput`."""
    parsed = parse_command_text(form.get("text", ""))
    return RunInput(
        video_uri=parsed["video_uri"],
        sensor_id=parsed.get("sensor_id"),
        processing_flags=parsed.get("processing_flags") or {},
        agent_id=parsed.get("agent_id", f"slack:{form.get('user_id', 'unknown')}"),
        run_id=parsed.get("run_id"),
    )


def _decode_form(body: bytes) -> Dict[str, str]:
    return {k: v[0] for k, v in parse_qs(body.decode("utf-8")).items()}


def handle_slack_command(
    headers: Dict[str, str], body: bytes, *,
    secret: Optional[str] = None,
    dispatcher: Optional[Dispatcher] = None,
    tracker: Optional[RunTracker] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Verify → parse → dispatch a slash command. Returns ``(status, slack_json)``."""
    timestamp = _header(headers, TIMESTAMP_HEADER)
    signature = _header(headers, SIGNATURE_HEADER)
    if not verify_slack_signature(timestamp, body, signature, secret=secret):
        return 401, {"response_type": "ephemeral", "text": "invalid Slack signature"}
    try:
        form = _decode_form(body)
        run_input = slack_payload_to_run_input(form)
    except ValueError as exc:
        return 200, {"response_type": "ephemeral",
                     "text": f"Usage: /dvsa-run <video_uri> [fps=5 ...] — {exc}"}
    dispatch = dispatcher or make_default_dispatcher(tracker)
    record = dispatch(run_input)
    return 200, {
        "response_type": "in_channel",
        "text": (f":satellite: DVSA run *{record['run_id']}* "
                 f"status *{record['status']}* "
                 f"({record.get('detections', 0)} detections)"),
        "run": record,
    }


def _header(headers: Dict[str, str], name: str) -> str:
    lowered = {k.lower(): v for k, v in headers.items()}
    return lowered.get(name.lower(), "")


__all__ = [
    "compute_slack_signature",
    "verify_slack_signature",
    "parse_command_text",
    "slack_payload_to_run_input",
    "handle_slack_command",
    "TIMESTAMP_HEADER",
    "SIGNATURE_HEADER",
]
