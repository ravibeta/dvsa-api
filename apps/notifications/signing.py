"""HMAC-SHA256 webhook signatures (``X-DVSA-Signature: t=<unix>,v1=<hex>``).

The signed message is ``f"{t}.{raw_body}"``, so a captured body cannot be replayed with a new
timestamp. Consumers verify with :func:`verify_signature` (or the same few lines in any language).
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional, Union

SIGNATURE_HEADER = "X-DVSA-Signature"
EVENT_HEADER = "X-DVSA-Event"
DELIVERY_HEADER = "X-DVSA-Delivery-ID"
TIMESTAMP_HEADER = "X-DVSA-Timestamp"


def _digest(secret: str, timestamp: int, body: bytes) -> str:
    message = str(timestamp).encode() + b"." + body
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def sign(secret: str, body: Union[bytes, str], timestamp: Optional[int] = None) -> str:
    if isinstance(body, str):
        body = body.encode()
    timestamp = int(time.time()) if timestamp is None else int(timestamp)
    return f"t={timestamp},v1={_digest(secret, timestamp, body)}"


def verify_signature(secret: str, body: Union[bytes, str], header: str,
                     tolerance_s: int = 300, now: Optional[float] = None) -> bool:
    """True when ``header`` is a valid, fresh signature of ``body`` under ``secret``."""
    if isinstance(body, str):
        body = body.encode()
    try:
        parts = dict(item.split("=", 1) for item in (header or "").split(","))
        timestamp = int(parts["t"])
        provided = parts["v1"]
    except (KeyError, ValueError):
        return False
    now = time.time() if now is None else now
    if abs(now - timestamp) > tolerance_s:
        return False
    return hmac.compare_digest(provided, _digest(secret, timestamp, body))
