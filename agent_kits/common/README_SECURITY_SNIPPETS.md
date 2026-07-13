# Security Snippets

Copy-paste-ready patterns used across the kits. See `agent_kits/SECURITY.md` for the
full policy.

## Constant-time HMAC verification (webhooks)

```python
import hashlib
import hmac
import os

def verify_webhook(body: bytes, signature_header: str) -> bool:
    secret = os.environ["WEBHOOK_SECRET"].encode()
    expected = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
    # Constant-time comparison avoids timing side channels.
    return hmac.compare_digest(expected, signature_header or "")
```

## Slack request signature (v0 scheme + replay window)

```python
import hashlib
import hmac
import os
import time

def verify_slack(ts: str, body: bytes, signature: str) -> bool:
    if abs(time.time() - int(ts)) > 60 * 5:      # reject replays > 5 min old
        return False
    secret = os.environ["SLACK_SIGNING_SECRET"].encode()
    base = b"v0:" + ts.encode() + b":" + body
    expected = "v0=" + hmac.new(secret, base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")
```

## Reading secrets (never hard-code)

```python
import os

api_key = os.environ.get("DVSA_API_KEY")   # returns None when unset — fail closed
if api_key is None:
    raise RuntimeError("DVSA_API_KEY is not configured")
```

## Minimal PII redaction hook

```python
def redact(detection: dict) -> dict:
    # Drop precise geo and any raw crop before persisting/logging.
    detection.pop("geo", None)
    detection.pop("crop", None)
    return detection
```

## Bearer-token dependency (FastAPI)

```python
from fastapi import Header, HTTPException
import hmac
import os

def require_api_key(authorization: str = Header(default="")) -> None:
    expected = os.environ.get("DVSA_API_KEY")
    if not expected:                     # auth disabled for local dev only
        return
    token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
```
