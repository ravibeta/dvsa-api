"""Celery task that delivers one signed webhook, retrying with exponential backoff."""

from __future__ import annotations

import json
import logging

import requests
from celery import shared_task
from django.conf import settings
from django.utils import timezone

from . import signing
from .models import WebhookDeliveryLog
from .url_safety import UnsafeWebhookURL, validate_target_url

logger = logging.getLogger("apps.notifications")

MAX_RETRIES = 3            # attempts = 1 + MAX_RETRIES
BACKOFF_BASE_S = 10        # 10 s, 20 s, 40 s
RETRYABLE_STATUS = {408, 425, 429}


def encode_body(payload) -> bytes:
    """Canonical JSON bytes; the signature covers exactly these bytes."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def backoff_seconds(attempt: int) -> int:
    return BACKOFF_BASE_S * 2 ** max(0, attempt - 1)


def _retryable(status_code) -> bool:
    return status_code is None or status_code >= 500 or status_code in RETRYABLE_STATUS


@shared_task(name="notifications.deliver_webhook")
def deliver_webhook(delivery_id: str) -> dict:
    """POST one delivery. Never raises; failures are recorded on the log and retried."""
    log = (WebhookDeliveryLog.objects.select_related("subscription")
           .filter(pk=delivery_id).first())
    if log is None or log.delivered_at is not None:
        return {"delivery_id": str(delivery_id), "skipped": True}
    sub = log.subscription
    attempt = log.attempt_count + 1
    body = encode_body(log.payload)
    status_code, error, retry = None, "", False
    if not sub.is_active:
        error = "subscription inactive"
    else:
        try:
            validate_target_url(sub.target_url)
            response = requests.post(
                sub.target_url, data=body, allow_redirects=False,
                timeout=float(getattr(settings, "DVSA_WEBHOOK_TIMEOUT_S", 10)),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "dvsa-webhooks/1",
                    signing.EVENT_HEADER: log.event_type,
                    signing.DELIVERY_HEADER: str(log.id),
                    signing.SIGNATURE_HEADER: signing.sign(sub.secret_key, body),
                })
            status_code = response.status_code
            if not 200 <= status_code < 300:
                error = f"HTTP {status_code}"
                retry = _retryable(status_code)
        except UnsafeWebhookURL as exc:
            error = f"unsafe target: {exc}"
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            retry = True

    delivered = not error
    WebhookDeliveryLog.objects.filter(pk=log.pk).update(
        attempt_count=attempt, response_status_code=status_code, error=error,
        delivered_at=timezone.now() if delivered else None)
    if retry and attempt <= MAX_RETRIES:
        deliver_webhook.apply_async((str(log.id),), countdown=backoff_seconds(attempt))
    elif not delivered:
        logger.info("webhook delivery %s gave up after %s attempt(s): %s", log.id, attempt, error)
    return {"delivery_id": str(log.id), "attempt": attempt, "delivered": delivered,
            "status_code": status_code, "error": error}
