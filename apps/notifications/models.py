"""Outbound webhook subscriptions and their delivery log."""

import secrets
import uuid

from django.conf import settings
from django.db import models


def new_secret() -> str:
    return secrets.token_urlsafe(32)


class WebhookSubscription(models.Model):
    """A user's HTTPS endpoint that receives HMAC-signed event deliveries."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                              related_name="webhook_subscriptions")
    target_url = models.URLField(max_length=1024)
    secret_key = models.CharField(max_length=128, default=new_secret, editable=False)
    events = models.JSONField(default=list)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"WebhookSubscription({self.id}, {self.target_url})"

    def wants(self, event_type: str) -> bool:
        return self.is_active and ("*" in self.events or event_type in self.events)


class WebhookDeliveryLog(models.Model):
    """One event delivery; its id is the ``X-DVSA-Delivery-ID`` (stable across retries)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    subscription = models.ForeignKey(WebhookSubscription, on_delete=models.CASCADE,
                                     related_name="deliveries")
    event_type = models.CharField(max_length=64)
    payload = models.JSONField(default=dict)
    response_status_code = models.IntegerField(null=True, blank=True)
    attempt_count = models.PositiveIntegerField(default=0)
    delivered_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"WebhookDeliveryLog({self.id}, {self.event_type}, attempts={self.attempt_count})"
