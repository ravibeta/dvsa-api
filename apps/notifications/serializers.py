"""Serializers for webhook subscriptions and deliveries."""

from rest_framework import serializers

from .events import ALL, EVENT_TYPES
from .models import WebhookDeliveryLog, WebhookSubscription
from .url_safety import UnsafeWebhookURL, validate_target_url


class WebhookSubscriptionSerializer(serializers.ModelSerializer):
    """The signing secret is never returned here (see :class:`WebhookSubscriptionCreateSerializer`)."""

    class Meta:
        model = WebhookSubscription
        fields = ("id", "target_url", "events", "is_active", "created_at", "updated_at")
        read_only_fields = ("id", "created_at", "updated_at")

    def validate_target_url(self, value):
        try:
            return validate_target_url(value)
        except UnsafeWebhookURL as exc:
            raise serializers.ValidationError(str(exc)) from None

    def validate_events(self, value):
        if not isinstance(value, list) or not value:
            raise serializers.ValidationError("events must be a non-empty list")
        unknown = [e for e in value if e != ALL and e not in EVENT_TYPES]
        if unknown:
            raise serializers.ValidationError(
                f"unknown event type(s): {', '.join(map(str, unknown))}; "
                f"allowed: {', '.join(EVENT_TYPES)} or {ALL!r}")
        return sorted(set(value))


class WebhookSubscriptionCreateSerializer(WebhookSubscriptionSerializer):
    """Create response: the only time ``secret_key`` is shown."""

    class Meta(WebhookSubscriptionSerializer.Meta):
        fields = WebhookSubscriptionSerializer.Meta.fields + ("secret_key",)
        read_only_fields = WebhookSubscriptionSerializer.Meta.read_only_fields + ("secret_key",)


class WebhookDeliveryLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = WebhookDeliveryLog
        fields = ("id", "event_type", "payload", "response_status_code", "attempt_count",
                  "delivered_at", "error", "created_at")
        read_only_fields = fields
