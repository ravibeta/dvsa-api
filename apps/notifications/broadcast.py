"""Fan one event out to the owner's WebSockets and matching webhook subscriptions."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from django.core.serializers.json import DjangoJSONEncoder
from django.utils import timezone

from .events import user_group

logger = logging.getLogger("apps.notifications")


def make_envelope(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    envelope = {"id": str(uuid.uuid4()), "type": event_type,
                "created_at": timezone.now().isoformat(), "data": data or {}}
    # Normalise UUIDs/datetimes/decimals so the same dict is valid for Channels and JSON bodies.
    return json.loads(json.dumps(envelope, cls=DjangoJSONEncoder))


def push_to_group(group: str, envelope: Dict[str, Any]) -> bool:
    """Send to a Channels group from sync code. Returns False (and logs) when no layer works."""
    try:
        from asgiref.sync import async_to_sync  # noqa: PLC0415
        from channels.layers import get_channel_layer  # noqa: PLC0415

        layer = get_channel_layer()
        if layer is None:
            return False
        async_to_sync(layer.group_send)(group, {"type": "dvsa.event", "event": envelope})
        return True
    except Exception as exc:  # noqa: BLE001 - live push is best-effort
        logger.warning("channel push to %s failed: %s", group, exc)
        return False


def queue_delivery(subscription, envelope: Dict[str, Any]) -> str:
    from .models import WebhookDeliveryLog  # noqa: PLC0415
    from .tasks import deliver_webhook  # noqa: PLC0415

    log = WebhookDeliveryLog.objects.create(subscription=subscription,
                                            event_type=envelope["type"], payload=envelope)
    deliver_webhook.delay(str(log.id))
    return str(log.id)


def queue_webhooks(owner_id, envelope: Dict[str, Any]) -> List[str]:
    from .models import WebhookSubscription  # noqa: PLC0415

    ids = []
    for sub in WebhookSubscription.objects.filter(owner_id=owner_id, is_active=True):
        if sub.wants(envelope["type"]):
            ids.append(queue_delivery(sub, envelope))
    return ids


def broadcast_live_alert(owner_id, event_type: str, payload: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """Push ``event_type`` to ``ws/alerts/`` for ``owner_id`` and queue matching webhooks.

    ``owner_id`` is the Django ``User`` pk (the UI's ``account_id``). Never raises.
    """
    envelope = make_envelope(event_type, payload or {})
    pushed = push_to_group(user_group(owner_id), envelope)
    try:
        deliveries = queue_webhooks(owner_id, envelope)
    except Exception as exc:  # noqa: BLE001 - alerts must never break the caller
        logger.warning("queueing webhooks for %s failed: %s", event_type, exc)
        deliveries = []
    return {"event": envelope, "pushed": pushed, "deliveries": deliveries}
