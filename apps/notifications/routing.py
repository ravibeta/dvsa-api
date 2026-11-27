"""WebSocket routes (mounted by ``config/asgi.py``)."""

from django.urls import re_path

from apps.media.models import FLIGHT_ID_PATTERN

from . import consumers

websocket_urlpatterns = [
    re_path(r"^ws/alerts/$", consumers.AlertsConsumer.as_asgi()),
    re_path(rf"^ws/flights/(?P<flight_id>{FLIGHT_ID_PATTERN})/telemetry/$",
            consumers.FlightTelemetryConsumer.as_asgi()),
]
