"""WebSocket consumers: per-user live alerts and per-flight telemetry."""

from __future__ import annotations

import math
from typing import List

from channels.generic.websocket import AsyncJsonWebsocketConsumer

from .events import flight_group, user_group

UNAUTHORIZED = 4401
BAD_MESSAGE = 4400


class _AuthenticatedConsumer(AsyncJsonWebsocketConsumer):
    """Joins ``groups_for(user)`` on connect; relays ``dvsa.event`` group messages."""

    group_names: List[str] = []

    def groups_for(self, user) -> List[str]:
        raise NotImplementedError

    async def connect(self):
        user = self.scope.get("user")
        if user is None or not user.is_authenticated:
            await self.close(code=UNAUTHORIZED)
            return
        self.group_names = self.groups_for(user)
        for group in self.group_names:
            await self.channel_layer.group_add(group, self.channel_name)
        await self.accept(subprotocol=self.scope.get("dvsa_subprotocol"))

    async def disconnect(self, code):
        for group in self.group_names:
            await self.channel_layer.group_discard(group, self.channel_name)

    async def dvsa_event(self, message):
        await self.send_json(message["event"])


class AlertsConsumer(_AuthenticatedConsumer):
    """``ws/alerts/``: job and detection events for the connected user."""

    def groups_for(self, user):
        return [user_group(user.pk)]

    async def receive_json(self, content, **kwargs):
        if isinstance(content, dict) and content.get("type") == "ping":
            await self.send_json({"type": "pong"})


def _number(value, lo=None, hi=None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def valid_telemetry(content) -> bool:
    if not isinstance(content, dict):
        return False
    kind = content.get("type")
    if kind == "telemetry":
        return (_number(content.get("lat"), -90, 90) and _number(content.get("lon"), -180, 180)
                and all(_number(content[k]) for k in ("alt", "t", "heading", "speed")
                        if k in content))
    if kind == "marker":
        return isinstance(content.get("label", ""), str) and all(
            _number(content[k]) for k in ("lat", "lon", "confidence", "t") if k in content)
    return False


class FlightTelemetryConsumer(_AuthenticatedConsumer):
    """``ws/flights/{flight_id}/telemetry/``: re-broadcasts telemetry/markers to the flight group."""

    def groups_for(self, user):
        self.flight_id = self.scope["url_route"]["kwargs"]["flight_id"]
        return [flight_group(user.pk, self.flight_id)]

    async def receive_json(self, content, **kwargs):
        if not valid_telemetry(content):
            await self.send_json({"type": "error", "error": "expected a telemetry or marker message"})
            return
        event = {"type": content["type"], "flight_id": self.flight_id, "data": content}
        await self.channel_layer.group_send(self.group_names[0],
                                            {"type": "dvsa.event", "event": event})
