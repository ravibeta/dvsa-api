"""WebSockets through the real ASGI app: JWT auth, live alerts, flight telemetry groups."""

from unittest import mock

import pytest
from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from channels.layers import get_channel_layer
from channels.testing import WebsocketCommunicator
from rest_framework_simplejwt.tokens import AccessToken

from apps.notifications.broadcast import broadcast_live_alert
from apps.users.models import User

ORIGIN = [(b"origin", b"http://localhost")]
UNAUTHORIZED = 4401


@pytest.fixture
def application():
    from config.asgi import application as app

    return app


@pytest.fixture(autouse=True)
def fresh_layer():
    layer = get_channel_layer()
    async_to_sync(layer.flush)()
    yield
    async_to_sync(layer.flush)()


@pytest.fixture
def users(db):
    return (User.objects.create_user(email="a@example.com", password="pw-123456"),
            User.objects.create_user(email="b@example.com", password="pw-123456"))


def token(user) -> str:
    return str(AccessToken.for_user(user))


def run(coro_fn):
    return async_to_sync(coro_fn)()


def socket(app, path, user=None, subprotocol=False):
    if user is not None and not subprotocol:
        path = f"{path}?token={token(user)}"
    protocols = ["bearer", token(user)] if subprotocol else None
    return WebsocketCommunicator(app, path, headers=ORIGIN, subprotocols=protocols)


@pytest.mark.django_db
class TestAuth:
    @pytest.mark.parametrize("path", ["/ws/alerts/", "/ws/flights/f1/telemetry/"])
    def test_rejects_missing_or_bad_token(self, application, path):
        async def scenario():
            for comm in (WebsocketCommunicator(application, path, headers=ORIGIN),
                         WebsocketCommunicator(application, f"{path}?token=not-a-jwt", headers=ORIGIN)):
                connected, code = await comm.connect()
                assert (connected, code) == (False, UNAUTHORIZED)
        run(scenario)

    def test_query_token_and_ping(self, application, users):
        async def scenario():
            comm = socket(application, "/ws/alerts/", users[0])
            connected, _ = await comm.connect()
            assert connected
            await comm.send_json_to({"type": "ping"})
            assert await comm.receive_json_from(timeout=2) == {"type": "pong"}
            await comm.disconnect()
        run(scenario)

    def test_subprotocol_token(self, application, users):
        async def scenario():
            comm = socket(application, "/ws/alerts/", users[0], subprotocol=True)
            connected, subprotocol = await comm.connect()
            assert connected and subprotocol == "bearer"
            await comm.disconnect()
        run(scenario)

    def test_inactive_user_rejected(self, application, users):
        users[0].is_active = False
        users[0].save()

        async def scenario():
            connected, code = await socket(application, "/ws/alerts/", users[0]).connect()
            assert (connected, code) == (False, UNAUTHORIZED)
        run(scenario)


@pytest.mark.django_db
class TestLiveAlerts:
    def test_broadcast_reaches_only_the_owner(self, application, users):
        alice, bob = users

        async def scenario():
            a = socket(application, "/ws/alerts/", alice)
            b = socket(application, "/ws/alerts/", bob)
            assert (await a.connect())[0] and (await b.connect())[0]
            with mock.patch("apps.notifications.broadcast.queue_webhooks", return_value=[]):
                result = await database_sync_to_async(broadcast_live_alert)(
                    alice.pk, "alert.detection_triggered",
                    {"job_id": "j1", "confidence": 0.9, "frame_timestamp": 3.5,
                     "geojson": {"type": "FeatureCollection", "features": []}})
            assert result["pushed"] is True
            event = await a.receive_json_from(timeout=2)
            assert event["type"] == "alert.detection_triggered" and event["data"]["confidence"] == 0.9
            assert event["id"] == result["event"]["id"]
            assert await b.receive_nothing(timeout=0.2)
            await a.disconnect()
            await b.disconnect()
        run(scenario)


@pytest.mark.django_db
class TestTelemetry:
    def test_rebroadcast_within_owner_flight_group(self, application, users):
        alice, bob = users

        async def scenario():
            drone = socket(application, "/ws/flights/f-7/telemetry/", alice)
            viewer = socket(application, "/ws/flights/f-7/telemetry/", alice)
            stranger = socket(application, "/ws/flights/f-7/telemetry/", bob)   # same id, other owner
            for comm in (drone, viewer, stranger):
                assert (await comm.connect())[0]
            await drone.send_json_to({"type": "telemetry", "lat": 47.6, "lon": -122.3, "alt": 80, "t": 1.5})
            for comm in (drone, viewer):
                event = await comm.receive_json_from(timeout=2)
                assert event["type"] == "telemetry" and event["flight_id"] == "f-7"
                assert event["data"]["lat"] == 47.6
            assert await stranger.receive_nothing(timeout=0.2)
            await drone.send_json_to({"type": "marker", "label": "car", "confidence": 0.8, "lat": 47.6, "lon": -122.3})
            assert (await viewer.receive_json_from(timeout=2))["type"] == "marker"
            for comm in (drone, viewer, stranger):
                await comm.disconnect()
        run(scenario)

    @pytest.mark.parametrize("message", [{"type": "telemetry", "lat": 91, "lon": 0},
                                         {"type": "telemetry", "lat": "x", "lon": 0},
                                         {"type": "shell", "cmd": "rm"}])
    def test_invalid_messages_are_not_broadcast(self, application, users, message):
        async def scenario():
            drone = socket(application, "/ws/flights/f-7/telemetry/", users[0])
            viewer = socket(application, "/ws/flights/f-7/telemetry/", users[0])
            assert (await drone.connect())[0] and (await viewer.connect())[0]
            await drone.send_json_to(message)
            assert (await drone.receive_json_from(timeout=2))["type"] == "error"
            assert await viewer.receive_nothing(timeout=0.2)
            await drone.disconnect()
            await viewer.disconnect()
        run(scenario)

    def test_flight_id_pattern_enforced(self, application, users):
        async def scenario():
            comm = socket(application, "/ws/flights/bad%20id/telemetry/", users[0])
            with pytest.raises(ValueError):
                await comm.connect()
        run(scenario)
