"""Webhooks: HMAC signing, subscription CRUD + SSRF guard, delivery retries, event filtering."""

from unittest import mock

import pytest
import requests
from django.test import override_settings
from rest_framework.test import APIClient

from apps.notifications import signing, tasks
from apps.notifications.broadcast import broadcast_live_alert
from apps.notifications.models import WebhookDeliveryLog, WebhookSubscription
from apps.notifications.url_safety import UnsafeWebhookURL, validate_target_url
from apps.users.models import User


@pytest.fixture
def user(db):
    return User.objects.create_user(email="hooks@example.com", password="pw-123456")


@pytest.fixture
def client(user):
    c = APIClient()
    c.force_authenticate(user=user)
    return c


@pytest.fixture
def no_channels():
    with mock.patch("apps.notifications.broadcast.push_to_group", return_value=False):
        yield


def subscribe(user, events=("*",), url="https://hooks.example.com/dvsa", active=True):
    return WebhookSubscription.objects.create(owner=user, target_url=url, events=list(events),
                                              is_active=active)


def response(code):
    return mock.Mock(status_code=code)


@pytest.fixture
def countdowns(monkeypatch):
    """Record retry countdowns while still running retries eagerly."""
    seen = []
    original = tasks.deliver_webhook.apply_async

    def apply_async(args=None, kwargs=None, countdown=None, **kw):
        if countdown is not None:
            seen.append(countdown)
        return original(args, kwargs, **kw)

    monkeypatch.setattr(tasks.deliver_webhook, "apply_async", apply_async)
    return seen


# ------------------------------------------------------------------------ signing
class TestSigning:
    def test_round_trip(self):
        header = signing.sign("sek", b'{"a":1}', timestamp=1_700_000_000)
        assert header.startswith("t=1700000000,v1=")
        assert signing.verify_signature("sek", b'{"a":1}', header, now=1_700_000_010)

    @pytest.mark.parametrize("secret,body,now", [
        ("other", b'{"a":1}', 1_700_000_010),       # wrong secret
        ("sek", b'{"a":2}', 1_700_000_010),         # tampered body
        ("sek", b'{"a":1}', 1_700_000_900),         # replayed outside tolerance
    ])
    def test_rejects(self, secret, body, now):
        header = signing.sign("sek", b'{"a":1}', timestamp=1_700_000_000)
        assert not signing.verify_signature(secret, body, header, now=now)

    def test_malformed_header(self):
        assert not signing.verify_signature("sek", b"{}", "garbage")
        assert not signing.verify_signature("sek", b"{}", "t=abc,v1=00")


# -------------------------------------------------------------------- URL safety
class TestUrlSafety:
    def test_public_https_ok(self, public_dns):
        assert validate_target_url("https://hooks.example.com/x")

    @pytest.mark.parametrize("url", ["http://hooks.example.com/x", "https://internal.corp/x",
                                     "https://localhost/x", "ftp://hooks.example.com/x",
                                     "https://user:pw@hooks.example.com/x", "not a url"])
    def test_rejected(self, public_dns, url):
        with pytest.raises(UnsafeWebhookURL):
            validate_target_url(url)

    @override_settings(DVSA_WEBHOOK_ALLOW_INSECURE_LOCALHOST="true")
    def test_localhost_allowed_in_dev(self, public_dns):
        assert validate_target_url("http://localhost:9000/hook")
        with pytest.raises(UnsafeWebhookURL):
            validate_target_url("http://internal.corp/x")    # still only loopback


# --------------------------------------------------------------------------- CRUD
@pytest.mark.django_db
class TestSubscriptionsAPI:
    def test_create_returns_secret_once(self, client, user, public_dns):
        res = client.post("/api/v1/webhooks/", {"target_url": "https://hooks.example.com/dvsa",
                                                "events": ["job.completed", "job.failed"]}, format="json")
        assert res.status_code == 201, res.content
        body = res.json()
        assert len(body["secret_key"]) >= 40 and body["events"] == ["job.completed", "job.failed"]
        sub = WebhookSubscription.objects.get(pk=body["id"])
        assert sub.owner == user and sub.secret_key == body["secret_key"]
        assert "secret_key" not in client.get(f"/api/v1/webhooks/{sub.id}/").json()
        assert "secret_key" not in client.get("/api/v1/webhooks/").json()["results"][0]

    @pytest.mark.parametrize("payload,field", [
        ({"target_url": "http://hooks.example.com/x", "events": ["*"]}, "target_url"),
        ({"target_url": "https://internal.corp/x", "events": ["*"]}, "target_url"),
        ({"target_url": "https://hooks.example.com/x", "events": ["job.exploded"]}, "events"),
        ({"target_url": "https://hooks.example.com/x", "events": []}, "events"),
    ])
    def test_validation(self, client, public_dns, payload, field):
        res = client.post("/api/v1/webhooks/", payload, format="json")
        assert res.status_code == 400 and field in res.json()

    def test_patch_delete_and_isolation(self, client, user, public_dns):
        sub = subscribe(user)
        res = client.patch(f"/api/v1/webhooks/{sub.id}/", {"is_active": False, "events": ["ping"]},
                           format="json")
        assert res.status_code == 200 and res.json()["is_active"] is False
        assert client.patch(f"/api/v1/webhooks/{sub.id}/", {"target_url": "http://x.example.com"},
                            format="json").status_code == 400
        other = APIClient()
        other.force_authenticate(User.objects.create_user(email="z@example.com", password="pw-123456"))
        assert other.get(f"/api/v1/webhooks/{sub.id}/").status_code == 404
        assert other.delete(f"/api/v1/webhooks/{sub.id}/").status_code == 404
        assert client.delete(f"/api/v1/webhooks/{sub.id}/").status_code == 204

    def test_test_endpoint_sends_signed_ping(self, client, user, public_dns):
        sub = subscribe(user, events=["job.completed"])
        with mock.patch("requests.post", return_value=response(200)) as post:
            res = client.post(f"/api/v1/webhooks/{sub.id}/test/")
        assert res.status_code == 202
        headers = post.call_args.kwargs["headers"]
        assert headers["X-DVSA-Event"] == "ping"
        assert headers["X-DVSA-Delivery-ID"] == res.json()["delivery_id"]
        assert signing.verify_signature(sub.secret_key, post.call_args.kwargs["data"],
                                        headers["X-DVSA-Signature"])
        assert post.call_args.kwargs["allow_redirects"] is False
        deliveries = client.get(f"/api/v1/webhooks/{sub.id}/deliveries/").json()["results"]
        assert deliveries[0]["attempt_count"] == 1 and deliveries[0]["response_status_code"] == 200


# ----------------------------------------------------------------------- delivery
@pytest.mark.django_db
class TestDelivery:
    def test_signed_body_matches_payload(self, user, public_dns, no_channels):
        sub = subscribe(user)
        with mock.patch("requests.post", return_value=response(204)) as post:
            out = broadcast_live_alert(user.pk, "job.completed", {"job_id": "j1"})
        body = post.call_args.kwargs["data"]
        assert body == tasks.encode_body(out["event"])
        assert signing.verify_signature(sub.secret_key, body, post.call_args.kwargs["headers"]["X-DVSA-Signature"])
        log = WebhookDeliveryLog.objects.get()
        assert log.delivered_at and log.attempt_count == 1 and str(log.id) == out["deliveries"][0]

    def test_retries_with_exponential_backoff_then_gives_up(self, user, public_dns, no_channels, countdowns):
        subscribe(user)
        with mock.patch("requests.post", return_value=response(503)) as post:
            broadcast_live_alert(user.pk, "job.failed", {})
        assert post.call_count == 4                       # 1 attempt + 3 retries
        assert countdowns == [10, 20, 40]
        log = WebhookDeliveryLog.objects.get()
        assert log.attempt_count == 4 and log.delivered_at is None and log.error == "HTTP 503"
        delivery_ids = {c.kwargs["headers"]["X-DVSA-Delivery-ID"] for c in post.call_args_list}
        assert delivery_ids == {str(log.id)}               # stable across retries

    def test_recovers_after_transient_failure(self, user, public_dns, no_channels):
        subscribe(user)
        with mock.patch("requests.post", side_effect=[requests.ConnectionError("reset"), response(200)]):
            broadcast_live_alert(user.pk, "job.completed", {})
        log = WebhookDeliveryLog.objects.get()
        assert log.attempt_count == 2 and log.delivered_at and log.error == ""

    def test_client_errors_are_not_retried(self, user, public_dns, no_channels):
        subscribe(user)
        with mock.patch("requests.post", return_value=response(404)) as post:
            broadcast_live_alert(user.pk, "job.completed", {})
        assert post.call_count == 1

    def test_target_rechecked_at_send_time(self, user, public_dns, no_channels):
        subscribe(user, url="https://internal.example/x")  # passed checks earlier; now resolves private
        with mock.patch("apps.notifications.url_safety._resolve",
                        return_value={__import__("ipaddress").ip_address("10.1.2.3")}), \
                mock.patch("requests.post") as post:
            broadcast_live_alert(user.pk, "job.completed", {})
        post.assert_not_called()
        assert WebhookDeliveryLog.objects.get().error.startswith("unsafe target")

    def test_event_filtering(self, user, public_dns, no_channels):
        wanted = subscribe(user, events=["job.completed"])
        everything = subscribe(user, events=["*"])
        subscribe(user, events=["job.failed"])
        subscribe(user, events=["*"], active=False)
        subscribe(User.objects.create_user(email="q@example.com", password="pw-123456"))
        with mock.patch("requests.post", return_value=response(200)):
            out = broadcast_live_alert(user.pk, "job.completed", {})
        targets = set(WebhookDeliveryLog.objects.filter(pk__in=out["deliveries"])
                      .values_list("subscription_id", flat=True))
        assert targets == {wanted.id, everything.id}
