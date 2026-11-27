"""Shared fixtures for the media / jobs / notifications tests (no network, no Azure)."""

from __future__ import annotations

import datetime
import hashlib
from typing import Dict, List, Optional

import pytest

import core.storage as storage
from core.storage import ObjectCreated, ObjectInfo, StorageEventError, UploadTarget


class FakeObjectStore:
    """In-memory :class:`core.storage.base.ObjectStore` (one shared bucket per test)."""

    provider = "azure"
    bucket = "input"

    def __init__(self):
        self.objects: Dict[str, bytes] = {}
        self.upload_requests: List[tuple] = []
        self.read_requests: List[str] = []

    def put(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    def presign_upload(self, key, content_type, ttl):
        self.upload_requests.append((key, content_type, ttl))
        expires = datetime.datetime(2030, 1, 1, tzinfo=datetime.timezone.utc)
        return UploadTarget(url=f"https://fake.example/{self.bucket}/{key}?sp=cw", method="PUT",
                            expires_at=expires,
                            headers={"x-ms-blob-type": "BlockBlob", "Content-Type": content_type})

    def presign_read(self, key, ttl):
        self.read_requests.append(key)
        return f"https://fake.example/{self.bucket}/{key}?sp=r"

    def head(self, key) -> Optional[ObjectInfo]:
        if key not in self.objects:
            return None
        data = self.objects[key]
        return ObjectInfo(key=key, size=len(data), etag=hashlib.md5(data).hexdigest())

    def read_bytes(self, key, max_bytes):
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def parse_created_events(self, body, headers):
        events = body if isinstance(body, list) else [body]
        out = []
        for event in events:
            if not isinstance(event, dict):
                raise StorageEventError("event must be an object")
            if event.get("eventType") == "Microsoft.EventGrid.SubscriptionValidationEvent":
                return [], {"validationResponse": event["data"]["validationCode"]}
            if event.get("eventType") == "Microsoft.Storage.BlobCreated":
                key = event["data"]["url"].split(f"/{self.bucket}/", 1)[1]
                out.append(ObjectCreated(key=key, size=event["data"].get("contentLength")))
        return out, None


@pytest.fixture
def fake_store(monkeypatch):
    store = FakeObjectStore()
    monkeypatch.setitem(storage._PROVIDERS, "azure", lambda: store)
    return store


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every webhook host to a public address (no real DNS)."""
    import ipaddress

    from apps.notifications import url_safety

    def resolve(host, port):
        if host in ("localhost", "127.0.0.1"):
            return {ipaddress.ip_address("127.0.0.1")}
        if host.startswith("internal"):
            return {ipaddress.ip_address("10.0.0.5")}
        return {ipaddress.ip_address("93.184.216.34")}

    monkeypatch.setattr(url_safety, "_resolve", resolve)
