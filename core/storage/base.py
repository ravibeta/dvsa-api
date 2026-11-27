"""Cloud-agnostic object storage contract used by the media/jobs features.

Feature code (``apps.media``, ``apps.jobs``) talks only to :class:`ObjectStore`; each public
cloud provides one implementation (``azure_blob`` today, S3 later). Keys are container/bucket
relative paths such as ``{owner_pk}/uploads/{flight}/{media_id}/{filename}``.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class UploadTarget:
    """Where and how a client uploads one object directly to the cloud."""

    url: str
    method: str
    expires_at: datetime.datetime
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    etag: str
    content_type: Optional[str] = None


@dataclass(frozen=True)
class ObjectCreated:
    """A provider "object created" notification, normalised to a key."""

    key: str
    size: Optional[int] = None
    etag: Optional[str] = None


class StorageEventError(ValueError):
    """The provider event payload is malformed."""


@runtime_checkable
class ObjectStore(Protocol):
    provider: str
    bucket: str

    def presign_upload(self, key: str, content_type: str,
                       ttl: datetime.timedelta) -> UploadTarget:
        """A write-only URL scoped to exactly ``key`` (no read/list/delete)."""

    def presign_read(self, key: str, ttl: datetime.timedelta) -> str:
        """A read-only URL for ``key``."""

    def head(self, key: str) -> Optional[ObjectInfo]:
        """Object metadata, or ``None`` when the object does not exist."""

    def read_bytes(self, key: str, max_bytes: int) -> bytes:
        """The object's content (for small sidecars); raises when larger than ``max_bytes``."""

    def parse_created_events(self, body: Any, headers: Dict[str, str]
                             ) -> Tuple[List[ObjectCreated], Optional[Dict[str, Any]]]:
        """Parse a provider push notification.

        Returns ``(events, handshake)``: ``handshake`` is the JSON body that answers a
        subscription-validation request (``None`` for ordinary deliveries).
        """
