"""Azure Blob Storage implementation of :class:`core.storage.base.ObjectStore`.

Reads the existing :class:`core.azure.config.AzureEnvironmentConfig` (account, key, input
container) and calls ``azure-storage-blob`` directly; it neither changes nor depends on the
helpers in :mod:`core.azure.blob`.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

from .base import ObjectCreated, ObjectInfo, StorageEventError, UploadTarget

BLOB_CREATED = "Microsoft.Storage.BlobCreated"
VALIDATION_EVENT = "Microsoft.EventGrid.SubscriptionValidationEvent"


def account_key_from_connection_string(conn: Optional[str]) -> Optional[str]:
    """``AccountKey=...`` from a connection string (the base64 key may itself end in ``=``)."""
    for part in (conn or "").split(";"):
        name, sep, value = part.partition("=")
        if sep and name.strip().lower() == "accountkey":
            return value.strip() or None
    return None


class AzureBlobObjectStore:
    provider = "azure"

    def __init__(self, account_name: str, account_key: Optional[str], container: str,
                 connection_string: Optional[str] = None):
        self.account_name = account_name
        self.account_key = account_key or account_key_from_connection_string(connection_string)
        self.connection_string = connection_string
        self.bucket = container

    @classmethod
    def from_settings(cls) -> "AzureBlobObjectStore":
        from core.azure.config import AzureEnvironmentConfig  # noqa: PLC0415

        cfg = AzureEnvironmentConfig.from_settings()
        return cls(cfg.storage_account, cfg.account_key, cfg.input_container,
                   cfg.storage_connection_string)

    @property
    def _host(self) -> str:
        return f"{self.account_name}.blob.core.windows.net"

    def _blob_url(self, key: str) -> str:
        return f"https://{self._host}/{self.bucket}/{quote(key, safe='/')}"

    def _sas(self, key: str, expiry: datetime.datetime, **perms) -> str:
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas  # noqa: PLC0415

        if not self.account_key:
            raise RuntimeError("an Azure storage account key is required to sign URLs")
        return generate_blob_sas(
            account_name=self.account_name, container_name=self.bucket, blob_name=key,
            account_key=self.account_key, permission=BlobSasPermissions(**perms),
            expiry=expiry, protocol="https")

    def presign_upload(self, key: str, content_type: str,
                       ttl: datetime.timedelta) -> UploadTarget:
        expires = datetime.datetime.now(datetime.timezone.utc) + ttl
        # Create + write only: the holder can upload this one blob, not read, list or delete.
        token = self._sas(key, expires, create=True, write=True)
        headers = {"x-ms-blob-type": "BlockBlob"}
        if content_type:
            headers["Content-Type"] = content_type
        return UploadTarget(url=f"{self._blob_url(key)}?{token}", method="PUT",
                            expires_at=expires, headers=headers)

    def presign_read(self, key: str, ttl: datetime.timedelta) -> str:
        expires = datetime.datetime.now(datetime.timezone.utc) + ttl
        return f"{self._blob_url(key)}?{self._sas(key, expires, read=True)}"

    def _client(self, key: str):
        from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

        if self.connection_string:
            service = BlobServiceClient.from_connection_string(self.connection_string)
        else:
            service = BlobServiceClient(account_url=f"https://{self._host}",
                                        credential=self.account_key)
        return service.get_blob_client(container=self.bucket, blob=key)

    def head(self, key: str) -> Optional[ObjectInfo]:
        from azure.core.exceptions import ResourceNotFoundError  # noqa: PLC0415

        try:
            props = self._client(key).get_blob_properties()
        except ResourceNotFoundError:
            return None
        content = getattr(props, "content_settings", None)
        return ObjectInfo(key=key, size=int(props.size), etag=str(props.etag or "").strip('"'),
                          content_type=getattr(content, "content_type", None))

    def read_bytes(self, key: str, max_bytes: int) -> bytes:
        info = self.head(key)
        if info is None:
            raise FileNotFoundError(key)
        if info.size > max_bytes:
            raise ValueError(f"object is {info.size} bytes; limit is {max_bytes}")
        return self._client(key).download_blob().readall()

    def key_from_url(self, url: str) -> Optional[str]:
        """Container-relative key for a blob URL on this account/container, else ``None``."""
        parsed = urlparse(url or "")
        prefix = f"/{self.bucket}/"
        if parsed.netloc.lower() != self._host.lower() or not parsed.path.startswith(prefix):
            return None
        return unquote(parsed.path[len(prefix):])

    def parse_created_events(self, body: Any, headers: Dict[str, str]
                             ) -> Tuple[List[ObjectCreated], Optional[Dict[str, Any]]]:
        """Event Grid (event-grid schema) ``BlobCreated`` deliveries and the validation handshake."""
        events = body if isinstance(body, list) else [body]
        out: List[ObjectCreated] = []
        for event in events:
            if not isinstance(event, dict):
                raise StorageEventError("event must be an object")
            etype = event.get("eventType") or event.get("type")
            data = event.get("data") or {}
            if etype == VALIDATION_EVENT:
                code = data.get("validationCode")
                if not code:
                    raise StorageEventError("validation event without validationCode")
                return [], {"validationResponse": code}
            if etype != BLOB_CREATED:
                continue
            key = self.key_from_url(data.get("url", ""))
            if key is None:
                continue
            size = data.get("contentLength")
            out.append(ObjectCreated(key=key, size=int(size) if size is not None else None,
                                     etag=(data.get("eTag") or "").strip('"') or None))
        return out, None
