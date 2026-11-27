"""Webhook target validation (SSRF guard).

Targets must be ``https`` and resolve only to public addresses. With
``DVSA_WEBHOOK_ALLOW_INSECURE_LOCALHOST`` (on in development) ``http``/``https`` to
``localhost`` / loopback is also allowed so a local receiver can be used.
Checked on create/update and again immediately before each delivery.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from django.conf import settings


class UnsafeWebhookURL(ValueError):
    pass


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _resolve(host: str, port: int):
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeWebhookURL(f"cannot resolve host {host!r}") from exc
    return {ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos}


def _is_public(ip) -> bool:
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified)


def validate_target_url(url: str) -> str:
    """Return ``url`` when it is an acceptable webhook target, else raise :class:`UnsafeWebhookURL`."""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("https", "http") or not host:
        raise UnsafeWebhookURL("target_url must be an absolute https URL")
    if parsed.username or parsed.password:
        raise UnsafeWebhookURL("target_url must not contain credentials")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = _resolve(host, port)
    allow_local = _truthy(getattr(settings, "DVSA_WEBHOOK_ALLOW_INSECURE_LOCALHOST", False))
    if allow_local and addresses and all(ip.is_loopback for ip in addresses):
        return url
    if parsed.scheme != "https":
        raise UnsafeWebhookURL("target_url must use https")
    if not addresses or not all(_is_public(ip) for ip in addresses):
        raise UnsafeWebhookURL("target_url must resolve to a public address")
    return url
