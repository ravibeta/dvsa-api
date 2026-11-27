"""JWT authentication for WebSocket connections.

The access token (the same simplejwt token the REST API uses) is read from ``?token=<jwt>``
or from the subprotocol pair ``Sec-WebSocket-Protocol: bearer, <jwt>``; in the latter case
the consumer accepts with subprotocol ``bearer``. Without a valid token ``scope["user"]`` is
anonymous and the consumers close with code 4401.
"""
from __future__ import annotations

from typing import Optional, Tuple
from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser

SUBPROTOCOL = "bearer"


def token_from_scope(scope) -> Tuple[Optional[str], Optional[str]]:
    """``(token, subprotocol_to_accept)``."""
    query = parse_qs((scope.get("query_string") or b"").decode())
    if query.get("token"):
        return query["token"][0], None
    protocols = list(scope.get("subprotocols") or [])
    if len(protocols) >= 2 and protocols[0].lower() == SUBPROTOCOL:
        return protocols[1], SUBPROTOCOL
    return None, None


@database_sync_to_async
def user_for_token(token: str):
    from rest_framework_simplejwt.exceptions import TokenError  # noqa: PLC0415
    from rest_framework_simplejwt.settings import api_settings  # noqa: PLC0415
    from rest_framework_simplejwt.tokens import AccessToken  # noqa: PLC0415

    try:
        claims = AccessToken(token)
        user_id = claims[api_settings.USER_ID_CLAIM]
    except (TokenError, KeyError):
        return AnonymousUser()
    user = get_user_model().objects.filter(pk=user_id, is_active=True).first()
    return user or AnonymousUser()


class JWTAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        scope = dict(scope)
        token, subprotocol = token_from_scope(scope)
        scope["user"] = await user_for_token(token) if token else AnonymousUser()
        scope["dvsa_subprotocol"] = subprotocol
        return await super().__call__(scope, receive, send)
