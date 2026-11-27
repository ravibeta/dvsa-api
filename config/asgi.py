"""
ASGI config for DVSA API project.

HTTP is the regular Django app; WebSockets (``ws/alerts/``,
``ws/flights/<flight_id>/telemetry/``) go through Channels with JWT auth.
Serve with ``daphne config.asgi:application``.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

django_asgi_app = get_asgi_application()

# Imported after Django is set up (consumers touch models).
from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402
from channels.security.websocket import AllowedHostsOriginValidator  # noqa: E402

from apps.notifications.routing import websocket_urlpatterns  # noqa: E402
from apps.notifications.ws_auth import JWTAuthMiddleware  # noqa: E402

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AllowedHostsOriginValidator(JWTAuthMiddleware(URLRouter(websocket_urlpatterns))),
})

# Same startup-only DVSA_QWEN_ENABLED check as `runserver` (see apps/videos/apps.py).
from core.azure import local_qwen_server  # noqa: E402

local_qwen_server.start_if_enabled()
