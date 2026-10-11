"""
ASGI config for DVSA API project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

application = get_asgi_application()

# Same startup-only DVSA_QWEN_ENABLED check as `runserver` (see apps/videos/apps.py).
from core.azure import local_qwen_server  # noqa: E402

local_qwen_server.start_if_enabled()
