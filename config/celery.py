"""Celery application for the DVSA API.

Reads broker / result-backend configuration from Django settings (the
``CELERY_*`` keys already defined in ``config/settings/base.py``) and
autodiscovers ``tasks.py`` modules in every installed app.
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

app = Celery("dvsa")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):  # pragma: no cover - diagnostic helper
    print(f"Request: {self.request!r}")
