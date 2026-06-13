"""DVSA project package.

Ensure the Celery app is imported when Django starts so that the ``@shared_task``
decorator and ``autodiscover_tasks`` work as expected.
"""

from .celery import app as celery_app

__all__ = ("celery_app",)
