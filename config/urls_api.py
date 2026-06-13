"""
API v1 URL configuration for DVSA API project.

Each Django app owns its routes via its own ``urls.py`` (generic-view based);
this module simply namespaces and includes them under ``/api/v1/``.
"""

from django.urls import include, path

urlpatterns = [
    path("users/", include("apps.users.urls")),
    path("videos/", include("apps.videos.urls")),
    path("analytics/", include("apps.analytics.urls")),
    path("storage/", include("apps.storage.urls")),
]
