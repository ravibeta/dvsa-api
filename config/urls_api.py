"""
API v1 URL configuration for DVSA API project.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

router = DefaultRouter()

# Import viewsets from apps
from apps.users.views import UserViewSet, AuthViewSet
from apps.videos.views import VideoViewSet
from apps.analytics.views import AnalyticsViewSet

# Register viewsets
router.register(r"users", UserViewSet, basename="user")
router.register(r"auth", AuthViewSet, basename="auth")
router.register(r"videos", VideoViewSet, basename="video")
router.register(r"analytics", AnalyticsViewSet, basename="analytics")

urlpatterns = [
    path("", include(router.urls)),
]
