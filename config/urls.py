"""
URL configuration for DVSA API project.
"""

from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView

urlpatterns = [
    # Admin
    path("admin/", admin.site.urls),
    
    # API v1
    path("api/v1/", include("config.urls_api")),

    # Reasoning-model integrations (Azure Foundry session lifecycle).
    path("api/reasoning/foundry/",
         include("dvsa_api.api.reasoning_foundry_router")),

    # API Documentation
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),
]

# Add debug toolbar in development
try:
    import debug_toolbar
    urlpatterns.insert(0, path("__debug__/", include(debug_toolbar.urls)))
except ImportError:
    pass
