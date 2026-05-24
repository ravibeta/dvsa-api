"""
Production settings for DVSA API project.
"""

import os
from .base import *  # noqa

# Production security settings
DEBUG = False
SECRET_KEY = os.environ.get("SECRET_KEY")
if SECRET_KEY is None:
    raise ValueError("SECRET_KEY environment variable must be set in production")

SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_SECURITY_POLICY = {
    "default-src": ("'self'",),
    "style-src": ("'self'", "'unsafe-inline'"),
    "script-src": ("'self'",),
    "img-src": ("'self'", "data:", "https:"),
}

SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "").split(",")

# Production database - PostgreSQL
import dj_database_url

DATABASES = {
    "default": dj_database_url.config(
        default="postgresql://user:password@localhost/dvsa_db",
        conn_max_age=600,
    )
}

# Email Configuration for Production
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", 587))
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD")
DEFAULT_FROM_EMAIL = EMAIL_HOST_USER

# Sentry Error Tracking
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.django import DjangoIntegration
    from sentry_sdk.integrations.celery import CeleryIntegration

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[
            DjangoIntegration(),
            CeleryIntegration(),
        ],
        traces_sample_rate=0.1,
        send_default_pii=False,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
    )

# REST Framework Production Settings
REST_FRAMEWORK["DEFAULT_RENDERER_CLASSES"] = [
    "rest_framework.renderers.JSONRenderer",
]

# Production logging
LOGGING["loggers"]["django"]["level"] = "INFO"
LOGGING["loggers"]["apps"]["level"] = "INFO"

# Static files configuration
STATICFILES_STORAGE = "django.contrib.staticfiles.storage.ManifestStaticFilesStorage"

# Disable debug toolbar in production
if "debug_toolbar" in INSTALLED_APPS:
    INSTALLED_APPS.remove("debug_toolbar")
if "debug_toolbar.middleware.DebugToolbarMiddleware" in MIDDLEWARE:
    MIDDLEWARE.remove("debug_toolbar.middleware.DebugToolbarMiddleware")

print("✓ Production settings loaded")
