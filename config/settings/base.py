"""
Base settings for DVSA API project.
This file contains common settings for all environments.
"""

import os
from pathlib import Path
from datetime import timedelta

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
APPS_DIR = BASE_DIR / "apps"

# Security
SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-development-key")
DEBUG = os.environ.get("DEBUG", "False") == "True"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# Application definition
DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

THIRD_PARTY_APPS = [
    "rest_framework",
    "rest_framework.authtoken",
    "corsheaders",
    "django_filters",
    "drf_spectacular",
    "django_extensions",
]

LOCAL_APPS = [
    "apps.users.apps.UsersConfig",
    "apps.videos.apps.VideosConfig",
    "apps.analytics.apps.AnalyticsConfig",
    "apps.storage.apps.StorageConfig",
    "apps.observability.apps.ObservabilityConfig",
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# Custom user model (email-based auth — see apps/users/models.py).
AUTH_USER_MODEL = "users.User"

# Middleware
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# Database
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {"min_length": 10},
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

# Media files
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# REST Framework Configuration
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "rest_framework.authentication.TokenAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_PAGINATION_CLASS": "core.pagination.StandardResultsSetPagination",
    "PAGE_SIZE": int(os.environ.get("API_PAGINATION_SIZE", 50)),
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/hour",
        "user": "1000/hour",
    },
    "EXCEPTION_HANDLER": "core.exceptions.exception_handler",
}

# JWT Configuration
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(
        hours=int(os.environ.get("JWT_EXPIRATION_HOURS", 24))
    ),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "ALGORITHM": os.environ.get("JWT_ALGORITHM", "HS256"),
    "SIGNING_KEY": os.environ.get("JWT_SECRET_KEY", SECRET_KEY),
    "AUTH_HEADER_TYPES": ("Bearer",),
}

# CORS Configuration
CORS_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000"
).split(",")

CORS_ALLOW_CREDENTIALS = True

# =========================================================================
# Azure configuration contract
# -------------------------------------------------------------------------
# Single source of truth for the session-scoped Azure environment provisioned
# by ``core.azure`` (see core/azure/README.md). Everything is env-driven so the
# same code runs locally (dry-run), in CI, and against a real subscription.
# =========================================================================

# Control-plane (ARM) credentials & placement — needed only to *provision*
# global resources. When absent, core.azure falls back to dry-run mode.
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "rg-dvsa")
AZURE_LOCATION = os.environ.get("AZURE_LOCATION", "eastus")
# Provisioning backend: "auto" (real if creds present else dry-run) | "sdk" |
# "terraform" | "dryrun".
AZURE_PROVISIONER = os.environ.get("AZURE_PROVISIONER", "auto")

# Azure Blob Storage (data lake for aerial frames + derived artifacts)
AZURE_ACCOUNT_NAME = os.environ.get("AZURE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.environ.get("AZURE_ACCOUNT_KEY")
AZURE_CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "videos")
# Dedicated drone-video account required by the session environment.
AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT", "sadronevideo")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_INPUT_CONTAINER = os.environ.get("AZURE_STORAGE_INPUT_CONTAINER", "input")
AZURE_STORAGE_OUTPUT_CONTAINER = os.environ.get(
    "AZURE_STORAGE_OUTPUT_CONTAINER", "output"
)

# Azure AI Vision
AZURE_AI_VISION_API_KEY = os.environ.get("AZURE_AI_VISION_API_KEY")
AZURE_AI_VISION_REGION = os.environ.get("AZURE_AI_VISION_REGION", "eastus")
AZURE_AI_VISION_ENDPOINT = os.environ.get("AZURE_AI_VISION_ENDPOINT")

# Azure AI Search
AZURE_SEARCH_SERVICE_NAME = os.environ.get("AZURE_SEARCH_SERVICE_NAME")
AZURE_SEARCH_ENDPOINT = os.environ.get(
    "AZURE_SEARCH_ENDPOINT",
    (f"https://{AZURE_SEARCH_SERVICE_NAME}.search.windows.net"
     if AZURE_SEARCH_SERVICE_NAME else None),
)
AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "dvsa-index")
AZURE_SEARCH_SKU = os.environ.get("AZURE_SEARCH_SKU", "standard")
# Embedding width stored per aerial frame (text-embedding-ada-002 == 1536).
AZURE_SEARCH_VECTOR_DIMENSIONS = int(
    os.environ.get("AZURE_SEARCH_VECTOR_DIMENSIONS", 1536)
)

# Azure OpenAI / AI Foundry (models, agents, deployments)
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ACCOUNT = os.environ.get("AZURE_OPENAI_ACCOUNT", "dvsa-foundry")
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_GPT_MODEL = os.environ.get("AZURE_OPENAI_GPT_MODEL", "gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)
AZURE_OPENAI_EMBEDDING_MODEL = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"
)

# Per-session isolation strategy: "index" (one search index per session) |
# "filter" (shared index, rows tagged + filtered by session/user).
AZURE_SESSION_ISOLATION = os.environ.get("AZURE_SESSION_ISOLATION", "index")
# Path to the Terraform module used by the "terraform" provisioner.
AZURE_TERRAFORM_DIR = os.environ.get(
    "AZURE_TERRAFORM_DIR", str(BASE_DIR / "solution-accelerator")
)

# --- Runtime data-plane (ported from ezvision my_droneworld_api) ----------
# API versions for the Vision / Search / embedding model REST calls.
AZURE_VISION_API_VERSION = os.environ.get("AZURE_VISION_API_VERSION", "2024-02-01")
AZURE_SEARCH_API_VERSION = os.environ.get(
    "AZURE_SEARCH_API_VERSION", "2025-08-01-preview"
)
AZURE_VISION_MODEL_VERSION = os.environ.get("AZURE_VISION_MODEL_VERSION", "2023-04-15")

# Azure AI Foundry project (agents/threads live here).
AZURE_PROJECT_ENDPOINT = os.environ.get("AZURE_PROJECT_ENDPOINT")
AZURE_PROJECT_API_KEY = os.environ.get("AZURE_PROJECT_API_KEY")
AZURE_AGENT_MODEL = os.environ.get("AZURE_AGENT_MODEL", "gpt-4o-mini")
AZURE_EMBEDDING_MODEL = os.environ.get("AZURE_EMBEDDING_MODEL", "text-embedding-ada-002")
# AI Search resource coordinates used by the agentic retrieval pipeline.
AZURE_SEARCH_CONNECTION_ID = os.environ.get("AZURE_SEARCH_CONNECTION_ID")
AZURE_SEARCH_RESOURCE_ID = os.environ.get("AZURE_SEARCH_RESOURCE_ID")
AZURE_SEARCH_SUBSCRIPTION = os.environ.get("AZURE_SEARCH_SUBSCRIPTION")
AZURE_SEARCH_RESOURCE_GROUP = os.environ.get("AZURE_SEARCH_RESOURCE_GROUP")
AZURE_SEARCH_LOCATION = os.environ.get("AZURE_SEARCH_LOCATION", "eastus")
# Foundry agent role names.
AZURE_FN_AGENT_NAME = os.environ.get("AZURE_FN_AGENT_NAME", "fn-agent-in-a-team")
AZURE_CHAT_AGENT_NAME = os.environ.get("AZURE_CHAT_AGENT_NAME", "chat-agent-in-a-team")
AZURE_SEARCH_AGENT_NAME = os.environ.get(
    "AZURE_SEARCH_AGENT_NAME", "search-agent-in-a-team"
)
AZURE_TOOL_AGENT_NAME = os.environ.get("AZURE_TOOL_AGENT_NAME", "tool-agent-in-a-team")

# Azure Video Indexer.
AZURE_VIDEO_INDEXER_URL = os.environ.get(
    "AZURE_VIDEO_INDEXER_URL", "https://api.videoindexer.ai"
)
AZURE_VIDEO_INDEXER_REGION = os.environ.get("AZURE_VIDEO_INDEXER_REGION", "eastus")
AZURE_VIDEO_INDEXER_ACCOUNT = os.environ.get("AZURE_VIDEO_INDEXER_ACCOUNT")
AZURE_VIDEO_INDEXER_API_KEY = os.environ.get("AZURE_VIDEO_INDEXER_API_KEY")
AZURE_VIDEO_INDEXER_ACCESS_TOKEN = os.environ.get("AZURE_VIDEO_INDEXER_ACCESS_TOKEN", "")

# Perplexity (image geolocation + multimodal retrieval fallback).
PERPLEXITY_CHAT_API_KEY = os.environ.get("PERPLEXITY_CHAT_API_KEY")
PERPLEXITY_CHAT_API_URL = os.environ.get(
    "PERPLEXITY_CHAT_API_URL", "https://api.perplexity.ai/chat/completions"
)
PERPLEXITY_GEO_API_KEY = os.environ.get("PERPLEXITY_GEO_API_KEY")
PERPLEXITY_GEO_API_URL = os.environ.get(
    "PERPLEXITY_GEO_API_URL", "https://api.perplexity.ai/v1/image/geolocation"
)

# Sample object/scene image URIs used by object-in-scene search.
SAMPLE_OBJECT_URI = os.environ.get("SAMPLE_OBJECT_URI", "")
SAMPLE_SCENE_URI = os.environ.get("SAMPLE_SCENE_URI", "")

# Celery Configuration
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"

# Logging Configuration
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{levelname}] {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        },
        "simple": {
            "format": "[{levelname}] {asctime} {module} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": BASE_DIR / "logs" / "dvsa.log",
            "maxBytes": 1024 * 1024 * 10,  # 10MB
            "backupCount": 10,
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console", "file"],
            "level": os.environ.get("LOG_LEVEL", "INFO"),
            "propagate": False,
        },
        "apps": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

# Create logs directory
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# API Configuration
API_TIMEOUT_SECONDS = int(os.environ.get("API_TIMEOUT_SECONDS", 30))
MAX_VIDEO_UPLOAD_SIZE = int(os.environ.get("MAX_VIDEO_UPLOAD_SIZE", 1073741824))  # 1GB
ALLOWED_VIDEO_FORMATS = os.environ.get("ALLOWED_VIDEO_FORMATS", "mp4,avi,mov,mkv").split(",")

# Geospatial Configuration
GEOSPATIAL_ENABLED = os.environ.get("GEOSPATIAL_ENABLED", "True") == "True"
GPS_PRECISION_METERS = int(os.environ.get("GPS_PRECISION_METERS", 5))

# Commentary-driven Observability Layer
# Parallel telemetry layer that turns vision-routine output into wide,
# query-time-aggregatable commentary events. Disabled by default so existing
# vision runs are unaffected until explicitly switched on.
COMMENTARY_ENABLED = os.environ.get("COMMENTARY_ENABLED", "False") == "True"
# Sink for routine-generated commentary: "null" | "db" | "memory" | "otel".
# Comma-separated values fan out to all (e.g. "db,otel").
COMMENTARY_SINK = os.environ.get("COMMENTARY_SINK", "db")
# Commentary generator: "template" (deterministic, no LLM) | "vlm" (model-backed).
COMMENTARY_COMMENTATOR = os.environ.get("COMMENTARY_COMMENTATOR", "template")
# LLM backend for the "vlm" commentator and the semantic agent (Phase 4):
# "echo" (deterministic, offline) | "azure" (Azure OpenAI via AZURE_OPENAI_*).
COMMENTARY_LLM = os.environ.get("COMMENTARY_LLM", "echo")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

# MELT / OpenTelemetry export (Phase 3). Commentary -> Logs, derived metrics ->
# Metrics, routine spans -> Traces, shipped over OTLP/HTTP+JSON to any collector.
# Base endpoint (no signal suffix), e.g. "http://localhost:4318".
OTEL_EXPORTER_OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "dvsa-api")
COMMENTARY_OTEL_SIGNALS = os.environ.get("COMMENTARY_OTEL_SIGNALS", "logs,metrics,traces")
