"""Typed exceptions for the reasoning runtime and Azure Foundry integration.

Every failure surfaced to the API layer maps to one of these so responses can
carry a stable ``error_code`` (see :func:`error_payload`). Keeping them in one
tiny module avoids import cycles between the adapter, the session manager and
the router.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ReasoningError(RuntimeError):
    """Base class for all reasoning-runtime errors.

    ``error_code`` is a short, stable machine string; ``details`` is an optional
    JSON-serializable dict with extra context (never secrets).
    """

    error_code = "reasoning_error"
    http_status = 500

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ProvisioningError(ReasoningError):
    """Raised when creating the ephemeral Foundry resources fails."""

    error_code = "provisioning_failed"
    http_status = 502


class TeardownError(ReasoningError):
    """Raised when releasing session resources fails irrecoverably."""

    error_code = "teardown_failed"
    http_status = 502


class SessionNotFoundError(ReasoningError):
    """Raised when a ``session_id`` is unknown or already torn down."""

    error_code = "session_not_found"
    http_status = 404


class ModelUnavailableError(ReasoningError):
    """Raised when the reasoning endpoint cannot serve a request."""

    error_code = "model_unavailable"
    http_status = 503


class AuthError(ReasoningError):
    """Raised when Azure authentication / credential resolution fails."""

    error_code = "auth_error"
    http_status = 401


class CostLimitExceeded(ReasoningError):
    """Raised when a session's hard cost cap would be breached."""

    error_code = "cost_limit_exceeded"
    http_status = 402


class QuotaExceeded(ReasoningError):
    """Raised when global or per-user session quotas are hit."""

    error_code = "quota_exceeded"
    http_status = 429


def error_payload(exc: Exception) -> Dict[str, Any]:
    """Return the structured ``{error_code, message, details}`` API envelope."""
    if isinstance(exc, ReasoningError):
        return {
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        }
    return {
        "error_code": "internal_error",
        "message": str(exc),
        "details": {},
    }
