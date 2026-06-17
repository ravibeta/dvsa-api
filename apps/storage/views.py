"""Storage views."""

import logging

from rest_framework import permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from core.azure import (
    create_session_azure_environment,
    teardown_session_azure_environment,
)

from .models import StorageConfiguration

logger = logging.getLogger("apps.storage")


class StorageViewSet(viewsets.ReadOnlyModelViewSet):
    """Storage configuration viewset."""
    queryset = StorageConfiguration.objects.all()
    permission_classes = [permissions.IsAdminUser]


def _session_id(request) -> str:
    """Resolve a stable session id: explicit value, Django session, or user."""
    explicit = request.data.get("session_id") or request.query_params.get("session_id")
    if explicit:
        return str(explicit)
    if getattr(request, "session", None) and request.session.session_key:
        return request.session.session_key
    return f"user-{request.user.pk}"


class AzureSessionView(APIView):
    """Setup / teardown the per-session Azure environment.

    ``POST``   provisions (or attaches to) the session's Azure resources and
               returns the environment summary.
    ``DELETE`` tears the session's resources down.

    Body / query params (all optional):
      - ``session_id``       stable id (defaults to Django session key / user).
      - ``provision_global`` bool — also ensure shared resources exist (POST).
      - ``isolation``        "index" | "filter" (POST).
      - ``mode``             "sdk" | "terraform" | "dryrun" (override backend).
      - ``delete_global``    bool — also delete shared resources (DELETE).
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        session_id = _session_id(request)
        try:
            env = create_session_azure_environment(
                session_id,
                user_id=request.user.pk,
                provision_global=bool(request.data.get("provision_global", False)),
                isolation=request.data.get("isolation"),
                mode=request.data.get("mode"),
            )
        except Exception as exc:  # noqa: BLE001 - surface as 502, log details
            logger.exception("azure session setup failed for %s", session_id)
            return Response(
                {"detail": f"Azure setup failed: {exc}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        return Response(env.to_dict(), status=status.HTTP_201_CREATED)

    def delete(self, request):
        session_id = _session_id(request)
        try:
            result = teardown_session_azure_environment(
                session_id,
                delete_global=bool(request.data.get("delete_global", False)),
                mode=request.data.get("mode"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("azure session teardown failed for %s", session_id)
            return Response(
                {"detail": f"Azure teardown failed: {exc}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        return Response(result, status=status.HTTP_200_OK)
