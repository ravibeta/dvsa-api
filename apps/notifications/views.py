"""Webhook subscription CRUD, test pings and delivery history (owner-scoped)."""

from django.shortcuts import get_object_or_404
from drf_spectacular.utils import extend_schema
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .broadcast import make_envelope, queue_delivery
from .events import PING
from .models import WebhookSubscription
from .serializers import (WebhookDeliveryLogSerializer, WebhookSubscriptionCreateSerializer,
                          WebhookSubscriptionSerializer)


class _OwnedSubscriptions:
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return WebhookSubscription.objects.filter(owner=self.request.user)


def _invalid(serializer):
    # Field errors are returned as-is (the global exception handler would collapse them).
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class WebhookListCreateView(_OwnedSubscriptions, generics.ListCreateAPIView):
    """``GET`` the caller's subscriptions; ``POST`` creates one and returns its ``secret_key`` once."""

    def get_serializer_class(self):
        if self.request.method == "POST":
            return WebhookSubscriptionCreateSerializer
        return WebhookSubscriptionSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return _invalid(serializer)
        serializer.save(owner=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class WebhookDetailView(_OwnedSubscriptions, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = WebhookSubscriptionSerializer

    def update(self, request, *args, **kwargs):
        serializer = self.get_serializer(self.get_object(), data=request.data,
                                         partial=kwargs.pop("partial", False))
        if not serializer.is_valid():
            return _invalid(serializer)
        serializer.save()
        return Response(serializer.data)


class WebhookTestView(_OwnedSubscriptions, APIView):
    """Queue a signed ``ping`` delivery to this subscription (whatever its event filter)."""

    @extend_schema(request=None, responses={202: dict})
    def post(self, request, pk):
        sub = get_object_or_404(self.get_queryset(), pk=pk)
        envelope = make_envelope(PING, {"subscription_id": str(sub.id)})
        delivery_id = queue_delivery(sub, envelope)
        return Response({"delivery_id": delivery_id, "event": envelope},
                        status=status.HTTP_202_ACCEPTED)


class WebhookDeliveryListView(_OwnedSubscriptions, generics.ListAPIView):
    serializer_class = WebhookDeliveryLogSerializer

    def get_queryset(self):
        sub = get_object_or_404(super().get_queryset(), pk=self.kwargs["pk"])
        return sub.deliveries.all()
