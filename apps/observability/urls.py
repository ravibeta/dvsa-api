"""Observability URLs."""

from django.urls import path

from . import views

app_name = "observability"

urlpatterns = [
    path("events/", views.EventIngestListView.as_view(), name="event_list"),
    path("events/aggregate/", views.EventAggregateView.as_view(), name="event_aggregate"),
]
