"""Storage URLs."""

from django.urls import path

from .views import AzureSessionView

app_name = 'storage'

urlpatterns = [
    # POST = setup, DELETE = teardown of the per-session Azure environment.
    path("azure-session/", AzureSessionView.as_view(), name="azure-session"),
]
