from django.urls import path

from . import views

urlpatterns = [
    path('', views.WebhookListCreateView.as_view(), name='webhook_list'),
    path('<uuid:pk>/', views.WebhookDetailView.as_view(), name='webhook_detail'),
    path('<uuid:pk>/test/', views.WebhookTestView.as_view(), name='webhook_test'),
    path('<uuid:pk>/deliveries/', views.WebhookDeliveryListView.as_view(), name='webhook_deliveries'),
]
