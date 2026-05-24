"""Analytics URLs."""

from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('', views.AnalysisListView.as_view(), name='analysis_list'),
    path('<int:pk>/', views.AnalysisDetailView.as_view(), name='analysis_detail'),
]