"""Analytics URLs."""

from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('', views.AnalysisListView.as_view(), name='analysis_list'),
    path('routines/', views.RoutineListView.as_view(), name='routine_list'),
    path('videos/<int:video_id>/run/', views.RunAnalysisView.as_view(), name='run_analysis'),
    path('<int:pk>/', views.AnalysisDetailView.as_view(), name='analysis_detail'),
]
