"""Video URLs."""

from django.urls import path
from . import views

app_name = 'videos'

urlpatterns = [
    path('', views.VideoListCreateView.as_view(), name='video_list'),
    path('<int:pk>/', views.VideoRetrieveUpdateDestroyView.as_view(), name='video_detail'),
    path('<int:pk>/process/', views.VideoProcessView.as_view(), name='video_process'),
]