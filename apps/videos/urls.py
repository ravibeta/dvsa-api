"""Video URLs."""

from django.urls import path
from rest_framework.routers import DefaultRouter

from . import views

app_name = 'videos'

router = DefaultRouter()
router.register(r'video-entities', views.VideoEntityViewSet, basename='video-entity')

urlpatterns = [
    path('', views.VideoListCreateView.as_view(), name='video_list'),
    path('<int:pk>/', views.VideoRetrieveUpdateDestroyView.as_view(), name='video_detail'),
    path('<int:pk>/process/', views.VideoProcessView.as_view(), name='video_process'),
    # Ported account-scoped pipeline endpoints.
    path('upload-video/', views.VideoUploadAPIView.as_view(), name='upload-video'),
    path('chat/', views.ChatAPIView.as_view(), name='chat'),
    # Raw qwen2.5vl:7b (Ollama) baseline — a peer of chat/ for direct comparison.
    path('baseline-test/', views.BaselineTestAPIView.as_view(), name='baseline-test'),
    # Direct tool: extract the four survey-area corner frames (bypasses RAG/agent).
    path('corners/', views.CornersAPIView.as_view(), name='corners'),
] + router.urls
