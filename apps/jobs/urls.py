from django.urls import path

from . import views

urlpatterns = [
    path('', views.JobListCreateView.as_view(), name='job_list'),
    path('<uuid:pk>/', views.JobDetailView.as_view(), name='job_detail'),
    path('<uuid:pk>/cancel/', views.JobCancelView.as_view(), name='job_cancel'),
    path('<uuid:pk>/retry/', views.JobRetryView.as_view(), name='job_retry'),
]
