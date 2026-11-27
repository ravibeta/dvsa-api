from django.urls import path

from . import views

urlpatterns = [
    path('', views.MediaListView.as_view(), name='media_list'),
    path('upload-destination/', views.UploadDestinationView.as_view(), name='media_upload_destination'),
    path('events/<slug:provider>/', views.StorageEventView.as_view(), name='media_storage_events'),
    path('<uuid:pk>/', views.MediaDetailView.as_view(), name='media_detail'),
    path('<uuid:pk>/upload-destination/', views.UploadDestinationRenewView.as_view(),
         name='media_upload_destination_renew'),
    path('<uuid:pk>/complete/', views.MediaCompleteView.as_view(), name='media_complete'),
]
