"""Tests for user endpoints."""

from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from apps.users.models import User

class UserRegistrationTest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.register_url = '/api/v1/users/register/'
    
    def test_user_registration(self):
        data = {
            'email': 'test@example.com',
            'password': 'testpass123',
            'password2': 'testpass123',
            'first_name': 'Test',
            'last_name': 'User',
        }
        response = self.client.post(self.register_url, data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(User.objects.count(), 1)

class UserProfileTest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            first_name='Test',
            last_name='User',
        )
        self.profile_url = '/api/v1/users/profile/'
    
    def test_get_profile_authenticated(self):
        self.client.force_authenticate(user=self.user)
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_get_profile_unauthenticated(self):
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)