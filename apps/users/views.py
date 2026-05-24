"""User views."""

from rest_framework import status, generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import User
from .serializers import UserRegistrationSerializer, UserSerializer

class UserRegistrationView(APIView):
    """View for user registration."""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {'message': 'User registered successfully', 'user': serializer.data},
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserProfileView(APIView):
    """View for retrieving user profile."""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

class UserProfileUpdateView(generics.UpdateAPIView):
    """View for updating user profile."""
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_object(self):
        return self.request.user