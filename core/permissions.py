"""Custom permission classes."""

from rest_framework import permissions

class IsOwner(permissions.BasePermission):
    """Only owners can access their objects."""
    def has_object_permission(self, request, view, obj):
        return obj.user == request.user

class IsOwnerOrReadOnly(permissions.BasePermission):
    """Owners can edit, others can read."""
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.user == request.user

class IsAdminOrReadOnly(permissions.BasePermission):
    """Admin can edit, others can read."""
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user and request.user.is_staff