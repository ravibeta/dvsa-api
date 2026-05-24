"""User admin configuration."""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ('email', 'first_name', 'last_name', 'is_verified', 'is_active', 'created_at')
    list_filter = ('is_active', 'is_verified', 'created_at')
    search_fields = ('email', 'first_name', 'last_name')
    readonly_fields = ('created_at', 'updated_at')