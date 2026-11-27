"""Notifications admin configuration."""

from django.contrib import admin

from .models import WebhookDeliveryLog, WebhookSubscription


@admin.register(WebhookSubscription)
class WebhookSubscriptionAdmin(admin.ModelAdmin):
    list_display = ('id', 'owner', 'target_url', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('target_url', 'owner__email')
    exclude = ('secret_key',)
    readonly_fields = ('created_at', 'updated_at')


@admin.register(WebhookDeliveryLog)
class WebhookDeliveryLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'subscription', 'event_type', 'response_status_code', 'attempt_count',
                    'delivered_at')
    list_filter = ('event_type', 'delivered_at')
    readonly_fields = ('created_at',)
