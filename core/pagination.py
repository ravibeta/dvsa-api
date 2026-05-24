"""Custom pagination classes."""

from rest_framework.pagination import PageNumberPagination, CursorPagination

class StandardResultsSetPagination(PageNumberPagination):
    """Standard pagination with configurable page size."""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

class LargeResultsSetPagination(PageNumberPagination):
    """Pagination for larger result sets."""
    page_size = 100
    page_size_query_param = 'page_size'
    max_page_size = 1000

class CursorSetPagination(CursorPagination):
    """Cursor-based pagination for large datasets."""
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 500
    ordering = '-created_at'