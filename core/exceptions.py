"""Custom exception handlers."""

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

def exception_handler(exc, context):
    """Custom exception handler with consistent error responses."""
    response = drf_exception_handler(exc, context)
    
    if response is not None:
        response.data = {
            'success': False,
            'error': response.data.get('detail', 'An error occurred'),
            'status_code': response.status_code,
        }
    
    return response

class APIException(Exception):
    """Base exception for API errors."""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'An error occurred.'
    
    def __init__(self, detail=None):
        self.detail = detail or self.default_detail