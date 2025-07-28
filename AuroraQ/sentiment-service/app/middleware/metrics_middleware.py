# app/middleware/metrics_middleware.py
"""Prometheus metrics middleware for request monitoring"""

import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from prometheus_client import Counter, Histogram, Gauge
import structlog

from utils.logging_config import get_logger, log_api_request

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

RESPONSE_SIZE = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

ERROR_COUNT = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'error_type']
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics"""
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        # Paths to exclude from metrics collection
        self.exclude_paths = exclude_paths or ['/metrics', '/health', '/docs', '/redoc', '/openapi.json']
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""
        
        # Skip metrics collection for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Extract request information
        method = request.method
        path = request.url.path
        
        # Clean up path for metrics (remove IDs and query params)
        endpoint = self._clean_endpoint(path)
        
        # Get request size
        request_size = int(request.headers.get('content-length', 0))
        
        # Start timing
        start_time = time.time()
        
        # Increment active requests
        ACTIVE_REQUESTS.inc()
        
        response = None
        status_code = 500  # Default to error
        error_type = None
        
        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code
            
            # Check if it's an error
            if status_code >= 400:
                error_type = f"{status_code//100}xx"
            
        except Exception as e:
            # Handle exceptions during request processing
            status_code = 500
            error_type = "exception"
            
            logger.error(
                "Request processing exception",
                method=method,
                path=path,
                error=str(e)
            )
            
            # Re-raise to let the application handle it
            raise
        
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Get response size
            response_size = 0
            if response and hasattr(response, 'headers'):
                response_size = int(response.headers.get('content-length', 0))
            
            # Update metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            if request_size > 0:
                REQUEST_SIZE.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(request_size)
            
            if response_size > 0:
                RESPONSE_SIZE.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(response_size)
            
            # Record errors
            if error_type:
                ERROR_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    error_type=error_type
                ).inc()
            
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
            
            # Log request details
            log_context = log_api_request(
                method=method,
                path=path,
                status_code=status_code,
                duration=duration,
                request_size=request_size,
                response_size=response_size
            )
            
            if status_code >= 400:
                logger.warning("HTTP error response", **log_context)
            elif duration > 5.0:  # Log slow requests
                logger.warning("Slow request detected", **log_context)
            else:
                logger.info("HTTP request completed", **log_context)
        
        return response
    
    def _clean_endpoint(self, path: str) -> str:
        """Clean endpoint path for consistent metrics labeling"""
        
        # Remove query parameters
        if '?' in path:
            path = path.split('?')[0]
        
        # Replace common ID patterns with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
            '/{uuid}',
            path,
            flags=re.IGNORECASE
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace common crypto symbols
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
        for symbol in crypto_symbols:
            path = path.replace(f'/{symbol}', '/{symbol}')
            path = path.replace(f'/{symbol.lower()}', '/{symbol}')
        
        # Limit path depth for metrics
        path_parts = path.split('/')
        if len(path_parts) > 6:  # /api/v1/sentiment/analyze/{symbol}
            path = '/'.join(path_parts[:6]) + '/...'
        
        return path or '/'