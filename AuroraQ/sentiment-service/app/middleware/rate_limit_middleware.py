# app/middleware/rate_limit_middleware.py
"""Rate limiting middleware using Redis"""

import time
import hashlib
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
import structlog

from utils.redis_client import get_cache
from config.settings import settings
from models import ErrorResponse

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-based rate limiting middleware"""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        requests_per_hour: int = 1000,
        burst_requests: int = 20,
        exclude_paths: list = None
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour  
        self.burst_requests = burst_requests
        self.exclude_paths = exclude_paths or ['/health', '/metrics', '/docs', '/redoc']
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests"""
        
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limits
        is_allowed, reason, retry_after = await self._check_rate_limits(client_id)
        
        if not is_allowed:
            logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                reason=reason,
                path=request.url.path,
                method=request.method
            )
            
            # Return rate limit error
            error_response = ErrorResponse(
                error="rate_limit_exceeded",
                message=f"Rate limit exceeded: {reason}",
                details={
                    "retry_after": retry_after,
                    "limit_type": reason
                }
            )
            
            return JSONResponse(
                status_code=429,
                content=error_response.dict(),
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        remaining = await self._get_remaining_requests(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request"""
        
        # Priority order: API Key > IP Address
        
        # Check for API key in headers
        api_key = request.headers.get(settings.api_key_header)
        if api_key:
            # Hash the API key for privacy
            return f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for forwarded IP (when behind proxy)
        forwarded_ip = request.headers.get("X-Forwarded-For")
        if forwarded_ip:
            client_ip = forwarded_ip.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"
    
    async def _check_rate_limits(self, client_id: str) -> tuple[bool, Optional[str], int]:
        """Check if client is within rate limits"""
        
        try:
            cache = await get_cache()
            current_time = int(time.time())
            
            # Define time windows
            minute_window = current_time // 60
            hour_window = current_time // 3600
            burst_window = current_time // 10  # 10-second burst window
            
            # Generate cache keys
            minute_key = f"rate_limit:{client_id}:minute:{minute_window}"
            hour_key = f"rate_limit:{client_id}:hour:{hour_window}"
            burst_key = f"rate_limit:{client_id}:burst:{burst_window}"
            
            # Get current counts
            minute_count = await cache.get(minute_key) or 0
            hour_count = await cache.get(hour_key) or 0
            burst_count = await cache.get(burst_key) or 0
            
            # Convert to integers
            minute_count = int(minute_count)
            hour_count = int(hour_count)
            burst_count = int(burst_count)
            
            # Check burst limit (10 seconds)
            if burst_count >= self.burst_requests:
                return False, "burst_limit", 10
            
            # Check minute limit
            if minute_count >= self.requests_per_minute:
                return False, "minute_limit", 60
            
            # Check hour limit
            if hour_count >= self.requests_per_hour:
                return False, "hour_limit", 3600
            
            # Increment counters
            await cache.set(minute_key, minute_count + 1, ttl=60)
            await cache.set(hour_key, hour_count + 1, ttl=3600)
            await cache.set(burst_key, burst_count + 1, ttl=10)
            
            return True, None, 0
            
        except Exception as e:
            logger.error("Rate limit check failed", client_id=client_id, error=str(e))
            # Allow request if rate limiting fails
            return True, None, 0
    
    async def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for current minute"""
        
        try:
            cache = await get_cache()
            current_time = int(time.time())
            minute_window = current_time // 60
            minute_key = f"rate_limit:{client_id}:minute:{minute_window}"
            
            minute_count = await cache.get(minute_key) or 0
            minute_count = int(minute_count)
            
            return max(0, self.requests_per_minute - minute_count)
            
        except Exception as e:
            logger.error("Failed to get remaining requests", client_id=client_id, error=str(e))
            return self.requests_per_minute


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """IP whitelist middleware for additional security"""
    
    def __init__(self, app, whitelist: list[str] = None, enabled: bool = False):
        super().__init__(app)
        self.whitelist = set(whitelist or [])
        self.enabled = enabled
        
        # Add localhost by default
        self.whitelist.update(['127.0.0.1', '::1', 'localhost'])
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check IP whitelist"""
        
        if not self.enabled or not self.whitelist:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check forwarded headers
        forwarded_ip = request.headers.get("X-Forwarded-For")
        if forwarded_ip:
            client_ip = forwarded_ip.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        # Check whitelist
        if client_ip not in self.whitelist:
            logger.warning("IP not in whitelist", client_ip=client_ip, path=request.url.path)
            
            error_response = ErrorResponse(
                error="access_denied",
                message="Access denied: IP not in whitelist"
            )
            
            return JSONResponse(
                status_code=403,
                content=error_response.dict()
            )
        
        return await call_next(request)