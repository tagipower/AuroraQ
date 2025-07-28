# app/main.py
"""FastAPI main application for sentiment analysis service"""

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import structlog

from config.settings import settings, get_settings
from models import HealthResponse, ErrorResponse
from utils.logging_config import setup_logging
from utils.redis_client import get_redis_client
from .routers import sentiment, fusion, admin
from .middleware import MetricsMiddleware, RateLimitMiddleware
from .dependencies import get_sentiment_analyzer, get_fusion_manager


# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load models')

# Global variables for service state
SERVICE_START_TIME = time.time()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    # Startup
    logger.info("Starting AuroraQ Sentiment Service", version=settings.app_version)
    
    try:
        # Initialize Redis connection
        redis_client = await get_redis_client()
        await redis_client.ping()
        logger.info("Redis connection established")
        
        # Initialize and warm up models if enabled
        if settings.model_warmup:
            logger.info("Warming up models...")
            model_start_time = time.time()
            
            # Load FinBERT analyzer
            analyzer = await get_sentiment_analyzer()
            fusion_manager = await get_fusion_manager()
            
            # Warm up with sample text
            await analyzer.analyze("Sample text for model warmup")
            await fusion_manager.fuse({"news": 0.5}, symbol="BTC")
            
            load_time = time.time() - model_start_time
            MODEL_LOAD_TIME.set(load_time)
            logger.info("Model warmup completed", load_time=load_time)
        
        app.state.start_time = SERVICE_START_TIME
        logger.info("Service startup complete")
        
    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AuroraQ Sentiment Service")
    
    try:
        # Close Redis connection
        redis_client = await get_redis_client()
        await redis_client.close()
        logger.info("Redis connection closed")
        
        # Close ML models
        analyzer = await get_sentiment_analyzer()
        fusion_manager = await get_fusion_manager()
        
        if hasattr(analyzer, 'close'):
            await analyzer.close()
        if hasattr(fusion_manager, 'close'):
            await fusion_manager.close()
            
        logger.info("Service shutdown complete")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Setup logging
    setup_logging()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered sentiment analysis service for financial markets",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add security middleware
    if settings.allowed_hosts != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(MetricsMiddleware)
    
    if settings.debug is False:  # Only in production
        app.add_middleware(RateLimitMiddleware)
    
    # Include routers
    app.include_router(sentiment.router, prefix="/api/v1/sentiment", tags=["sentiment"])
    app.include_router(fusion.router, prefix="/api/v1/fusion", tags=["fusion"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Service health check"""
        try:
            uptime = time.time() - SERVICE_START_TIME
            
            # Check Redis connectivity
            redis_client = await get_redis_client()
            await redis_client.ping()
            redis_status = "healthy"
        except Exception:
            redis_status = "unhealthy"
        
        # Check model status
        try:
            analyzer = await get_sentiment_analyzer()
            model_status = "healthy" if analyzer._initialized else "loading"
        except Exception:
            model_status = "unhealthy"
        
        overall_status = "healthy" if redis_status == "healthy" and model_status == "healthy" else "degraded"
        
        return HealthResponse(
            status=overall_status,
            version=settings.app_version,
            uptime=uptime,
            components={
                "redis": redis_status,
                "finbert_model": model_status
            },
            metrics={
                "memory_usage_mb": 0,  # TODO: Implement actual memory monitoring
                "active_connections": ACTIVE_CONNECTIONS._value.get(),
                "total_requests": REQUEST_COUNT._value.sum()
            }
        )
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint"""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs": "/docs" if settings.debug else "disabled",
            "health": "/health"
        }
    
    # Metrics endpoint for Prometheus
    if settings.enable_metrics:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler"""
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc)
        )
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                message="An internal server error occurred",
                details={"path": str(request.url.path)} if settings.debug else None
            ).dict()
        )
    
    # HTTP exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP exception handler"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"http_{exc.status_code}",
                message=exc.detail,
                details={"path": str(request.url.path)} if settings.debug else None
            ).dict()
        )
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    # Run with uvicorn
    log_level = settings.log_level.lower()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=log_level,
        workers=1 if settings.debug else settings.max_workers,
        access_log=True,
        use_colors=True
    )