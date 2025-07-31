#!/usr/bin/env python3
"""
Simple AuroraQ Sentiment Service - Test Version
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import sys

# Add project root to path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.settings import settings
    from utils.redis_client import get_redis_client, close_redis_client
    from utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    # Create minimal settings
    class SimpleSettings:
        host = "0.0.0.0"
        port = 8000
        app_name = "AuroraQ Sentiment Service"
        debug = False
        cors_origins = "*"
        allowed_hosts = "*"
    settings = SimpleSettings()
    
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AuroraQ Sentiment Service",
    description="Real-time cryptocurrency sentiment analysis and trading signals",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.cors_origins == "*" else settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AuroraQ Sentiment Service",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection if available
        redis_status = "unknown"
        try:
            from utils.redis_client import get_redis_client
            redis_client = await get_redis_client()
            await redis_client.ping()
            redis_status = "connected"
        except Exception as redis_error:
            redis_status = f"error: {str(redis_error)}"
        
        return {
            "status": "healthy",
            "service": settings.app_name,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "redis": redis_status,
                "api": "healthy"
            },
            "uptime": "running"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/fusion/current-sentiment")
async def get_current_sentiment():
    """Mock current sentiment endpoint"""
    return {
        "status": "success",
        "fusion_score": 15.3,
        "change": 2.1,
        "confidence": 0.85,
        "timestamp": datetime.now().isoformat(),
        "category_breakdown": {
            "news": 18.5,
            "social": 12.1,
            "technical": 16.8,
            "historical": 14.2
        },
        "sources": {
            "news_articles": 45,
            "reddit_posts": 23,
            "technical_indicators": 8
        }
    }

@app.get("/events/timeline")
async def get_events_timeline():
    """Mock events timeline endpoint"""
    return {
        "status": "success",
        "events": [
            {
                "timestamp": "2025-07-29T14:30:00Z",
                "title": "Bitcoin ETF Approval Speculation",
                "impact_score": 0.82,
                "sentiment_score": 35.5,
                "volatility": 0.65,
                "source": "news"
            },
            {
                "timestamp": "2025-07-29T13:15:00Z",
                "title": "Fed Interest Rate Decision",
                "impact_score": 0.91,
                "sentiment_score": -22.0,
                "volatility": 0.75,
                "source": "economic"
            }
        ]
    }

@app.get("/trading/strategies/performance")
async def get_strategy_performance():
    """Mock strategy performance endpoint"""
    return {
        "status": "success",
        "strategies": [
            {
                "name": "AuroraQ RuleA",
                "roi": 3.2,
                "sharpe_ratio": 1.25,
                "current_score": 5.0,
                "trades": 127,
                "win_rate": 0.68
            },
            {
                "name": "MacroQ BTC Portfolio", 
                "roi": 2.5,
                "sharpe_ratio": 0.90,
                "current_score": -1.0,
                "trades": 89,
                "win_rate": 0.61
            }
        ]
    }

@app.get("/admin/metrics")
async def get_metrics():
    """Mock system metrics endpoint"""
    import psutil
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "network_percent": 15.0,  # Mock value
            "redis_hit_rate": 92.0   # Mock value
        }
    except ImportError:
        # Fallback if psutil not available
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": 45.0,
            "memory_percent": 60.0,
            "disk_percent": 72.0,
            "network_percent": 15.0,
            "redis_hit_rate": 92.0
        }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Server running on {settings.host}:{settings.port}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down AuroraQ Sentiment Service")
    try:
        await close_redis_client()
    except:
        pass

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main_simple:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )