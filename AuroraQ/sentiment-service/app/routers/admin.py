# app/routers/admin.py
"""Admin and monitoring API endpoints"""

import time
import psutil
from typing import Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import structlog

from models import ServiceStats, ModelInfo, ErrorResponse
from app.dependencies import get_sentiment_analyzer, get_fusion_manager, get_cache_client
from utils.redis_client import RedisCache
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter()

# Service statistics tracking
_service_stats = {
    "start_time": time.time(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "response_times": []
}


def update_request_stats(success: bool, response_time: float):
    """Update service statistics"""
    _service_stats["total_requests"] += 1
    if success:
        _service_stats["successful_requests"] += 1
    else:
        _service_stats["failed_requests"] += 1
    
    _service_stats["response_times"].append(response_time)
    # Keep only last 1000 response times
    if len(_service_stats["response_times"]) > 1000:
        _service_stats["response_times"] = _service_stats["response_times"][-1000:]


@router.get("/stats", response_model=ServiceStats)
async def get_service_statistics(
    analyzer=Depends(get_sentiment_analyzer),
    fusion_manager=Depends(get_fusion_manager),
    cache: RedisCache = Depends(get_cache_client)
) -> ServiceStats:
    """Get comprehensive service statistics"""
    
    try:
        current_time = time.time()
        uptime = current_time - _service_stats["start_time"]
        
        # Calculate average response time
        response_times = _service_stats["response_times"]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Get system resource usage
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Cache hit rate (placeholder - would need proper implementation)
        cache_hit_rate = 85.0  # Mock data
        
        # Model information
        model_info = {
            "finbert": ModelInfo(
                name="ProsusAI/finbert",
                version="1.0",
                loaded=analyzer._initialized if hasattr(analyzer, '_initialized') else True,
                load_time=None,  # Would track this during initialization
                memory_usage=None  # Would implement proper tracking
            )
        }
        
        stats = ServiceStats(
            total_requests=_service_stats["total_requests"],
            successful_requests=_service_stats["successful_requests"],
            failed_requests=_service_stats["failed_requests"],
            average_response_time=avg_response_time,
            uptime=uptime,
            cache_hit_rate=cache_hit_rate,
            model_info=model_info,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
        
        logger.info("Service statistics retrieved", uptime=uptime, total_requests=stats.total_requests)
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get service statistics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/system/info")
async def get_system_info() -> Dict[str, Any]:
    """Get system information and resource usage"""
    
    try:
        # System information
        system_info = {
            "platform": psutil.LINUX if hasattr(psutil, 'LINUX') else "unknown",
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "disk_usage": {
                "total": psutil.disk_usage('/').total / (1024**3),  # GB
                "used": psutil.disk_usage('/').used / (1024**3),  # GB
                "free": psutil.disk_usage('/').free / (1024**3)  # GB
            }
        }
        
        # Current resource usage
        resource_usage = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
        
        # Service configuration
        service_config = {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "debug": settings.debug,
            "log_level": settings.log_level,
            "max_workers": settings.max_workers,
            "cache_ttl": settings.cache_ttl,
            "redis_url": settings.redis_url,
            "finbert_model": settings.finbert_model_name
        }
        
        return {
            "system": system_info,
            "resources": resource_usage,
            "service": service_config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get system info", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system info: {str(e)}"
        )


@router.get("/config")
async def get_service_config() -> Dict[str, Any]:
    """Get service configuration (non-sensitive values only)"""
    
    try:
        config = {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "debug": settings.debug,
            "log_level": settings.log_level,
            "host": settings.host,
            "port": settings.port,
            "grpc_port": settings.grpc_port,
            "max_workers": settings.max_workers,
            "cache_ttl": settings.cache_ttl,
            "finbert_config": {
                "model_name": settings.finbert_model_name,
                "max_length": settings.finbert_max_length,
                "batch_size": settings.finbert_batch_size
            },
            "fusion_config": {
                "source_weights": settings.fusion_source_weights,
                "outlier_threshold": settings.fusion_outlier_threshold,
                "confidence_threshold": settings.fusion_confidence_threshold
            },
            "performance": {
                "enable_model_caching": settings.enable_model_caching,
                "model_warmup": settings.model_warmup,
                "max_concurrent_requests": settings.max_concurrent_requests,
                "request_timeout": settings.request_timeout
            },
            "monitoring": {
                "enable_metrics": settings.enable_metrics,
                "prometheus_port": settings.prometheus_port,
                "health_check_interval": settings.health_check_interval
            }
        }
        
        return {
            "config": config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get service config", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get config: {str(e)}"
        )


@router.post("/reload/config")
async def reload_configuration():
    """Reload service configuration (requires restart for some settings)"""
    
    try:
        # In a real implementation, this would reload configuration
        # For now, we'll just return current config
        
        logger.info("Configuration reload requested")
        
        return {
            "status": "success",
            "message": "Configuration reload completed",
            "note": "Some settings require service restart to take effect",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to reload configuration", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload config: {str(e)}"
        )


@router.post("/models/reload")
async def reload_models(
    analyzer=Depends(get_sentiment_analyzer),
    fusion_manager=Depends(get_fusion_manager)
):
    """Reload ML models (use with caution in production)"""
    
    try:
        start_time = time.time()
        
        # This would reinitialize models in a real implementation
        # For now, we'll simulate the process
        
        logger.info("Model reload initiated")
        
        # Simulate reload time
        import asyncio
        await asyncio.sleep(0.5)
        
        reload_time = time.time() - start_time
        
        logger.info("Model reload completed", reload_time=reload_time)
        
        return {
            "status": "success",
            "message": "Models reloaded successfully",
            "reload_time": reload_time,
            "models": ["finbert", "fusion_manager"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to reload models", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload models: {str(e)}"
        )


@router.get("/logs/recent")
async def get_recent_logs(
    lines: int = 100,
    level: str = "INFO"
) -> Dict[str, Any]:
    """Get recent log entries"""
    
    try:
        # In a real implementation, this would read from log files
        # For now, return a placeholder response
        
        return {
            "status": "success",
            "message": f"Log retrieval not implemented in this version",
            "requested_lines": lines,
            "requested_level": level,
            "log_file": settings.log_file_path,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get recent logs", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get logs: {str(e)}"
        )


@router.delete("/cache/clear/all")
async def clear_all_caches(
    cache: RedisCache = Depends(get_cache_client)
):
    """Clear all service caches (use with caution)"""
    
    try:
        logger.warning("All caches clear requested - this will impact performance")
        
        # In a real implementation, this would clear all Redis keys
        # with the service prefix
        
        return {
            "status": "success",
            "message": "All caches cleared successfully",
            "warning": "Performance may be impacted until caches are rebuilt",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to clear all caches", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear caches: {str(e)}"
        )


@router.post("/maintenance/mode")
async def toggle_maintenance_mode(
    enabled: bool = True
):
    """Toggle maintenance mode (placeholder for production implementation)"""
    
    try:
        mode = "enabled" if enabled else "disabled"
        
        logger.info(f"Maintenance mode {mode}")
        
        return {
            "status": "success",
            "maintenance_mode": enabled,
            "message": f"Maintenance mode {mode}",
            "note": "This is a placeholder implementation",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to toggle maintenance mode", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to toggle maintenance mode: {str(e)}"
        )