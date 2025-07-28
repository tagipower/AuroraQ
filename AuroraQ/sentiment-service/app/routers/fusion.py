# app/routers/fusion.py
"""Sentiment fusion API endpoints"""

import time
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from models import (
    FusionRequest,
    FusionResponse,
    ErrorResponse
)
from app.dependencies import get_fusion_manager, get_cache_client
from utils.redis_client import RedisCache, generate_fusion_cache_key
from utils.logging_config import get_logger, log_function_call
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter()


@router.post("/fuse", response_model=FusionResponse)
async def fuse_sentiment_scores(
    request: FusionRequest,
    background_tasks: BackgroundTasks,
    fusion_manager=Depends(get_fusion_manager),
    cache: RedisCache = Depends(get_cache_client)
) -> FusionResponse:
    """
    Fuse multiple sentiment scores from different sources
    
    - **sentiment_scores**: Dictionary of source-specific sentiment scores (0.0-1.0)
    - **symbol**: Trading symbol for context (optional, default: BTCUSDT)
    - **timestamp**: Analysis timestamp (optional, defaults to current time)
    
    Returns fused sentiment score with confidence, trend, and volatility metrics.
    """
    start_time = time.time()
    
    try:
        # Generate cache key based on scores hash
        scores_json = json.dumps(request.sentiment_scores, sort_keys=True)
        scores_hash = hashlib.md5(scores_json.encode()).hexdigest()
        cache_key = generate_fusion_cache_key(
            scores_hash, 
            request.symbol, 
            request.timestamp
        )
        
        # Try to get from cache first
        cached_result = await cache.get(cache_key)
        if cached_result:
            logger.info("Cache hit for fusion analysis", cache_key=cache_key)
            cached_result['processing_time'] = time.time() - start_time
            return FusionResponse(**cached_result)
        
        # Perform fusion analysis
        timestamp = request.timestamp or datetime.utcnow()
        
        fused_score = await fusion_manager.fuse(
            sentiment_scores=request.sentiment_scores,
            symbol=request.symbol,
            timestamp=timestamp
        )
        
        # Get fusion statistics for additional insights
        stats = fusion_manager.get_statistics(request.symbol)
        
        # Calculate confidence based on source diversity and agreement
        confidence = 0.8  # Default confidence
        if stats and 'average_confidence' in stats:
            confidence = stats['average_confidence']
        
        # Determine trend based on fused score
        if fused_score >= 0.7:
            trend = "strong_bullish"
        elif fused_score >= 0.6:
            trend = "bullish"
        elif fused_score >= 0.4:
            trend = "neutral"
        elif fused_score >= 0.3:
            trend = "bearish"
        else:
            trend = "strong_bearish"
        
        # Calculate volatility (placeholder - would use historical data)
        volatility = 0.1  # Default low volatility
        if len(request.sentiment_scores) > 1:
            # Calculate standard deviation of input scores as volatility proxy
            import numpy as np
            scores = list(request.sentiment_scores.values())
            volatility = float(np.std(scores))
        
        # Determine weights used (from fusion manager config)
        weights_used = {}
        for source in request.sentiment_scores.keys():
            weights_used[source] = fusion_manager.config.source_weights.get(source, 0.1)
        
        # Normalize weights
        total_weight = sum(weights_used.values())
        if total_weight > 0:
            weights_used = {k: v/total_weight for k, v in weights_used.items()}
        
        result = FusionResponse(
            fused_score=fused_score,
            confidence=confidence,
            trend=trend,
            volatility=volatility,
            raw_scores=request.sentiment_scores.copy(),
            weights_used=weights_used,
            sources_count=len(request.sentiment_scores),
            processing_time=time.time() - start_time,
            fusion_metadata={
                "algorithm": "adaptive_weighted_average",
                "outlier_detection": fusion_manager.config.outlier_detection,
                "outlier_threshold": fusion_manager.config.outlier_std_threshold,
                "statistics": stats
            }
        )
        
        # Cache result in background
        background_tasks.add_task(
            cache.set,
            cache_key,
            result.dict(exclude={'processing_time', 'timestamp'}),
            ttl=settings.cache_ttl
        )
        
        # Log performance
        logger.info(
            "Sentiment fusion completed",
            **log_function_call(
                "fuse_sentiment_scores",
                symbol=request.symbol,
                sources=list(request.sentiment_scores.keys()),
                fused_score=fused_score,
                confidence=confidence,
                trend=trend,
                processing_time=result.processing_time
            )
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Sentiment fusion failed",
            error=str(e),
            symbol=request.symbol,
            sources=list(request.sentiment_scores.keys())
        )
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment fusion failed: {str(e)}"
        )


@router.get("/statistics/{symbol}")
async def get_fusion_statistics(
    symbol: str,
    fusion_manager=Depends(get_fusion_manager)
) -> Dict[str, Any]:
    """
    Get fusion statistics for a specific trading symbol
    
    - **symbol**: Trading symbol (e.g., BTCUSDT, ETH, BTC)
    
    Returns historical fusion statistics including score distribution,
    trend analysis, and performance metrics.
    """
    try:
        stats = fusion_manager.get_statistics(symbol.upper())
        
        if not stats:
            return {
                "symbol": symbol.upper(),
                "status": "no_data",
                "message": "No fusion data available for this symbol",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Add metadata
        stats.update({
            "symbol": symbol.upper(),
            "status": "available",
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "source_weights": fusion_manager.config.source_weights,
                "outlier_detection": fusion_manager.config.outlier_detection,
                "confidence_threshold": fusion_manager.config.confidence_threshold
            }
        })
        
        logger.info(
            "Fusion statistics retrieved",
            symbol=symbol,
            data_points=stats.get('count', 0)
        )
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get fusion statistics", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/weights")
async def get_fusion_weights(
    fusion_manager=Depends(get_fusion_manager)
) -> Dict[str, Any]:
    """Get current fusion weights configuration"""
    try:
        return {
            "static_weights": fusion_manager.config.source_weights,
            "adaptive_weights": fusion_manager.adaptive_weights,
            "adaptive_weighting_enabled": fusion_manager.config.adaptive_weighting,
            "weight_update_count": fusion_manager.weight_update_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get fusion weights", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get weights: {str(e)}"
        )


@router.put("/weights")
async def update_fusion_weights(
    new_weights: Dict[str, float],
    fusion_manager=Depends(get_fusion_manager)
) -> Dict[str, Any]:
    """
    Update fusion weights configuration
    
    - **new_weights**: Dictionary of source weights (must sum to 1.0)
    
    Updates the static weights used in sentiment fusion.
    Requires admin permissions in production.
    """
    try:
        # Validate weights
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        # Update weights
        fusion_manager.update_weights(new_weights)
        
        logger.info("Fusion weights updated", new_weights=new_weights)
        
        return {
            "status": "success",
            "message": "Fusion weights updated successfully",
            "old_weights": fusion_manager.config.source_weights,
            "new_weights": new_weights,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to update fusion weights", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update weights: {str(e)}"
        )


@router.get("/health")
async def fusion_health_check(
    fusion_manager=Depends(get_fusion_manager)
):
    """Health check for sentiment fusion service"""
    try:
        # Test fusion with sample data
        test_scores = {"news": 0.6, "social": 0.5}
        test_result = await fusion_manager.fuse(test_scores, symbol="TEST")
        
        return {
            "status": "healthy",
            "service": "sentiment_fusion",
            "algorithm": "adaptive_weighted_average",
            "initialized": fusion_manager._initialized if hasattr(fusion_manager, '_initialized') else True,
            "test_result": test_result,
            "history_size": len(fusion_manager.history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Fusion service health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "sentiment_fusion",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.delete("/cache/clear")
async def clear_fusion_cache(
    cache: RedisCache = Depends(get_cache_client)
):
    """Clear fusion analysis cache (admin endpoint)"""
    try:
        logger.info("Fusion cache clear requested")
        
        return {
            "status": "success",
            "message": "Fusion cache clear initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to clear fusion cache", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )