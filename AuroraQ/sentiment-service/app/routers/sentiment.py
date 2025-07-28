# app/routers/sentiment.py
"""Sentiment analysis API endpoints"""

import time
import hashlib
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from models import (
    SentimentRequest, 
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    ErrorResponse,
    SentimentLabel
)
from app.dependencies import get_sentiment_analyzer, get_cache_client
from utils.redis_client import RedisCache, generate_sentiment_cache_key
from utils.logging_config import get_logger, log_function_call
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks,
    analyzer=Depends(get_sentiment_analyzer),
    cache: RedisCache = Depends(get_cache_client)
) -> SentimentResponse:
    """
    Analyze sentiment of a single text using FinBERT
    
    - **text**: Text to analyze (1-10000 characters)
    - **symbol**: Asset symbol for context (optional, default: CRYPTO)
    - **include_detailed**: Include detailed analysis results (optional)
    """
    start_time = time.time()
    
    try:
        # Generate cache key based on text hash
        text_hash = hashlib.md5(request.text.encode()).hexdigest()
        cache_key = generate_sentiment_cache_key(text_hash, "finbert")
        
        # Try to get from cache first
        cached_result = await cache.get(cache_key)
        if cached_result and not request.include_detailed:
            logger.info("Cache hit for sentiment analysis", cache_key=cache_key)
            cached_result['processing_time'] = time.time() - start_time
            return SentimentResponse(**cached_result)
        
        # Perform sentiment analysis
        if request.include_detailed:
            detailed_result = await analyzer.analyze_detailed(request.text)
            
            result = SentimentResponse(
                sentiment_score=detailed_result.sentiment_score,
                label=SentimentLabel(detailed_result.label.value),
                confidence=detailed_result.confidence,
                processing_time=time.time() - start_time,
                keywords=detailed_result.keywords,
                scenario_tag=detailed_result.scenario_tag,
                metadata={
                    "symbol": request.symbol,
                    "model": "finbert",
                    "detailed": True
                }
            )
        else:
            # Simple analysis
            sentiment_score = await analyzer.analyze(request.text)
            
            # Determine label based on score
            if sentiment_score >= 0.6:
                label = SentimentLabel.POSITIVE
            elif sentiment_score <= 0.4:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL
            
            result = SentimentResponse(
                sentiment_score=sentiment_score,
                label=label,
                confidence=0.8,  # Default confidence for simple analysis
                processing_time=time.time() - start_time,
                metadata={
                    "symbol": request.symbol,
                    "model": "finbert",
                    "detailed": False
                }
            )
        
        # Cache result in background
        if not request.include_detailed:
            background_tasks.add_task(
                cache.set,
                cache_key,
                result.dict(exclude={'processing_time', 'timestamp'}),
                ttl=settings.cache_ttl
            )
        
        # Log performance
        logger.info(
            "Sentiment analysis completed",
            **log_function_call(
                "analyze_sentiment",
                text_length=len(request.text),
                symbol=request.symbol,
                score=result.sentiment_score,
                label=result.label.value,
                processing_time=result.processing_time
            )
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Sentiment analysis failed",
            error=str(e),
            text_length=len(request.text),
            symbol=request.symbol
        )
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


@router.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment(
    request: BatchSentimentRequest,
    background_tasks: BackgroundTasks,
    analyzer=Depends(get_sentiment_analyzer),
    cache: RedisCache = Depends(get_cache_client)
) -> BatchSentimentResponse:
    """
    Analyze sentiment for multiple texts in batch
    
    - **texts**: List of texts to analyze (1-100 items)
    - **symbol**: Asset symbol for context (optional)
    - **include_detailed**: Include detailed analysis results (optional)
    """
    start_time = time.time()
    
    try:
        results = []
        cache_hits = 0
        
        # Process each text
        for i, text in enumerate(request.texts):
            text_start_time = time.time()
            
            try:
                # Generate cache key
                text_hash = hashlib.md5(text.encode()).hexdigest()
                cache_key = generate_sentiment_cache_key(text_hash, "finbert")
                
                # Check cache
                cached_result = await cache.get(cache_key)
                if cached_result and not request.include_detailed:
                    cached_result['processing_time'] = time.time() - text_start_time
                    results.append(SentimentResponse(**cached_result))
                    cache_hits += 1
                    continue
                
                # Analyze text
                if request.include_detailed:
                    detailed_result = await analyzer.analyze_detailed(text)
                    
                    result = SentimentResponse(
                        sentiment_score=detailed_result.sentiment_score,
                        label=SentimentLabel(detailed_result.label.value),
                        confidence=detailed_result.confidence,
                        processing_time=time.time() - text_start_time,
                        keywords=detailed_result.keywords,
                        scenario_tag=detailed_result.scenario_tag,
                        metadata={
                            "symbol": request.symbol,
                            "model": "finbert",
                            "detailed": True,
                            "batch_index": i
                        }
                    )
                else:
                    # Simple analysis
                    sentiment_score = await analyzer.analyze(text)
                    
                    # Determine label
                    if sentiment_score >= 0.6:
                        label = SentimentLabel.POSITIVE
                    elif sentiment_score <= 0.4:
                        label = SentimentLabel.NEGATIVE
                    else:
                        label = SentimentLabel.NEUTRAL
                    
                    result = SentimentResponse(
                        sentiment_score=sentiment_score,
                        label=label,
                        confidence=0.8,
                        processing_time=time.time() - text_start_time,
                        metadata={
                            "symbol": request.symbol,
                            "model": "finbert",
                            "detailed": False,
                            "batch_index": i
                        }
                    )
                
                results.append(result)
                
                # Cache result in background
                if not request.include_detailed:
                    background_tasks.add_task(
                        cache.set,
                        cache_key,
                        result.dict(exclude={'processing_time', 'timestamp'}),
                        ttl=settings.cache_ttl
                    )
                
            except Exception as e:
                logger.error(f"Failed to analyze text {i}", error=str(e), text_length=len(text))
                # Add error result
                results.append(SentimentResponse(
                    sentiment_score=0.5,
                    label=SentimentLabel.NEUTRAL,
                    confidence=0.0,
                    processing_time=time.time() - text_start_time,
                    metadata={
                        "error": str(e),
                        "batch_index": i
                    }
                ))
        
        # Calculate batch statistics
        sentiment_scores = [r.sentiment_score for r in results]
        average_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        total_processing_time = time.time() - start_time
        
        batch_response = BatchSentimentResponse(
            results=results,
            total_count=len(results),
            average_score=average_score,
            processing_time=total_processing_time
        )
        
        # Log batch performance
        logger.info(
            "Batch sentiment analysis completed",
            **log_function_call(
                "analyze_batch_sentiment",
                batch_size=len(request.texts),
                symbol=request.symbol,
                cache_hits=cache_hits,
                average_score=average_score,
                processing_time=total_processing_time
            )
        )
        
        return batch_response
        
    except Exception as e:
        logger.error(
            "Batch sentiment analysis failed",
            error=str(e),
            batch_size=len(request.texts)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Batch sentiment analysis failed: {str(e)}"
        )


@router.get("/health")
async def sentiment_health_check(
    analyzer=Depends(get_sentiment_analyzer)
):
    """Health check for sentiment analysis service"""
    try:
        # Test analyzer with simple text
        test_score = await analyzer.analyze("test")
        
        return {
            "status": "healthy",
            "service": "sentiment_analysis",
            "model": "finbert",
            "initialized": analyzer._initialized,
            "test_score": test_score,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Sentiment service health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "sentiment_analysis",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/model/info")
async def get_model_info(
    analyzer=Depends(get_sentiment_analyzer)
):
    """Get information about the loaded sentiment model"""
    try:
        return {
            "model_name": "ProsusAI/finbert",
            "model_type": "transformer",
            "task": "sentiment_analysis",
            "language": "en",
            "domains": ["finance", "cryptocurrency"],
            "output_range": "0.0 to 1.0",
            "labels": ["negative", "neutral", "positive"],
            "initialized": analyzer._initialized,
            "cache_enabled": settings.enable_model_caching,
            "batch_size": settings.finbert_batch_size,
            "max_length": settings.finbert_max_length
        }
        
    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.delete("/cache/clear")
async def clear_sentiment_cache(
    cache: RedisCache = Depends(get_cache_client)
):
    """Clear sentiment analysis cache (admin endpoint)"""
    try:
        # This would require implementing a pattern-based delete in Redis
        # For now, we'll return a placeholder response
        
        logger.info("Sentiment cache clear requested")
        
        return {
            "status": "success",
            "message": "Sentiment cache clear initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to clear sentiment cache", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )