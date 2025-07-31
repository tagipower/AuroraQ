#!/usr/bin/env python3
"""
Events Router for AuroraQ Sentiment Service  
Big Event Detection API endpoints
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from ..dependencies import get_big_event_detector, get_enhanced_news_collector
from ...processors.big_event_detector import BigEvent, EventType, EventImpact
from ...collectors.enhanced_news_collector import NewsItem
from ...models import SuccessResponse, ErrorResponse

logger = structlog.get_logger(__name__)

router = APIRouter()

# Request/Response Models
class EventDetectionRequest(BaseModel):
    """이벤트 감지 요청"""
    symbol: Optional[str] = Field(None, description="특정 심볼 필터링")
    hours_back: int = Field(24, ge=1, le=168, description="과거 몇 시간 데이터 조회")
    min_confidence: float = Field(0.6, ge=0.0, le=1.0, description="최소 신뢰도")
    min_impact: float = Field(0.5, ge=0.0, le=2.0, description="최소 영향도")

class EventResponse(BaseModel):
    """이벤트 응답"""
    event_id: str
    event_type: str
    title: str
    description: str
    impact_level: str
    base_impact_score: float
    sentiment_bias: float
    volatility_factor: float
    final_impact_score: float
    confidence: float
    detected_at: str
    event_time: Optional[str]
    source_urls: List[str]
    keywords: List[str]
    symbols_affected: List[str]
    news_count: int

class EventsListResponse(BaseModel):
    """이벤트 목록 응답"""
    success: bool = True
    events: List[EventResponse]
    total_count: int
    processing_time: float
    metadata: Dict[str, Any]

class EventStatsResponse(BaseModel):
    """이벤트 통계 응답"""
    success: bool = True
    stats: Dict[str, Any]
    active_events: int
    by_type: Dict[str, int]
    by_impact: Dict[str, int]

# API Endpoints

@router.post("/detect", response_model=EventsListResponse)
async def detect_big_events(
    request: EventDetectionRequest,
    background_tasks: BackgroundTasks,
    detector=Depends(get_big_event_detector),
    news_collector=Depends(get_enhanced_news_collector)
):
    """
    실시간 빅 이벤트 감지
    
    최신 뉴스를 수집하고 주요 시장 이벤트를 감지합니다.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info("Starting big event detection", 
                   symbol=request.symbol, 
                   hours_back=request.hours_back)
        
        # 1. 최신 뉴스 수집
        if request.symbol:
            news_items = await news_collector.collect_all_sources(
                symbol=request.symbol, 
                hours_back=request.hours_back
            )
        else:
            # 일반 뉴스 수집
            crypto_news = await news_collector.collect_all_sources("crypto", request.hours_back)
            market_news = await news_collector.collect_all_sources("market", request.hours_back) 
            news_items = crypto_news + market_news
        
        logger.info(f"Collected {len(news_items)} news items for event detection")
        
        # 2. 이벤트 감지 실행
        detected_events = await detector.detect_events(news_items)
        
        # 3. 필터링
        filtered_events = [
            event for event in detected_events
            if (event.confidence >= request.min_confidence and 
                event.final_impact_score >= request.min_impact)
        ]
        
        # 4. 응답 생성
        event_responses = []
        for event in filtered_events:
            event_response = EventResponse(
                event_id=event.event_id,
                event_type=event.event_type.value,
                title=event.title,
                description=event.description,
                impact_level=event.impact_level.value,
                base_impact_score=event.base_impact_score,
                sentiment_bias=event.sentiment_bias,
                volatility_factor=event.volatility_factor,
                final_impact_score=event.final_impact_score,
                confidence=event.confidence,
                detected_at=event.detected_at.isoformat(),
                event_time=event.event_time.isoformat() if event.event_time else None,
                source_urls=event.source_urls,
                keywords=event.keywords,
                symbols_affected=event.symbols_affected,
                news_count=event.news_count
            )
            event_responses.append(event_response)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # 백그라운드에서 이벤트 정리 작업
        background_tasks.add_task(detector.cleanup_old_events)
        
        logger.info("Big event detection completed",
                   total_events=len(detected_events),
                   filtered_events=len(filtered_events),
                   processing_time=processing_time)
        
        return EventsListResponse(
            events=event_responses,
            total_count=len(event_responses),
            processing_time=processing_time,
            metadata={
                "news_items_analyzed": len(news_items),
                "total_events_detected": len(detected_events),
                "min_confidence_applied": request.min_confidence,
                "min_impact_applied": request.min_impact,
                "symbol_filter": request.symbol,
                "hours_back": request.hours_back
            }
        )
        
    except Exception as e:
        logger.error("Big event detection failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Event detection failed: {str(e)}"
        )

@router.get("/active", response_model=EventsListResponse)
async def get_active_events(
    min_impact: float = Query(0.5, ge=0.0, le=2.0, description="최소 영향도"),
    max_age_hours: int = Query(24, ge=1, le=168, description="최대 이벤트 나이(시간)"),
    symbol: Optional[str] = Query(None, description="특정 심볼 필터"),
    detector=Depends(get_big_event_detector)
):
    """
    활성 이벤트 조회
    
    현재 활성 상태인 주요 이벤트들을 반환합니다.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info("Retrieving active events", 
                   min_impact=min_impact, 
                   max_age_hours=max_age_hours,
                   symbol=symbol)
        
        # 활성 이벤트 조회
        active_events = detector.get_active_events(min_impact, max_age_hours)
        
        # 심볼 필터링
        if symbol:
            active_events = [
                event for event in active_events 
                if symbol.upper() in event.symbols_affected
            ]
        
        # 응답 생성
        event_responses = []
        for event in active_events:
            event_response = EventResponse(
                event_id=event.event_id,
                event_type=event.event_type.value,
                title=event.title,
                description=event.description,
                impact_level=event.impact_level.value,
                base_impact_score=event.base_impact_score,
                sentiment_bias=event.sentiment_bias,
                volatility_factor=event.volatility_factor,
                final_impact_score=event.final_impact_score,
                confidence=event.confidence,
                detected_at=event.detected_at.isoformat(),
                event_time=event.event_time.isoformat() if event.event_time else None,
                source_urls=event.source_urls,
                keywords=event.keywords,
                symbols_affected=event.symbols_affected,
                news_count=event.news_count
            )
            event_responses.append(event_response)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        logger.info("Active events retrieved",
                   count=len(event_responses),
                   processing_time=processing_time)
        
        return EventsListResponse(
            events=event_responses,
            total_count=len(event_responses),
            processing_time=processing_time,
            metadata={
                "min_impact_filter": min_impact,
                "max_age_hours": max_age_hours,
                "symbol_filter": symbol
            }
        )
        
    except Exception as e:
        logger.error("Failed to retrieve active events", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve active events: {str(e)}"
        )

@router.get("/by-symbol/{symbol}", response_model=EventsListResponse)
async def get_events_by_symbol(
    symbol: str,
    limit: int = Query(10, ge=1, le=100, description="최대 반환 개수"),
    detector=Depends(get_big_event_detector)
):
    """
    심볼별 이벤트 조회
    
    특정 심볼과 관련된 이벤트들을 반환합니다.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info("Retrieving events by symbol", symbol=symbol, limit=limit)
        
        # 심볼별 이벤트 조회
        symbol_events = detector.get_events_by_symbol(symbol)
        
        # 제한 적용
        symbol_events = symbol_events[:limit]
        
        # 응답 생성
        event_responses = []
        for event in symbol_events:
            event_response = EventResponse(
                event_id=event.event_id,
                event_type=event.event_type.value,
                title=event.title,
                description=event.description,
                impact_level=event.impact_level.value,
                base_impact_score=event.base_impact_score,
                sentiment_bias=event.sentiment_bias,
                volatility_factor=event.volatility_factor,
                final_impact_score=event.final_impact_score,
                confidence=event.confidence,
                detected_at=event.detected_at.isoformat(),
                event_time=event.event_time.isoformat() if event.event_time else None,
                source_urls=event.source_urls,
                keywords=event.keywords,
                symbols_affected=event.symbols_affected,
                news_count=event.news_count
            )
            event_responses.append(event_response)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        logger.info("Events by symbol retrieved",
                   symbol=symbol,
                   count=len(event_responses),
                   processing_time=processing_time)
        
        return EventsListResponse(
            events=event_responses,
            total_count=len(event_responses),
            processing_time=processing_time,   
            metadata={
                "symbol": symbol.upper(),
                "limit_applied": limit
            }
        )
        
    except Exception as e:
        logger.error("Failed to retrieve events by symbol", 
                    symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve events for symbol {symbol}: {str(e)}"
        )

@router.get("/stats", response_model=EventStatsResponse)
async def get_event_stats(detector=Depends(get_big_event_detector)):
    """
    이벤트 감지 통계
    
    이벤트 감지기의 현재 통계를 반환합니다.
    """
    try:
        logger.info("Retrieving event detection stats")
        
        # 통계 조회
        stats = detector.get_detector_stats()
        
        # 활성 이벤트들 분석
        active_events = detector.get_active_events(min_impact=0.0, max_age_hours=24)
        
        by_type = {}
        by_impact = {}
        
        for event in active_events:
            event_type = event.event_type.value
            impact_level = event.impact_level.value
            
            by_type[event_type] = by_type.get(event_type, 0) + 1
            by_impact[impact_level] = by_impact.get(impact_level, 0) + 1
        
        logger.info("Event stats retrieved", 
                   active_events=len(active_events),
                   total_detected=stats.get("total_detected", 0))
        
        return EventStatsResponse(
            stats=stats,
            active_events=len(active_events),
            by_type=by_type,
            by_impact=by_impact
        )
        
    except Exception as e:
        logger.error("Failed to retrieve event stats", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve event stats: {str(e)}"
        )

@router.post("/cleanup")
async def cleanup_old_events(detector=Depends(get_big_event_detector)):
    """
    오래된 이벤트 정리
    
    만료된 이벤트들을 정리합니다.
    """
    try:
        logger.info("Starting event cleanup")
        
        cleaned_count = detector.cleanup_old_events()
        
        logger.info("Event cleanup completed", cleaned_count=cleaned_count)
        
        return SuccessResponse(
            success=True,
            message=f"Cleaned up {cleaned_count} old events",
            data={"cleaned_events": cleaned_count}
        )
        
    except Exception as e:
        logger.error("Event cleanup failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Event cleanup failed: {str(e)}"
        )

@router.get("/types")
async def get_supported_event_types():
    """
    지원되는 이벤트 유형 목록
    
    시스템이 감지할 수 있는 이벤트 유형들을 반환합니다.
    """
    try:
        event_types = []
        
        for event_type in EventType:
            event_types.append({
                "type": event_type.value,
                "name": event_type.name,
                "description": f"Event type: {event_type.value}"
            })
        
        impact_levels = []
        for impact in EventImpact:
            impact_levels.append({
                "level": impact.value,
                "name": impact.name,
                "description": f"Impact level: {impact.value}"
            })
        
        return SuccessResponse(
            success=True,
            message="Supported event types and impact levels",
            data={
                "event_types": event_types,
                "impact_levels": impact_levels,
                "total_types": len(event_types),
                "total_levels": len(impact_levels)
            }
        )
        
    except Exception as e:
        logger.error("Failed to retrieve event types", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve event types: {str(e)}"
        )