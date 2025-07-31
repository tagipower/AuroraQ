#!/usr/bin/env python3
"""
Trading Router for AuroraQ Sentiment Service
실전 및 가상 매매 연동 API 엔드포인트
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from ..dependencies import get_aurora_adapter
from ...integrations.aurora_adapter import TradingSignal, TradingMode, SignalStrength, MarketContext
from ...models import SuccessResponse, ErrorResponse

logger = structlog.get_logger(__name__)

router = APIRouter()

# Request/Response Models
class SignalGenerationRequest(BaseModel):
    """신호 생성 요청"""
    symbol: str = Field(..., description="거래 심볼 (예: BTC, ETH)")
    text_content: Optional[str] = Field(None, description="분석할 텍스트 내용")
    trading_mode: TradingMode = Field(TradingMode.PAPER, description="매매 모드")
    send_to_aurora: bool = Field(True, description="AuroraQ로 신호 전송 여부")

class BatchSignalRequest(BaseModel):
    """배치 신호 생성 요청"""
    symbols: List[str] = Field(..., description="심볼 목록")
    trading_mode: TradingMode = Field(TradingMode.PAPER, description="매매 모드")
    send_to_aurora: bool = Field(True, description="AuroraQ로 신호 전송 여부")

class TradingSignalResponse(BaseModel):
    """매매 신호 응답"""
    symbol: str
    signal_type: str
    strength: str
    confidence: float
    sentiment_score: float
    sentiment_direction: str
    price_target: Optional[float]
    stop_loss: Optional[float]
    risk_level: str
    news_count: int
    big_events: List[str]
    fusion_method: str
    processing_time: float
    created_at: str
    expires_at: str

class SignalResponse(BaseModel):
    """신호 응답"""
    success: bool = True
    signal: Optional[TradingSignalResponse]
    message: str
    processing_time: float
    metadata: Dict[str, Any]

class SignalListResponse(BaseModel):
    """신호 목록 응답"""
    success: bool = True
    signals: List[TradingSignalResponse]
    total_count: int
    processing_time: float
    metadata: Dict[str, Any]

class AdapterStatsResponse(BaseModel):
    """어댑터 통계 응답"""
    success: bool = True
    stats: Dict[str, Any]
    health_status: Dict[str, Any]

# Helper Functions
def _convert_trading_signal(signal: TradingSignal) -> TradingSignalResponse:
    """TradingSignal을 TradingSignalResponse로 변환"""
    return TradingSignalResponse(
        symbol=signal.symbol,
        signal_type=signal.signal_type,
        strength=signal.strength.value,
        confidence=signal.confidence,
        sentiment_score=signal.sentiment_score,
        sentiment_direction=signal.sentiment_direction,
        price_target=signal.price_target,
        stop_loss=signal.stop_loss,
        risk_level=signal.risk_level,
        news_count=signal.news_count,
        big_events=signal.big_events,
        fusion_method=signal.fusion_method,
        processing_time=signal.processing_time,
        created_at=signal.created_at.isoformat(),
        expires_at=signal.expires_at.isoformat()
    )

# API Endpoints

@router.post("/signal/generate", response_model=SignalResponse)
async def generate_trading_signal(
    request: SignalGenerationRequest,
    background_tasks: BackgroundTasks,
    adapter=Depends(get_aurora_adapter)
):
    """
    단일 심볼에 대한 매매 신호 생성
    
    최신 뉴스와 감정 분석을 기반으로 매매 신호를 생성합니다.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info("Generating trading signal",
                   symbol=request.symbol,
                   trading_mode=request.trading_mode,
                   has_text=bool(request.text_content))
        
        # 매매 신호 생성
        trading_signal = await adapter.generate_trading_signal(
            symbol=request.symbol,
            text_content=request.text_content
        )
        
        if not trading_signal:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to generate signal for {request.symbol}. Insufficient data or low confidence."
            )
        
        # AuroraQ로 신호 전송 (요청 시)
        send_success = True
        if request.send_to_aurora:
            send_success = await adapter.send_signal_to_aurora(
                trading_signal, request.trading_mode
            )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # 응답 생성
        signal_response = _convert_trading_signal(trading_signal)
        
        message = f"Signal generated for {request.symbol}: {trading_signal.signal_type.upper()}"
        if request.send_to_aurora:
            message += f" (sent to AuroraQ: {'success' if send_success else 'failed'})"
        
        logger.info("Trading signal generated successfully",
                   symbol=request.symbol,
                   signal_type=trading_signal.signal_type,
                   strength=trading_signal.strength.value,
                   confidence=trading_signal.confidence)
        
        return SignalResponse(
            signal=signal_response,
            message=message,
            processing_time=processing_time,
            metadata={
                "trading_mode": request.trading_mode.value,
                "aurora_send_attempted": request.send_to_aurora,
                "aurora_send_success": send_success if request.send_to_aurora else None,
                "has_text_content": bool(request.text_content)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate trading signal",
                    symbol=request.symbol,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate trading signal: {str(e)}"
        )

@router.post("/signal/batch", response_model=SignalListResponse)
async def generate_batch_signals(
    request: BatchSignalRequest,
    background_tasks: BackgroundTasks,
    adapter=Depends(get_aurora_adapter)
):
    """
    여러 심볼에 대한 배치 매매 신호 생성
    
    여러 심볼을 동시에 처리하여 매매 신호를 생성합니다.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info("Generating batch trading signals",
                   symbols=request.symbols,
                   count=len(request.symbols),
                   trading_mode=request.trading_mode)
        
        # 배치 처리
        trading_signals = await adapter.process_symbol_batch(
            symbols=request.symbols,
            trading_mode=request.trading_mode,
            send_to_aurora=request.send_to_aurora
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # 응답 생성
        signal_responses = [_convert_trading_signal(signal) for signal in trading_signals]
        
        logger.info("Batch trading signals generated",
                   requested=len(request.symbols),
                   generated=len(trading_signals),
                   processing_time=processing_time)
        
        return SignalListResponse(
            signals=signal_responses,
            total_count=len(signal_responses),
            processing_time=processing_time,
            metadata={
                "requested_symbols": len(request.symbols),
                "successful_signals": len(trading_signals),
                "trading_mode": request.trading_mode.value,
                "aurora_send_attempted": request.send_to_aurora
            }
        )
        
    except Exception as e:
        logger.error("Failed to generate batch trading signals",
                    symbols=request.symbols,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate batch trading signals: {str(e)}"
        )

@router.get("/signal/history", response_model=SignalListResponse)
async def get_signal_history(
    symbol: Optional[str] = Query(None, description="특정 심볼 필터"),
    hours_back: int = Query(24, ge=1, le=168, description="과거 시간 범위"),
    signal_type: Optional[str] = Query(None, regex="^(buy|sell|hold)$", description="신호 유형 필터"),
    limit: int = Query(50, ge=1, le=500, description="최대 반환 개수"),
    adapter=Depends(get_aurora_adapter)
):
    """
    매매 신호 히스토리 조회
    
    과거 생성된 매매 신호들을 조회합니다.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info("Retrieving signal history",
                   symbol=symbol,
                   hours_back=hours_back,
                   signal_type=signal_type,
                   limit=limit)
        
        # 히스토리 조회
        signal_history = adapter.get_signal_history(
            symbol=symbol,
            hours_back=hours_back,
            signal_type=signal_type
        )
        
        # 제한 적용
        limited_signals = signal_history[:limit]
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # 응답 생성
        signal_responses = [_convert_trading_signal(signal) for signal in limited_signals]
        
        logger.info("Signal history retrieved",
                   total_found=len(signal_history),
                   returned=len(limited_signals),
                   processing_time=processing_time)
        
        return SignalListResponse(
            signals=signal_responses,
            total_count=len(signal_responses),
            processing_time=processing_time,
            metadata={
                "symbol_filter": symbol,
                "hours_back": hours_back,
                "signal_type_filter": signal_type,
                "limit_applied": limit,
                "total_found": len(signal_history)
            }
        )
        
    except Exception as e:
        logger.error("Failed to retrieve signal history",
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve signal history: {str(e)}"
        )

@router.get("/stats", response_model=AdapterStatsResponse)
async def get_adapter_stats(adapter=Depends(get_aurora_adapter)):
    """
    어댑터 통계 및 상태
    
    AuroraQ 어댑터의 현재 통계와 상태를 반환합니다.
    """
    try:
        logger.info("Retrieving adapter stats")
        
        # 통계 및 헬스체크
        stats = adapter.get_adapter_stats()
        health_status = await adapter.health_check()
        
        logger.info("Adapter stats retrieved",
                   signals_generated=stats.get("signals_generated", 0),
                   adapter_status=health_status.get("adapter_status"))
        
        return AdapterStatsResponse(
            stats=stats,
            health_status=health_status
        )
        
    except Exception as e:
        logger.error("Failed to retrieve adapter stats",
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve adapter stats: {str(e)}"
        )

@router.post("/test-connection")
async def test_aurora_connection(adapter=Depends(get_aurora_adapter)):
    """
    AuroraQ 연결 테스트
    
    AuroraQ 메인 시스템과의 연결을 테스트합니다.
    """
    try:
        logger.info("Testing AuroraQ connection")
        
        # 헬스체크를 통한 연결 테스트
        health_status = await adapter.health_check()
        
        aurora_status = health_status.get("aurora_connectivity", "unknown")
        connection_ok = aurora_status == "healthy"
        
        logger.info("AuroraQ connection test completed",
                   status=aurora_status,
                   success=connection_ok)
        
        return SuccessResponse(
            success=connection_ok,
            message=f"AuroraQ connection status: {aurora_status}",
            data={
                "connection_status": aurora_status,
                "api_calls": adapter.stats.get("aurora_api_calls", 0),
                "api_errors": adapter.stats.get("aurora_api_errors", 0),
                "full_health_check": health_status
            }
        )
        
    except Exception as e:
        logger.error("AuroraQ connection test failed",
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Connection test failed: {str(e)}"
        )

@router.get("/signal-strength-levels")
async def get_signal_strength_levels():
    """
    신호 강도 레벨 정보
    
    사용 가능한 신호 강도 레벨들을 반환합니다.
    """
    try:
        strength_levels = []
        
        for strength in SignalStrength:
            strength_levels.append({
                "level": strength.value,
                "name": strength.name,
                "description": f"Signal strength: {strength.value}"
            })
        
        trading_modes = []
        for mode in TradingMode:
            trading_modes.append({
                "mode": mode.value,
                "name": mode.name,
                "description": f"Trading mode: {mode.value}"
            })
        
        return SuccessResponse(
            success=True,
            message="Signal strength levels and trading modes",
            data={
                "signal_strengths": strength_levels,
                "trading_modes": trading_modes,
                "signal_types": ["buy", "sell", "hold"],
                "risk_levels": ["low", "medium", "high"]
            }
        )
        
    except Exception as e:
        logger.error("Failed to retrieve signal strength levels",
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve signal strength levels: {str(e)}"
        )

@router.post("/signal/manual")
async def manual_signal_analysis(
    symbol: str = Query(..., description="분석할 심볼"),
    text: str = Query(..., description="분석할 텍스트"),
    adapter=Depends(get_aurora_adapter)
):
    """
    수동 텍스트 분석
    
    사용자가 제공한 텍스트를 즉시 분석하여 매매 신호를 생성합니다.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info("Manual signal analysis requested",
                   symbol=symbol,
                   text_length=len(text))
        
        # 수동 분석
        trading_signal = await adapter.generate_trading_signal(
            symbol=symbol,
            text_content=text
        )
        
        if not trading_signal:
            raise HTTPException(
                status_code=400,
                detail="Unable to analyze the provided text. Please check the content."
            )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # 응답 생성
        signal_response = _convert_trading_signal(trading_signal)
        
        logger.info("Manual signal analysis completed",
                   symbol=symbol,
                   signal_type=trading_signal.signal_type,
                   confidence=trading_signal.confidence)
        
        return SignalResponse(
            signal=signal_response,
            message=f"Manual analysis completed for {symbol}",
            processing_time=processing_time,
            metadata={
                "analysis_mode": "manual",
                "text_length": len(text),
                "sentiment_score": trading_signal.sentiment_score,
                "confidence": trading_signal.confidence
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Manual signal analysis failed",
                    symbol=symbol,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Manual analysis failed: {str(e)}"
        )