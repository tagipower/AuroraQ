"""
VPS Metrics API Router
9패널 대시보드용 메트릭 API 엔드포인트

ONNX 센티먼트 서비스와 통합된 메트릭 제공 API
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import time
from datetime import datetime
import logging

# VPS 센티먼트 서비스 임포트
from ..models.advanced_keyword_scorer_vps import analyze_sentiment_vps, get_vps_performance_stats
from ..processors.advanced_fusion_manager_vps import analyze_fusion_vps, get_vps_fusion_stats

# 로거 설정
logger = logging.getLogger("VPSMetricsRouter")

# API 라우터 생성
router = APIRouter(prefix="/metrics", tags=["VPS Metrics"])

# 요청/응답 모델들
class SentimentAnalysisRequest(BaseModel):
    """센티먼트 분석 요청"""
    text: str = Field(..., description="분석할 텍스트")
    price_data: Optional[Dict[str, Any]] = Field(None, description="가격 데이터")
    volume_data: Optional[Dict[str, Any]] = Field(None, description="거래량 데이터")
    social_data: Optional[Dict[str, Any]] = Field(None, description="소셜 데이터")
    use_onnx: bool = Field(True, description="ONNX 모델 사용 여부")

class FusionAnalysisRequest(BaseModel):
    """융합 분석 요청"""
    text_data: List[str] = Field(..., description="텍스트 데이터 리스트")
    market_data: Dict[str, Any] = Field(..., description="시장 데이터")
    social_data: Optional[Dict[str, Any]] = Field(None, description="소셜 데이터")
    enable_ml: bool = Field(True, description="ML 예측 활성화")

class BatchAnalysisRequest(BaseModel):
    """배치 분석 요청"""
    requests: List[SentimentAnalysisRequest] = Field(..., description="분석 요청 리스트")
    batch_size: int = Field(50, description="배치 크기")

class DashboardMetricsResponse(BaseModel):
    """대시보드 메트릭 응답"""
    timestamp: datetime
    sentiment_fusion: Dict[str, Any]
    ml_predictions: Dict[str, Any]
    event_impact: Dict[str, Any]
    strategy_performance: Dict[str, Any]
    anomaly_detection: Dict[str, Any]
    network_analysis: Dict[str, Any]
    market_pulse: Dict[str, Any]
    system_intelligence: Dict[str, Any]
    live_feed: Dict[str, Any]
    processing_time: float
    cache_performance: Dict[str, Any]

# 글로벌 상태 관리
class MetricsState:
    def __init__(self):
        self.last_analysis_time = None
        self.cached_metrics = None
        self.cache_ttl = 300  # 5분 캐시
        self.analysis_count = 0
        self.error_count = 0

metrics_state = MetricsState()

# 의존성 함수들
async def get_metrics_state():
    """메트릭 상태 의존성"""
    return metrics_state

# API 엔드포인트들

@router.get("/health", summary="메트릭 서비스 헬스체크")
async def health_check():
    """메트릭 서비스 상태 확인"""
    try:
        # VPS 성능 통계 조회
        perf_stats = get_vps_performance_stats()
        fusion_stats = get_vps_fusion_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "VPS Metrics API",
            "version": "2.0.0",
            "performance": {
                "sentiment_service": perf_stats,
                "fusion_service": fusion_stats
            },
            "uptime": time.time() - (metrics_state.last_analysis_time or time.time())
        }
    except Exception as e:
        logger.error(f"헬스체크 실패: {e}")
        raise HTTPException(status_code=500, detail=f"서비스 상태 확인 실패: {str(e)}")

@router.post("/sentiment", summary="단일 센티먼트 분석")
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    state: MetricsState = Depends(get_metrics_state)
):
    """단일 텍스트에 대한 센티먼트 분석"""
    try:
        start_time = time.time()
        
        # 센티먼트 분석 실행
        result = await analyze_sentiment_vps(
            text=request.text,
            price_data=request.price_data,
            volume_data=request.volume_data,
            social_data=request.social_data
        )
        
        # 상태 업데이트
        state.analysis_count += 1
        state.last_analysis_time = time.time()
        
        # 결과 포맷팅
        response = {
            "success": True,
            "data": {
                "text_sentiment": result.text_sentiment,
                "price_action_sentiment": result.price_action_sentiment,
                "volume_sentiment": result.volume_sentiment,
                "social_engagement": result.social_engagement,
                "emotional_state": result.emotional_state.value,
                "market_regime": result.market_regime.value,
                "viral_score": result.viral_score,
                "network_effect": result.network_effect,
                "momentum": {
                    "1h": result.momentum_1h,
                    "4h": result.momentum_4h,
                    "24h": result.momentum_24h
                },
                "risk_indicators": {
                    "volatility_regime": result.volatility_regime,
                    "black_swan_probability": result.black_swan_probability,
                    "tail_risk": result.tail_risk,
                    "panic_indicator": result.panic_indicator
                },
                "confidence_metrics": {
                    "confidence_calibration": result.confidence_calibration,
                    "prediction_stability": result.prediction_stability,
                    "ensemble_agreement": result.ensemble_agreement
                },
                "onnx_metadata": {
                    "model_version": result.onnx_model_version,
                    "inference_time": result.onnx_inference_time,
                    "memory_usage": result.onnx_memory_usage,
                    "cache_hit": result.cache_hit
                }
            },
            "metadata": {
                "processing_time": result.processing_time,
                "timestamp": result.timestamp.isoformat(),
                "request_id": f"sent_{int(time.time() * 1000)}"
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        state.error_count += 1
        logger.error(f"센티먼트 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"센티먼트 분석 실패: {str(e)}")

@router.post("/fusion", summary="고급 융합 분석")
async def analyze_fusion(
    request: FusionAnalysisRequest,
    state: MetricsState = Depends(get_metrics_state)
):
    """고급 융합 분석 (9패널 대시보드용)"""
    try:
        start_time = time.time()
        
        # 융합 분석 실행
        result = await analyze_fusion_vps(
            text_data=request.text_data,
            market_data=request.market_data,
            social_data=request.social_data
        )
        
        # 상태 업데이트
        state.analysis_count += 1
        state.last_analysis_time = time.time()
        
        # 결과 포맷팅
        response = {
            "success": True,
            "data": {
                "fusion_score": result.fusion_score,
                "ml_prediction": {
                    "direction": result.ml_refined_prediction.direction,
                    "confidence": result.ml_refined_prediction.confidence,
                    "timeframe": result.ml_refined_prediction.timeframe,
                    "risk_level": result.ml_refined_prediction.risk_level,
                    "model_version": result.ml_refined_prediction.model_version,
                    "inference_time": result.ml_refined_prediction.inference_time,
                    "feature_importance": result.ml_refined_prediction.feature_importance
                },
                "event_impact": {
                    "event_type": result.event_impact.event_type,
                    "impact_score": result.event_impact.impact_score,
                    "duration_hours": result.event_impact.duration_hours,
                    "affected_markets": result.event_impact.affected_markets,
                    "spillover_risk": result.event_impact.spillover_risk,
                    "sentiment_shift": result.event_impact.sentiment_shift,
                    "volume_impact": result.event_impact.volume_impact,
                    "volatility_impact": result.event_impact.volatility_impact
                },
                "strategy_performance": {
                    "strategy_name": result.strategy_performance.strategy_name,
                    "total_return": result.strategy_performance.total_return,
                    "sharpe_ratio": result.strategy_performance.sharpe_ratio,
                    "max_drawdown": result.strategy_performance.max_drawdown,
                    "win_rate": result.strategy_performance.win_rate,
                    "risk_metrics": {
                        "var_95": result.strategy_performance.var_95,
                        "expected_shortfall": result.strategy_performance.expected_shortfall,
                        "beta": result.strategy_performance.beta,
                        "alpha": result.strategy_performance.alpha
                    }
                },
                "anomaly_detection": {
                    "anomaly_score": result.anomaly_detection.anomaly_score,
                    "anomaly_type": result.anomaly_detection.anomaly_type,
                    "severity": result.anomaly_detection.severity,
                    "description": result.anomaly_detection.description,
                    "statistical_indicators": {
                        "z_score": result.anomaly_detection.z_score,
                        "p_value": result.anomaly_detection.p_value,
                        "threshold_breach": result.anomaly_detection.threshold_breach
                    }
                },
                "advanced_features": {
                    "semantic_density": result.advanced_features.semantic_density,
                    "sentiment_volatility": result.advanced_features.sentiment_volatility,
                    "information_flow": result.advanced_features.information_flow,
                    "social_amplification": result.advanced_features.social_amplification,
                    "attention_score": result.advanced_features.attention_score,
                    "credibility_score": result.advanced_features.credibility_score,
                    "regime_probability": result.advanced_features.regime_probability
                },
                "timeseries_features": result.timeseries_features,
                "network_features": result.network_graph_features,
                "meta_learning": result.meta_learning_features,
                "quality_metrics": {
                    "feature_quality_score": result.feature_quality_score,
                    "completeness_ratio": result.completeness_ratio
                }
            },
            "metadata": {
                "processing_time": result.processing_time,
                "memory_usage": result.memory_usage,
                "cache_hits": result.cache_hits,
                "onnx_calls": result.onnx_calls,
                "timestamp": result.timestamp.isoformat(),
                "feature_version": result.feature_version,
                "request_id": f"fusion_{int(time.time() * 1000)}"
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        state.error_count += 1
        logger.error(f"융합 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"융합 분석 실패: {str(e)}")

@router.post("/batch", summary="배치 센티먼트 분석")
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    state: MetricsState = Depends(get_metrics_state)
):
    """배치 센티먼트 분석"""
    try:
        start_time = time.time()
        
        # 배치 처리
        batch_results = []
        batch_size = min(request.batch_size, 100)  # 최대 100개 제한
        
        for i in range(0, len(request.requests), batch_size):
            batch = request.requests[i:i + batch_size]
            
            # 병렬 처리
            tasks = [
                analyze_sentiment_vps(
                    text=req.text,
                    price_data=req.price_data,
                    volume_data=req.volume_data,
                    social_data=req.social_data
                )
                for req in batch
            ]
            
            batch_results.extend(await asyncio.gather(*tasks, return_exceptions=True))
        
        # 결과 처리
        successful_results = []
        error_count = 0
        
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                error_count += 1
                logger.error(f"배치 항목 {idx} 분석 실패: {result}")
                continue
            
            successful_results.append({
                "index": idx,
                "sentiment_score": result.text_sentiment,
                "emotional_state": result.emotional_state.value,
                "market_regime": result.market_regime.value,
                "confidence": result.confidence_calibration,
                "processing_time": result.processing_time,
                "cache_hit": result.cache_hit
            })
        
        # 상태 업데이트
        state.analysis_count += len(successful_results)
        state.error_count += error_count
        state.last_analysis_time = time.time()
        
        response = {
            "success": True,
            "data": {
                "total_requests": len(request.requests),
                "successful_analyses": len(successful_results),
                "failed_analyses": error_count,
                "results": successful_results,
                "summary": {
                    "avg_sentiment": sum(r["sentiment_score"] for r in successful_results) / max(len(successful_results), 1),
                    "avg_confidence": sum(r["confidence"] for r in successful_results) / max(len(successful_results), 1),
                    "cache_hit_rate": sum(1 for r in successful_results if r["cache_hit"]) / max(len(successful_results), 1),
                    "avg_processing_time": sum(r["processing_time"] for r in successful_results) / max(len(successful_results), 1)
                }
            },
            "metadata": {
                "total_processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "batch_size": batch_size,
                "request_id": f"batch_{int(time.time() * 1000)}"
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        state.error_count += 1
        logger.error(f"배치 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 분석 실패: {str(e)}")

@router.get("/dashboard", summary="9패널 대시보드 메트릭")
async def get_dashboard_metrics(
    state: MetricsState = Depends(get_metrics_state)
):
    """9패널 대시보드용 통합 메트릭"""
    try:
        # 캐시 확인
        current_time = time.time()
        if (state.cached_metrics and 
            state.last_analysis_time and 
            current_time - state.last_analysis_time < state.cache_ttl):
            
            cached_response = state.cached_metrics.copy()
            cached_response["cache_performance"]["cache_hit"] = True
            return JSONResponse(content=cached_response)
        
        start_time = time.time()
        
        # 성능 통계 조회
        sentiment_stats = get_vps_performance_stats()
        fusion_stats = get_vps_fusion_stats()
        
        # 모의 시장 데이터 (실제로는 외부 데이터 소스에서 가져옴)
        mock_market_data = {
            "price": {"current": 50000, "previous": 49500, "change_1h": 1.0, "change_4h": 2.0, "change_24h": -1.5},
            "volume": {"current": 1500000, "average": 1200000},
            "price_history": [49000, 49200, 49500, 49800, 50000],
            "volume_history": [1100000, 1200000, 1300000, 1400000, 1500000]
        }
        
        mock_text_data = ["Market showing bullish momentum", "Strong buying pressure observed"]
        
        # 융합 분석 실행 (샘플 데이터)
        fusion_result = await analyze_fusion_vps(
            text_data=mock_text_data,
            market_data=mock_market_data
        )
        
        # 대시보드 메트릭 구성
        dashboard_metrics = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "panels": {
                "1_sentiment_fusion": {
                    "title": "ONNX Sentiment Fusion",
                    "fusion_score": fusion_result.fusion_score,
                    "sentiment_breakdown": {
                        "text": 0.3,  # 모의 데이터
                        "price": 0.2,
                        "volume": 0.1,
                        "social": 0.05
                    },
                    "onnx_metadata": {
                        "model_version": "v1.0",
                        "inference_time": 0.15,
                        "memory_usage": 150.0
                    }
                },
                "2_ml_predictions": {
                    "title": "ONNX ML Predictions",
                    "direction": fusion_result.ml_refined_prediction.direction,
                    "confidence": fusion_result.ml_refined_prediction.confidence,
                    "risk_level": fusion_result.ml_refined_prediction.risk_level,
                    "feature_importance": fusion_result.ml_refined_prediction.feature_importance
                },
                "3_event_impact": {
                    "title": "Event Impact Analysis",
                    "event_type": fusion_result.event_impact.event_type,
                    "impact_score": fusion_result.event_impact.impact_score,
                    "affected_markets": fusion_result.event_impact.affected_markets,
                    "spillover_risk": fusion_result.event_impact.spillover_risk
                },
                "4_strategy_performance": {
                    "title": "ONNX Strategy Performance",
                    "total_return": fusion_result.strategy_performance.total_return,
                    "sharpe_ratio": fusion_result.strategy_performance.sharpe_ratio,
                    "win_rate": fusion_result.strategy_performance.win_rate,
                    "max_drawdown": fusion_result.strategy_performance.max_drawdown
                },
                "5_anomaly_detection": {
                    "title": "ONNX Anomaly Detection",
                    "anomaly_score": fusion_result.anomaly_detection.anomaly_score,
                    "severity": fusion_result.anomaly_detection.severity,
                    "anomaly_type": fusion_result.anomaly_detection.anomaly_type,
                    "description": fusion_result.anomaly_detection.description
                },
                "6_network_analysis": {
                    "title": "Network Analysis",
                    "network_density": fusion_result.network_graph_features.get("network_density", 0.0),
                    "influence_spread": fusion_result.network_graph_features.get("influence_spread", 0.0),
                    "community_strength": fusion_result.network_graph_features.get("community_strength", 0.0)
                },
                "7_market_pulse": {
                    "title": "Market Pulse",
                    "trend_strength": fusion_result.advanced_features.trend_strength,
                    "volatility": fusion_result.advanced_features.sentiment_volatility,
                    "momentum_24h": 0.02,  # 모의 데이터
                    "volume_ratio": 1.25
                },
                "8_system_intelligence": {
                    "title": "ONNX System Intelligence",
                    "model_confidence": fusion_result.meta_learning_features.get("model_confidence", 0.8),
                    "prediction_consistency": fusion_result.meta_learning_features.get("prediction_consistency", 0.7),
                    "ensemble_agreement": fusion_result.meta_learning_features.get("ensemble_agreement", 0.75),
                    "memory_usage": fusion_result.memory_usage,
                    "processing_efficiency": 1.0 / max(fusion_result.processing_time, 0.001)
                },
                "9_live_feed": {
                    "title": "ONNX Live Feed",
                    "latest_analysis_time": fusion_result.timestamp.isoformat(),
                    "total_analyses": state.analysis_count,
                    "error_rate": state.error_count / max(state.analysis_count, 1),
                    "cache_performance": sentiment_stats.get("cache_stats", {}),
                    "onnx_calls": fusion_result.onnx_calls,
                    "avg_response_time": sentiment_stats.get("performance", {}).get("avg_processing_time", 0)
                }
            },
            "cache_performance": {
                "cache_hit": False,
                "cache_age": 0,
                "cache_ttl": state.cache_ttl,
                "generation_time": time.time() - start_time
            },
            "metadata": {
                "processing_time": time.time() - start_time,
                "feature_version": fusion_result.feature_version,
                "quality_score": fusion_result.feature_quality_score,
                "completeness": fusion_result.completeness_ratio,
                "request_id": f"dash_{int(time.time() * 1000)}"
            }
        }
        
        # 캐시 업데이트
        state.cached_metrics = dashboard_metrics
        state.last_analysis_time = current_time
        
        return JSONResponse(content=dashboard_metrics)
        
    except Exception as e:
        state.error_count += 1
        logger.error(f"대시보드 메트릭 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"대시보드 메트릭 생성 실패: {str(e)}")

@router.get("/stats", summary="서비스 통계")
async def get_service_stats(state: MetricsState = Depends(get_metrics_state)):
    """서비스 성능 통계"""
    try:
        sentiment_stats = get_vps_performance_stats()
        fusion_stats = get_vps_fusion_stats()
        
        return {
            "service_stats": {
                "total_analyses": state.analysis_count,
                "error_count": state.error_count,
                "error_rate": state.error_count / max(state.analysis_count, 1),
                "uptime": time.time() - (metrics_state.last_analysis_time or time.time()),
                "cache_ttl": state.cache_ttl
            },
            "sentiment_service": sentiment_stats,
            "fusion_service": fusion_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@router.post("/cache/clear", summary="캐시 정리")
async def clear_cache(state: MetricsState = Depends(get_metrics_state)):
    """캐시 정리"""
    try:
        # VPS 캐시 정리
        from ..models.advanced_keyword_scorer_vps import clear_vps_cache
        from ..processors.advanced_fusion_manager_vps import cleanup_vps_fusion
        
        clear_vps_cache()
        cleanup_vps_fusion()
        
        # 상태 캐시 정리
        state.cached_metrics = None
        state.last_analysis_time = None
        
        return {
            "success": True,
            "message": "캐시가 성공적으로 정리되었습니다",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"캐시 정리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"캐시 정리 실패: {str(e)}")