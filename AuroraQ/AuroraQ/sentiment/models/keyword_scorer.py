#!/usr/bin/env python3
"""
Enhanced Keyword Scorer with Unified Logging
통합 로깅이 적용된 향상된 키워드 스코어링 시스템
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# 통합 로깅 시스템 (try import with fallback)
try:
    from ...aurora_logging import get_vps_log_integrator, LogCategory, LogLevel
except (ImportError, ValueError):
    # Fallback for standalone or testing execution
    def get_vps_log_integrator():
        return None
    
    class MockLogIntegrator:
        async def log_onnx_inference(self, **kwargs): pass
        async def log_batch_processing(self, **kwargs): pass
        async def log_security_event(self, **kwargs): pass
        async def log_system_metrics(self, **kwargs): pass
        def get_logger(self, name): 
            import logging
            return logging.getLogger(name)
    
    def get_vps_log_integrator():
        return MockLogIntegrator()
    
    class LogCategory:
        pass
    class LogLevel:
        pass

# 기존 VPS 모듈 (호환성 유지)
try:
    from .advanced_keyword_scorer_vps import analyze_sentiment_vps, get_vps_performance_stats
except ImportError:
    # 폴백 구현
    async def analyze_sentiment_vps(text: str, **kwargs) -> Dict[str, Any]:
        return {"sentiment": 0.0, "confidence": 0.5, "processing_time": 0.1}
    
    async def get_vps_performance_stats() -> Dict[str, Any]:
        return {"memory_usage": 0.0, "cpu_usage": 0.0}

class KeywordScorer:
    """기본 키워드 스코어러"""
    
    def __init__(self):
        self.keywords = {
            'positive': {'good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain'},
            'negative': {'bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'drop'},
            'neutral': {'neutral', 'stable', 'unchanged', 'flat', 'same'}
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """기본 키워드 분석"""
        if not text:
            return {"sentiment": 0.0, "confidence": 0.0, "keywords": []}
        
        text_lower = text.lower()
        pos_count = sum(1 for word in self.keywords['positive'] if word in text_lower)
        neg_count = sum(1 for word in self.keywords['negative'] if word in text_lower)
        
        total_count = pos_count + neg_count
        if total_count == 0:
            return {"sentiment": 0.0, "confidence": 0.0, "keywords": []}
        
        sentiment = (pos_count - neg_count) / total_count
        confidence = total_count / len(text.split())
        
        return {
            "sentiment": sentiment,
            "confidence": min(confidence, 1.0),
            "keywords": []
        }

class UnifiedKeywordScorer:
    """통합 로깅이 적용된 키워드 스코어링 시스템"""
    
    def __init__(self, 
                 model_path: str = "/app/models/finbert.onnx",
                 enable_logging: bool = True):
        """
        초기화
        
        Args:
            model_path: ONNX 모델 경로
            enable_logging: 통합 로깅 활성화
        """
        self.model_path = model_path
        self.enable_logging = enable_logging
        
        # 통합 로깅 시스템
        if enable_logging:
            self.log_integrator = get_vps_log_integrator()
            self.logger = self.log_integrator.get_logger("keyword_scorer")
        else:
            self.log_integrator = None
            self.logger = None
        
        # 성능 통계
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0,
            "last_update": datetime.now()
        }
    
    async def analyze_sentiment_unified(self, 
                                      text: str,
                                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        통합 로깅이 적용된 센티먼트 분석
        
        Args:
            text: 분석할 텍스트
            metadata: 추가 메타데이터
            
        Returns:
            Dict: 분석 결과
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # 기존 VPS 분석 수행
            result = await analyze_sentiment_vps(text, metadata=metadata or {})
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            # 성능 통계 업데이트
            self.stats["successful_requests"] += 1
            self._update_avg_processing_time(processing_time)
            
            # 통합 로깅 (Training 범주 - 학습 데이터로 활용)
            if self.log_integrator:
                await self.log_integrator.log_onnx_inference(
                    text=text,
                    inference_time=processing_time,
                    confidence=result.get("confidence", 0.0),
                    model_version="finbert_onnx_unified",
                    sentiment_score=result.get("sentiment", 0.0),
                    metadata=metadata or {}
                )
            
            # Raw 로깅 (디버깅용)
            if self.logger:
                self.logger.info(
                    f"Sentiment analysis completed: {result.get('confidence', 0):.4f} confidence",
                    text_length=len(text),
                    processing_time=processing_time,
                    sentiment=result.get("sentiment", 0)
                )
            
            return result
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            
            # 에러 로깅
            if self.logger:
                self.logger.error(
                    f"Sentiment analysis failed: {str(e)}",
                    text_length=len(text),
                    error_type=type(e).__name__,
                    processing_time=time.time() - start_time
                )
            
            # 보안 이벤트로 로깅 (critical 에러인 경우)
            if self.log_integrator and "security" in str(e).lower():
                await self.log_integrator.log_security_event(
                    event_type="analysis_failure",
                    severity="medium",
                    description=f"Sentiment analysis security issue: {str(e)}",
                    text_length=len(text),
                    error_details=str(e)
                )
            
            raise
    
    async def batch_analyze_unified(self, 
                                  texts: List[str],
                                  batch_size: int = 32,
                                  metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        통합 로깅이 적용된 배치 분석
        
        Args:
            texts: 분석할 텍스트 목록
            batch_size: 배치 크기
            metadata: 추가 메타데이터
            
        Returns:
            List[Dict]: 분석 결과 목록
        """
        start_time = time.time()
        total_texts = len(texts)
        results = []
        success_count = 0
        error_count = 0
        
        try:
            # 배치 단위로 처리
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = []
                
                # 개별 텍스트 분석
                for text in batch_texts:
                    try:
                        result = await self.analyze_sentiment_unified(text, metadata)
                        batch_results.append(result)
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        batch_results.append({
                            "error": str(e),
                            "sentiment": 0.0,
                            "confidence": 0.0,
                            "processing_time": 0.0
                        })
                
                results.extend(batch_results)
            
            # 전체 처리 시간
            total_processing_time = time.time() - start_time
            
            # 배치 처리 로깅 (Summary 범주)
            if self.log_integrator:
                await self.log_integrator.log_batch_processing(
                    batch_size=total_texts,
                    processing_time=total_processing_time,
                    success_count=success_count,
                    error_count=error_count,
                    avg_processing_time=total_processing_time / total_texts if total_texts > 0 else 0,
                    metadata=metadata or {}
                )
            
            return results
            
        except Exception as e:
            # 배치 전체 실패 로깅
            if self.logger:
                self.logger.error(
                    f"Batch analysis failed: {str(e)}",
                    total_texts=total_texts,
                    batch_size=batch_size,
                    processing_time=time.time() - start_time
                )
            raise
    
    async def get_performance_metrics_unified(self) -> Dict[str, Any]:
        """통합 로깅이 적용된 성능 메트릭"""
        try:
            # VPS 시스템 메트릭
            vps_stats = await get_vps_performance_stats()
            
            # 통합 메트릭
            unified_metrics = {
                **self.stats,
                **vps_stats,
                "success_rate": (self.stats["successful_requests"] / 
                               max(self.stats["total_requests"], 1)) * 100,
                "error_rate": (self.stats["failed_requests"] / 
                              max(self.stats["total_requests"], 1)) * 100,
                "last_update": datetime.now().isoformat()
            }
            
            # 시스템 메트릭 로깅 (Summary 범주)
            if self.log_integrator:
                await self.log_integrator.log_system_metrics(
                    component="keyword_scorer",
                    metrics=unified_metrics
                )
            
            return unified_metrics
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get performance metrics: {str(e)}")
            return self.stats
    
    def _update_avg_processing_time(self, new_time: float):
        """평균 처리 시간 업데이트"""
        current_avg = self.stats["avg_processing_time"]
        total_requests = self.stats["successful_requests"]
        
        if total_requests == 1:
            self.stats["avg_processing_time"] = new_time
        else:
            # 지수 이동 평균 사용
            alpha = 0.1  # 가중치
            self.stats["avg_processing_time"] = (alpha * new_time + 
                                               (1 - alpha) * current_avg)
    
    async def health_check_unified(self) -> Dict[str, Any]:
        """통합 로깅이 적용된 헬스 체크"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "logging_enabled": self.enable_logging,
            "stats": self.stats
        }
        
        try:
            # 간단한 테스트 분석
            test_result = await analyze_sentiment_vps("Test health check", {})
            health_status["test_analysis"] = {
                "success": True,
                "processing_time": test_result.get("processing_time", 0)
            }
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["test_analysis"] = {
                "success": False,
                "error": str(e)
            }
            
            # 헬스 체크 실패를 보안 이벤트로 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="health_check_failure",
                    severity="high",
                    description=f"Keyword scorer health check failed: {str(e)}",
                    component="keyword_scorer",
                    error_details=str(e)
                )
        
        return health_status

# 호환성을 위한 팩토리 함수
def create_unified_keyword_scorer(model_path: str = "/app/models/finbert.onnx",
                                enable_logging: bool = True) -> UnifiedKeywordScorer:
    """통합 키워드 스코어러 생성"""
    return UnifiedKeywordScorer(model_path=model_path, enable_logging=enable_logging)

# 기존 함수들과의 호환성 어댑터
_global_scorer: Optional[UnifiedKeywordScorer] = None

async def analyze_sentiment_unified(text: str, **kwargs) -> Dict[str, Any]:
    """기존 함수와 호환되는 통합 분석 함수"""
    global _global_scorer
    
    if _global_scorer is None:
        _global_scorer = create_unified_keyword_scorer()
    
    return await _global_scorer.analyze_sentiment_unified(text, kwargs.get('metadata'))

async def get_unified_performance_stats() -> Dict[str, Any]:
    """기존 함수와 호환되는 통합 성능 통계"""
    global _global_scorer
    
    if _global_scorer is None:
        _global_scorer = create_unified_keyword_scorer()
    
    return await _global_scorer.get_performance_metrics_unified()

if __name__ == "__main__":
    # 테스트 실행
    async def test_unified_scorer():
        scorer = create_unified_keyword_scorer()
        
        # 단일 분석 테스트
        result = await scorer.analyze_sentiment_unified(
            "Bitcoin price is rising significantly today!",
            metadata={"source": "test", "symbol": "BTC"}
        )
        print("Single analysis:", result)
        
        # 배치 분석 테스트
        texts = [
            "Market is bullish on cryptocurrency",
            "Bearish sentiment in traditional markets", 
            "Neutral outlook for the week"
        ]
        batch_results = await scorer.batch_analyze_unified(texts)
        print("Batch analysis:", len(batch_results), "results")
        
        # 성능 메트릭
        metrics = await scorer.get_performance_metrics_unified()
        print("Performance metrics:", metrics)
        
        # 헬스 체크
        health = await scorer.health_check_unified()
        print("Health status:", health["status"])
    
    asyncio.run(test_unified_scorer())