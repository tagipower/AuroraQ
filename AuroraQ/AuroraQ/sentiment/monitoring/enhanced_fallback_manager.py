#!/usr/bin/env python3
"""
Enhanced Fallback Manager for AuroraQ
AuroraQ 향상된 폴백 관리자 - 폴백 빈도 최적화 및 데이터 품질 향상
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class FallbackReason(Enum):
    """폴백 발생 원인"""
    DATA_CORRUPTION = "data_corruption"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout" 
    RATE_LIMIT = "rate_limit"
    VALIDATION_FAILED = "validation_failed"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"

class FallbackLevel(Enum):
    """폴백 수준"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    BASIC = "basic"
    EMERGENCY = "emergency"

class DataQualityLevel(Enum):
    """데이터 품질 수준"""
    EXCELLENT = "excellent"  # 90%+
    GOOD = "good"           # 80-90%
    ADEQUATE = "adequate"   # 70-80%
    POOR = "poor"           # 50-70%
    CRITICAL = "critical"   # <50%

@dataclass
class FallbackEvent:
    """폴백 이벤트"""
    timestamp: datetime
    component: str
    reason: FallbackReason
    level: FallbackLevel
    original_operation: str
    fallback_action: str
    success: bool
    recovery_time: float
    data_quality_impact: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FallbackStrategy:
    """폴백 전략"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: int
    max_retries: int = 3
    backoff_factor: float = 2.0
    timeout: float = 30.0
    quality_threshold: float = 0.5

class EnhancedFallbackManager:
    """향상된 폴백 관리자"""
    
    def __init__(self, 
                 target_fallback_rate: float = 0.6,
                 target_data_quality: float = 0.8,
                 monitoring_window: int = 300):  # 5분 윈도우
        """
        초기화
        
        Args:
            target_fallback_rate: 목표 폴백 비율 (60%)
            target_data_quality: 목표 데이터 품질 (80%)
            monitoring_window: 모니터링 윈도우 (초)
        """
        self.target_fallback_rate = target_fallback_rate
        self.target_data_quality = target_data_quality
        self.monitoring_window = monitoring_window
        
        # 폴백 이벤트 추적
        self.fallback_events: deque = deque(maxlen=1000)
        self.success_events: deque = deque(maxlen=1000)
        
        # 컴포넌트별 통계
        self.component_stats = defaultdict(lambda: {
            "total_operations": 0,
            "fallback_count": 0,
            "success_count": 0,
            "avg_quality": 0.0,
            "recent_failures": deque(maxlen=10)
        })
        
        # 예방적 조치
        self.predictive_thresholds = {
            "error_rate_threshold": 0.1,
            "response_time_threshold": 5.0,
            "quality_degradation_threshold": 0.2
        }
        
        # 폴백 전략 등록
        self.fallback_strategies: Dict[str, List[FallbackStrategy]] = defaultdict(list)
        self._register_default_strategies()
        
        # 모니터링 스레드
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Enhanced Fallback Manager initialized with targets: "
                   f"fallback_rate={target_fallback_rate:.1%}, "
                   f"data_quality={target_data_quality:.1%}")
    
    def _register_default_strategies(self):
        """기본 폴백 전략 등록"""
        
        # 뉴스 수집기 전략
        self.register_strategy(
            component="news_collector",
            strategy=FallbackStrategy(
                name="alternative_source",
                condition=lambda ctx: ctx.get("source_failed", False),
                action=self._switch_to_alternative_source,
                priority=1,
                quality_threshold=0.8
            )
        )
        
        self.register_strategy(
            component="news_collector",
            strategy=FallbackStrategy(
                name="cached_data",
                condition=lambda ctx: ctx.get("all_sources_failed", False),
                action=self._use_cached_data,
                priority=2,
                quality_threshold=0.6
            )
        )
        
        # 센티먼트 분석기 전략
        self.register_strategy(
            component="sentiment_analyzer",
            strategy=FallbackStrategy(
                name="rule_based_sentiment",
                condition=lambda ctx: ctx.get("model_unavailable", False),
                action=self._use_rule_based_sentiment,
                priority=1,
                quality_threshold=0.7
            )
        )
        
        self.register_strategy(
            component="sentiment_analyzer",
            strategy=FallbackStrategy(
                name="historical_average",
                condition=lambda ctx: ctx.get("processing_failed", False),
                action=self._use_historical_sentiment,
                priority=2,
                quality_threshold=0.5
            )
        )
        
        # 토픽 분류기 전략
        self.register_strategy(
            component="topic_classifier",
            strategy=FallbackStrategy(
                name="keyword_based_classification",
                condition=lambda ctx: ctx.get("classification_failed", False),
                action=self._use_keyword_classification,
                priority=1,
                quality_threshold=0.6
            )
        )
        
        # 전략 선택기 전략
        self.register_strategy(
            component="strategy_selector",
            strategy=FallbackStrategy(
                name="default_strategy",
                condition=lambda ctx: ctx.get("scoring_failed", False),
                action=self._use_default_strategy,
                priority=1,
                quality_threshold=0.7
            )
        )
    
    def register_strategy(self, component: str, strategy: FallbackStrategy):
        """폴백 전략 등록"""
        self.fallback_strategies[component].append(strategy)
        # 우선순위로 정렬
        self.fallback_strategies[component].sort(key=lambda s: s.priority)
        logger.debug(f"Registered fallback strategy '{strategy.name}' for component '{component}'")
    
    async def execute_with_fallback(self, 
                                    component: str,
                                    operation: str, 
                                    primary_func: Callable,
                                    context: Dict[str, Any],
                                    *args, **kwargs) -> Dict[str, Any]:
        """
        폴백과 함께 작업 실행
        
        Args:
            component: 컴포넌트 이름
            operation: 작업 이름
            primary_func: 주요 함수
            context: 실행 컨텍스트
            
        Returns:
            실행 결과와 메타데이터
        """
        start_time = time.time()
        
        # 예방적 검사
        if self._should_skip_primary(component, context):
            logger.warning(f"Skipping primary operation for {component} due to predictive analysis")
            return await self._execute_fallback(component, operation, context, 
                                              FallbackReason.PROCESSING_ERROR, 
                                              start_time, *args, **kwargs)
        
        try:
            # 주요 작업 실행
            result = await primary_func(*args, **kwargs)
            
            # 성공 기록
            execution_time = time.time() - start_time
            self._record_success(component, operation, execution_time, 1.0)
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "fallback_used": False,
                "data_quality": 1.0
            }
            
        except Exception as e:
            logger.warning(f"Primary operation failed for {component}.{operation}: {e}")
            
            # 폴백 실행
            reason = self._classify_error(e)
            return await self._execute_fallback(component, operation, context, 
                                              reason, start_time, *args, **kwargs)
    
    async def _execute_fallback(self, 
                               component: str,
                               operation: str, 
                               context: Dict[str, Any],
                               reason: FallbackReason,
                               start_time: float,
                               *args, **kwargs) -> Dict[str, Any]:
        """폴백 실행"""
        
        # 적합한 폴백 전략 찾기
        strategies = self.fallback_strategies.get(component, [])
        
        for strategy in strategies:
            if strategy.condition(context):
                try:
                    logger.info(f"Executing fallback strategy '{strategy.name}' for {component}")
                    
                    # 폴백 실행
                    fallback_result = await asyncio.wait_for(
                        strategy.action(context),
                        timeout=strategy.timeout
                    )
                    
                    execution_time = time.time() - start_time
                    data_quality = fallback_result.get("data_quality", strategy.quality_threshold)
                    
                    # 폴백 이벤트 기록
                    self._record_fallback(
                        component=component,
                        operation=operation,
                        reason=reason,
                        strategy_name=strategy.name,
                        success=True,
                        execution_time=execution_time,
                        data_quality=data_quality
                    )
                    
                    return {
                        "success": True,
                        "result": fallback_result.get("result"),
                        "execution_time": execution_time,
                        "fallback_used": True,
                        "fallback_strategy": strategy.name,
                        "data_quality": data_quality,
                        "fallback_reason": reason.value
                    }
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy '{strategy.name}' failed: {fallback_error}")
                    continue
        
        # 모든 폴백 실패
        execution_time = time.time() - start_time
        self._record_fallback(
            component=component,
            operation=operation,
            reason=reason,
            strategy_name="none",
            success=False,
            execution_time=execution_time,
            data_quality=0.0
        )
        
        return {
            "success": False,
            "result": None,
            "execution_time": execution_time,
            "fallback_used": True,
            "fallback_strategy": "none",
            "data_quality": 0.0,
            "fallback_reason": reason.value,
            "error": "All fallback strategies failed"
        }
    
    def _should_skip_primary(self, component: str, context: Dict[str, Any]) -> bool:
        """예방적 분석으로 주요 작업 스킵 여부 결정"""
        stats = self.component_stats[component]
        
        # 최근 실패율 체크
        recent_failures = len(stats["recent_failures"])
        if recent_failures >= 5:  # 최근 5번 연속 실패
            return True
        
        # 에러율 체크
        if stats["total_operations"] > 10:
            error_rate = stats["fallback_count"] / stats["total_operations"]
            if error_rate > self.predictive_thresholds["error_rate_threshold"] * 2:
                return True
        
        # 컨텍스트 기반 예측
        if context.get("system_load", 0) > 0.9:
            return True
            
        if context.get("available_memory", 1.0) < 0.1:
            return True
        
        return False
    
    def _classify_error(self, error: Exception) -> FallbackReason:
        """에러 분류"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FallbackReason.TIMEOUT
        elif "rate limit" in error_str:
            return FallbackReason.RATE_LIMIT
        elif "unavailable" in error_str or "connection" in error_str:
            return FallbackReason.SERVICE_UNAVAILABLE
        elif "memory" in error_str or "resource" in error_str:
            return FallbackReason.RESOURCE_EXHAUSTED
        elif "network" in error_str:
            return FallbackReason.NETWORK_ERROR
        elif "validation" in error_str:
            return FallbackReason.VALIDATION_FAILED
        else:
            return FallbackReason.PROCESSING_ERROR
    
    # 폴백 전략 구현
    async def _switch_to_alternative_source(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """대체 소스로 전환"""
        failed_source = context.get("failed_source", "unknown")
        alternative_sources = context.get("alternative_sources", ["fallback_source"])
        
        logger.info(f"Switching from {failed_source} to alternative sources: {alternative_sources}")
        
        # 시뮬레이션: 대체 소스에서 데이터 수집
        await asyncio.sleep(0.1)  # 네트워크 지연 시뮬레이션
        
        return {
            "result": {
                "source": alternative_sources[0] if alternative_sources else "fallback",
                "data": context.get("fallback_data", []),
                "count": len(context.get("fallback_data", [])),
                "quality_note": "Data from alternative source"
            },
            "data_quality": 0.85
        }
    
    async def _use_cached_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """캐시된 데이터 사용"""
        cached_data = context.get("cached_data", [])
        cache_age = context.get("cache_age", 3600)  # 1시간
        
        logger.info(f"Using cached data (age: {cache_age}s, items: {len(cached_data)})")
        
        # 캐시 나이에 따른 품질 조정
        if cache_age < 1800:  # 30분 미만
            quality = 0.8
        elif cache_age < 3600:  # 1시간 미만
            quality = 0.7
        else:  # 1시간 이상
            quality = 0.6
        
        return {
            "result": {
                "source": "cache",
                "data": cached_data,
                "count": len(cached_data),
                "cache_age": cache_age,
                "quality_note": f"Cached data (age: {cache_age}s)"
            },
            "data_quality": quality
        }
    
    async def _use_rule_based_sentiment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """규칙 기반 센티먼트 분석"""
        text = context.get("text", "")
        
        # 간단한 규칙 기반 센티먼트 분석
        positive_words = ["good", "great", "excellent", "positive", "up", "rise", "gain"]
        negative_words = ["bad", "terrible", "negative", "down", "fall", "loss", "drop"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = min(0.7, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = max(-0.7, -0.5 - (negative_count - positive_count) * 0.1)
        else:
            sentiment = 0.0
        
        logger.info(f"Rule-based sentiment: {sentiment:.3f} (pos: {positive_count}, neg: {negative_count})")
        
        return {
            "result": {
                "sentiment": sentiment,
                "confidence": 0.6,
                "method": "rule_based",
                "positive_words": positive_count,
                "negative_words": negative_count
            },
            "data_quality": 0.7
        }
    
    async def _use_historical_sentiment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """과거 센티먼트 평균 사용"""
        historical_sentiments = context.get("historical_sentiments", [0.0, 0.1, -0.05])
        
        if historical_sentiments:
            avg_sentiment = statistics.mean(historical_sentiments)
            confidence = 0.4  # 낮은 신뢰도
        else:
            avg_sentiment = 0.0
            confidence = 0.2
        
        logger.info(f"Historical sentiment average: {avg_sentiment:.3f} (samples: {len(historical_sentiments)})")
        
        return {
            "result": {
                "sentiment": avg_sentiment,
                "confidence": confidence,
                "method": "historical_average",
                "sample_count": len(historical_sentiments)
            },
            "data_quality": 0.5
        }
    
    async def _use_keyword_classification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """키워드 기반 토픽 분류"""
        title = context.get("title", "")
        content = context.get("content", "")
        
        # 간단한 키워드 기반 분류
        text = (title + " " + content).lower()
        
        topic_keywords = {
            "macro": ["fed", "federal reserve", "interest rate", "inflation", "gdp"],
            "regulation": ["sec", "regulation", "law", "policy", "compliance"],
            "technology": ["blockchain", "upgrade", "protocol", "development"],
            "market": ["price", "trading", "volume", "market", "rally"],
            "security": ["hack", "breach", "vulnerability", "security"]
        }
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            topic_scores[topic] = score
        
        # 최고 점수 토픽 선택
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            confidence = min(0.8, topic_scores[best_topic] * 0.2)
        else:
            best_topic = "other"
            confidence = 0.1
        
        logger.info(f"Keyword-based classification: {best_topic} (confidence: {confidence:.3f})")
        
        return {
            "result": {
                "topic": best_topic,
                "confidence": confidence,
                "method": "keyword_based",
                "topic_scores": topic_scores
            },
            "data_quality": 0.6
        }
    
    async def _use_default_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """기본 전략 사용"""
        available_strategies = context.get("available_strategies", ["conservative_strategy"])
        default_strategy = available_strategies[0] if available_strategies else "emergency_strategy"
        
        logger.info(f"Using default strategy: {default_strategy}")
        
        return {
            "result": {
                "strategy": default_strategy,
                "score": 0.5,  # 중간 점수
                "method": "default_fallback",
                "confidence": 0.7
            },
            "data_quality": 0.7
        }
    
    def _record_success(self, component: str, operation: str, execution_time: float, data_quality: float):
        """성공 기록"""
        self.success_events.append({
            "timestamp": datetime.now(),
            "component": component,
            "operation": operation,
            "execution_time": execution_time,
            "data_quality": data_quality
        })
        
        stats = self.component_stats[component]
        stats["total_operations"] += 1
        stats["success_count"] += 1
        
        # 최근 실패 목록 초기화 (성공했으므로)
        stats["recent_failures"].clear()
        
        # 평균 품질 업데이트
        self._update_avg_quality(component, data_quality)
    
    def _record_fallback(self, component: str, operation: str, reason: FallbackReason,
                        strategy_name: str, success: bool, execution_time: float, 
                        data_quality: float):
        """폴백 기록"""
        event = FallbackEvent(
            timestamp=datetime.now(),
            component=component,
            reason=reason,
            level=self._determine_fallback_level(data_quality),
            original_operation=operation,
            fallback_action=strategy_name,
            success=success,
            recovery_time=execution_time,
            data_quality_impact=1.0 - data_quality
        )
        
        self.fallback_events.append(event)
        
        stats = self.component_stats[component]
        stats["total_operations"] += 1
        stats["fallback_count"] += 1
        
        if not success:
            stats["recent_failures"].append(datetime.now())
        
        # 평균 품질 업데이트
        self._update_avg_quality(component, data_quality)
        
        logger.info(f"Recorded fallback event: {component}.{operation} -> {strategy_name} "
                   f"(success: {success}, quality: {data_quality:.2f})")
    
    def _update_avg_quality(self, component: str, new_quality: float):
        """평균 품질 업데이트"""
        stats = self.component_stats[component]
        total_ops = stats["total_operations"]
        
        if total_ops == 1:
            stats["avg_quality"] = new_quality
        else:
            # 이동 평균 계산
            stats["avg_quality"] = (stats["avg_quality"] * (total_ops - 1) + new_quality) / total_ops
    
    def _determine_fallback_level(self, data_quality: float) -> FallbackLevel:
        """데이터 품질에 따른 폴백 레벨 결정"""
        if data_quality >= 0.9:
            return FallbackLevel.NORMAL
        elif data_quality >= 0.7:
            return FallbackLevel.DEGRADED
        elif data_quality >= 0.5:
            return FallbackLevel.BASIC
        else:
            return FallbackLevel.EMERGENCY
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 반환"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.monitoring_window)
        
        # 최근 윈도우 내 이벤트 필터링
        recent_fallbacks = [e for e in self.fallback_events if e.timestamp >= window_start]
        recent_successes = [e for e in self.success_events if e["timestamp"] >= window_start]
        
        total_recent_ops = len(recent_fallbacks) + len(recent_successes)
        current_fallback_rate = len(recent_fallbacks) / total_recent_ops if total_recent_ops > 0 else 0
        
        # 평균 데이터 품질
        if recent_fallbacks or recent_successes:
            fallback_qualities = [1.0 - e.data_quality_impact for e in recent_fallbacks]
            success_qualities = [e["data_quality"] for e in recent_successes]
            all_qualities = fallback_qualities + success_qualities
            avg_quality = statistics.mean(all_qualities) if all_qualities else 1.0
        else:
            avg_quality = 1.0
        
        return {
            "current_fallback_rate": current_fallback_rate,
            "target_fallback_rate": self.target_fallback_rate,
            "current_data_quality": avg_quality,
            "target_data_quality": self.target_data_quality,
            "total_operations": total_recent_ops,
            "fallback_events": len(recent_fallbacks),
            "success_events": len(recent_successes),
            "fallback_rate_status": "GOOD" if current_fallback_rate <= self.target_fallback_rate else "NEEDS_IMPROVEMENT",
            "data_quality_status": "GOOD" if avg_quality >= self.target_data_quality else "NEEDS_IMPROVEMENT",
            "component_stats": dict(self.component_stats)
        }
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                metrics = self.get_current_metrics()
                
                # 임계값 검사 및 조정
                if metrics["fallback_rate_status"] == "NEEDS_IMPROVEMENT":
                    self._adjust_fallback_strategies()
                
                if metrics["data_quality_status"] == "NEEDS_IMPROVEMENT":
                    self._improve_data_quality()
                
                # 예방적 조치
                self._apply_predictive_measures()
                
                time.sleep(60)  # 1분마다 모니터링
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)
    
    def _adjust_fallback_strategies(self):
        """폴백 전략 조정"""
        logger.info("Adjusting fallback strategies to reduce fallback rate")
        
        # 예방적 임계값 강화
        self.predictive_thresholds["error_rate_threshold"] *= 0.8
        self.predictive_thresholds["response_time_threshold"] *= 0.9
        
        # 타임아웃 증가 (더 오래 기다려서 성공 확률 증가)
        for component_strategies in self.fallback_strategies.values():
            for strategy in component_strategies:
                strategy.timeout = min(strategy.timeout * 1.2, 60.0)
                strategy.max_retries = min(strategy.max_retries + 1, 5)
    
    def _improve_data_quality(self):
        """데이터 품질 개선"""
        logger.info("Improving data quality thresholds")
        
        # 품질 임계값 상향 조정
        for component_strategies in self.fallback_strategies.values():
            for strategy in component_strategies:
                strategy.quality_threshold = min(strategy.quality_threshold * 1.1, 0.95)
    
    def _apply_predictive_measures(self):
        """예방적 조치 적용"""
        # 컴포넌트별 성능 분석
        for component, stats in self.component_stats.items():
            if stats["total_operations"] > 20:
                fallback_rate = stats["fallback_count"] / stats["total_operations"]
                
                if fallback_rate > 0.7:  # 70% 이상 폴백
                    logger.warning(f"High fallback rate for {component}: {fallback_rate:.1%}")
                    # 해당 컴포넌트의 예방적 스킵 확률 증가
                    self.predictive_thresholds[f"{component}_skip_probability"] = 0.3
                
                if stats["avg_quality"] < 0.6:  # 60% 미만 품질
                    logger.warning(f"Low data quality for {component}: {stats['avg_quality']:.1%}")
    
    def get_improvement_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        metrics = self.get_current_metrics()
        
        if metrics["fallback_rate_status"] == "NEEDS_IMPROVEMENT":
            recommendations.append(
                f"폴백 비율이 목표치를 초과했습니다 ({metrics['current_fallback_rate']:.1%} > {self.target_fallback_rate:.1%}). "
                "주요 장애 원인을 분석하고 예방 조치를 강화하세요."
            )
        
        if metrics["data_quality_status"] == "NEEDS_IMPROVEMENT":
            recommendations.append(
                f"데이터 품질이 목표치에 미달했습니다 ({metrics['current_data_quality']:.1%} < {self.target_data_quality:.1%}). "
                "폴백 전략의 품질을 개선하세요."
            )
        
        # 컴포넌트별 권장사항
        for component, stats in metrics["component_stats"].items():
            if stats["total_operations"] > 10:
                fallback_rate = stats["fallback_count"] / stats["total_operations"]
                if fallback_rate > 0.8:
                    recommendations.append(f"{component} 컴포넌트의 폴백 비율이 매우 높습니다 ({fallback_rate:.1%}). 안정성 개선이 필요합니다.")
                
                if stats["avg_quality"] < 0.7:
                    recommendations.append(f"{component} 컴포넌트의 데이터 품질이 낮습니다 ({stats['avg_quality']:.1%}). 품질 향상 조치가 필요합니다.")
        
        if not recommendations:
            recommendations.append("현재 시스템이 목표 성능을 달성하고 있습니다. 지속적인 모니터링을 유지하세요.")
        
        return recommendations
    
    def shutdown(self):
        """폴백 매니저 종료"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Enhanced Fallback Manager shut down")

# 전역 폴백 매니저 인스턴스
_fallback_manager: Optional[EnhancedFallbackManager] = None

def get_fallback_manager() -> EnhancedFallbackManager:
    """전역 폴백 매니저 반환"""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = EnhancedFallbackManager()
    return _fallback_manager

async def execute_with_enhanced_fallback(component: str,
                                       operation: str,
                                       primary_func: Callable,
                                       context: Dict[str, Any],
                                       *args, **kwargs) -> Dict[str, Any]:
    """향상된 폴백과 함께 작업 실행 (편의 함수)"""
    manager = get_fallback_manager()
    return await manager.execute_with_fallback(component, operation, primary_func, context, *args, **kwargs)

# 테스트 코드
if __name__ == "__main__":
    async def test_enhanced_fallback_manager():
        """향상된 폴백 매니저 테스트"""
        print("=== Enhanced Fallback Manager Test ===\n")
        
        manager = EnhancedFallbackManager(
            target_fallback_rate=0.6,
            target_data_quality=0.8
        )
        
        # 테스트 시나리오
        async def successful_operation():
            await asyncio.sleep(0.1)
            return {"data": "success", "count": 10}
        
        async def failing_operation():
            await asyncio.sleep(0.05)
            raise Exception("Service unavailable")
        
        # 테스트 1: 성공적인 작업
        print("1. Testing successful operation...")
        result = await manager.execute_with_fallback(
            component="test_component",
            operation="test_operation",
            primary_func=successful_operation,
            context={}
        )
        print(f"   Result: {result['success']}, Quality: {result['data_quality']:.1%}")
        
        # 테스트 2: 실패하는 작업 (폴백 트리거)
        print("\n2. Testing failing operation with fallback...")
        result = await manager.execute_with_fallback(
            component="news_collector",
            operation="collect_news",
            primary_func=failing_operation,
            context={
                "source_failed": True,
                "alternative_sources": ["backup_source"],
                "fallback_data": [{"title": "Fallback news"}]
            }
        )
        print(f"   Result: {result['success']}, Fallback: {result['fallback_used']}, "
              f"Strategy: {result.get('fallback_strategy')}, Quality: {result['data_quality']:.1%}")
        
        # 테스트 3: 센티먼트 분석 폴백
        print("\n3. Testing sentiment analysis fallback...")
        result = await manager.execute_with_fallback(
            component="sentiment_analyzer",
            operation="analyze_sentiment",
            primary_func=failing_operation,
            context={
                "model_unavailable": True,
                "text": "This is great news for the market!"
            }
        )
        print(f"   Result: {result['success']}, Sentiment: {result['result']['sentiment']:.3f}, "
              f"Method: {result['result']['method']}, Quality: {result['data_quality']:.1%}")
        
        # 현재 메트릭 출력
        print("\n4. Current metrics:")
        metrics = manager.get_current_metrics()
        print(f"   Fallback Rate: {metrics['current_fallback_rate']:.1%} (target: {metrics['target_fallback_rate']:.1%})")
        print(f"   Data Quality: {metrics['current_data_quality']:.1%} (target: {metrics['target_data_quality']:.1%})")
        print(f"   Total Operations: {metrics['total_operations']}")
        
        # 권장사항 출력
        print("\n5. Improvement recommendations:")
        recommendations = manager.get_improvement_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        manager.shutdown()
        print("\n✅ Enhanced Fallback Manager test completed")
    
    # 테스트 실행
    asyncio.run(test_enhanced_fallback_manager())