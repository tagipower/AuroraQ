#!/usr/bin/env python3
"""
Sentiment Fusion Manager V2 for AuroraQ Sentiment Service
VPS 최적화 버전 - 메모리 효율성, 캐시 최적화, 성능 개선
"""

import asyncio
import time
import statistics
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import hashlib
from collections import deque, defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class FusionMethod(Enum):
    """융합 방법"""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    ADAPTIVE = "adaptive"
    ENSEMBLE = "ensemble"
    SMART_FALLBACK = "smart_fallback"  # 신규 추가

class SignalSource(Enum):
    """신호 소스"""
    KEYWORD = "keyword"
    FINBERT = "finbert"
    FUSION = "fusion"
    TECHNICAL = "technical"
    SOCIAL = "social"
    NEWS = "news"

class SentimentDirection(Enum):
    """감정 방향"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VERY_BULLISH = "very_bullish"
    VERY_BEARISH = "very_bearish"
    UNKNOWN = "unknown"

@dataclass
class SentimentSignalV2:
    """개선된 감정 신호"""
    source: SignalSource
    score: float  # -1.0 ~ 1.0
    confidence: float  # 0.0 ~ 1.0
    timestamp: datetime
    processing_time: float
    reliability: float = 0.8  # 소스별 기본 신뢰도
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """후처리 - 값 범위 검증"""
        self.score = max(-1.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.reliability = max(0.0, min(1.0, self.reliability))

@dataclass
class FusedSentimentV2:
    """개선된 융합된 감정 결과"""
    content_hash: str
    final_score: float  # -1.0 ~ 1.0
    final_confidence: float  # 0.0 ~ 1.0
    direction: SentimentDirection
    intensity: float  # 0.0 ~ 1.0 (감정 강도)
    signals: List[SentimentSignalV2]
    fusion_method: FusionMethod
    quality_score: float
    reliability_score: float
    consensus_score: float  # 신호 간 합의 정도
    processing_metadata: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (VPS 메모리 최적화)"""
        return {
            "content_hash": self.content_hash,
            "final_score": round(self.final_score, 4),
            "final_confidence": round(self.final_confidence, 4),
            "direction": self.direction.value,
            "intensity": round(self.intensity, 3),
            "signals_count": len(self.signals),
            "signal_sources": [s.source.value for s in self.signals],
            "fusion_method": self.fusion_method.value,
            "quality_score": round(self.quality_score, 4),
            "reliability_score": round(self.reliability_score, 4),
            "consensus_score": round(self.consensus_score, 4),
            "processing_time": self.processing_metadata.get("processing_time_total", 0),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        return datetime.now() > self.expires_at

class AdaptiveCacheManager:
    """적응형 캐시 매니저 (메모리 효율적)"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, FusedSentimentV2] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = {}
        self.lock = threading.RLock()
        
        # 통계
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired_cleanups": 0
        }
    
    def get(self, key: str) -> Optional[FusedSentimentV2]:
        """캐시에서 조회"""
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            result = self.cache[key]
            
            # 만료 체크
            if result.is_expired():
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
                if key in self.last_access:
                    del self.last_access[key]
                self.stats["expired_cleanups"] += 1
                self.stats["misses"] += 1
                return None
            
            # 액세스 정보 업데이트
            self.access_count[key] += 1
            self.last_access[key] = time.time()
            self.stats["hits"] += 1
            
            return result
    
    def put(self, key: str, value: FusedSentimentV2):
        """캐시에 저장"""
        with self.lock:
            # 크기 제한 체크
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_count[key] = 1
            self.last_access[key] = time.time()
    
    def _evict_lru(self):
        """LRU 방식으로 항목 제거"""
        if not self.cache:
            return
        
        # 가장 적게 사용되고 오래된 항목 찾기
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.access_count[k], self.last_access.get(k, 0))
        )
        
        del self.cache[lru_key]
        del self.access_count[lru_key]
        if lru_key in self.last_access:
            del self.last_access[lru_key]
        
        self.stats["evictions"] += 1
    
    def cleanup_expired(self) -> int:
        """만료된 항목 정리"""
        with self.lock:
            expired_keys = []
            for key, value in self.cache.items():
                if value.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
                if key in self.last_access:
                    del self.last_access[key]
            
            self.stats["expired_cleanups"] += len(expired_keys)
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        with self.lock:
            hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) if (self.stats["hits"] + self.stats["misses"]) > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": round(hit_rate, 3),
                **self.stats
            }

class SentimentFusionManagerV2:
    """개선된 감정 융합 관리자 V2"""
    
    def __init__(self,
                 cache_size: int = 1000,
                 cache_ttl: int = 300,  # 5분
                 enable_adaptive_weights: bool = True):
        """
        초기화 (VPS 최적화)
        
        Args:
            cache_size: 캐시 최대 크기
            cache_ttl: 캐시 TTL (초)
            enable_adaptive_weights: 적응형 가중치 활성화
        """
        self.cache_manager = AdaptiveCacheManager(cache_size, cache_ttl)
        self.enable_adaptive_weights = enable_adaptive_weights
        
        # 소스별 기본 가중치 (VPS 환경에 맞게 조정)
        self.base_weights = {
            SignalSource.FINBERT: 0.6,     # 높은 정확도
            SignalSource.KEYWORD: 0.4,     # 빠른 응답
            SignalSource.TECHNICAL: 0.3,   # 기술적 지표
            SignalSource.SOCIAL: 0.2,      # 소셜 미디어
            SignalSource.NEWS: 0.5         # 뉴스 분석
        }
        
        # 신뢰도 임계값
        self.confidence_threshold = 0.5  # VPS 환경에 맞게 낮춤
        self.quality_threshold = 0.4
        self.consensus_threshold = 0.7
        
        # 이상치 제거 설정
        self.outlier_z_threshold = 2.5  # 더 관대하게 설정
        self.min_signals_for_outlier_removal = 3
        
        # 심볼별 실시간 집계 캐시
        self.symbol_cache: Dict[str, Dict] = {}
        self.symbol_cache_ttl = 180  # 3분
        
        # 성능 통계
        self.stats = {
            "total_fused": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_fusion_time": 0.0,
            "avg_signals_per_fusion": 0.0,
            "outliers_removed": 0,
            "adaptive_adjustments": 0,
            "source_usage": defaultdict(int),
            "direction_distribution": defaultdict(int)
        }
        
        # 백그라운드 정리 작업
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # 스레드 안전성
        self.fusion_lock = threading.RLock()
    
    async def start(self):
        """매니저 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Sentiment Fusion Manager V2 started")
    
    async def stop(self):
        """매니저 중지"""
        self.is_running = False
        
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Sentiment Fusion Manager V2 stopped")
    
    async def _cleanup_loop(self):
        """백그라운드 정리 루프"""
        while self.is_running:
            try:
                # 5분마다 정리
                await asyncio.sleep(300)
                
                if not self.is_running:
                    break
                
                # 만료된 캐시 정리
                expired_count = self.cache_manager.cleanup_expired()
                
                # 심볼 캐시 정리
                self._cleanup_symbol_cache()
                
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired cache entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    def _cleanup_symbol_cache(self):
        """심볼 캐시 정리"""
        current_time = time.time()
        expired_symbols = []
        
        for symbol, data in self.symbol_cache.items():
            if current_time - data.get('timestamp', 0) > self.symbol_cache_ttl:
                expired_symbols.append(symbol)
        
        for symbol in expired_symbols:
            del self.symbol_cache[symbol]
    
    def _calculate_adaptive_weights(self, signals: List[SentimentSignalV2]) -> Dict[SignalSource, float]:
        """적응형 가중치 계산"""
        if not self.enable_adaptive_weights or len(signals) < 2:
            return self.base_weights
        
        weights = self.base_weights.copy()
        
        # 신뢰도 기반 조정
        for signal in signals:
            source = signal.source
            confidence_boost = (signal.confidence - 0.5) * 0.2  # ±0.1 범위
            reliability_boost = (signal.reliability - 0.8) * 0.1  # ±0.02 범위
            
            if source in weights:
                weights[source] += confidence_boost + reliability_boost
                weights[source] = max(0.1, min(1.0, weights[source]))  # 범위 제한
        
        # 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        self.stats["adaptive_adjustments"] += 1
        return weights
    
    def _remove_outliers_smart(self, signals: List[SentimentSignalV2]) -> List[SentimentSignalV2]:
        """스마트 이상치 제거"""
        if len(signals) < self.min_signals_for_outlier_removal:
            return signals
        
        scores = [s.score for s in signals]
        
        # Z-score 계산
        mean_score = statistics.mean(scores)
        try:
            std_score = statistics.stdev(scores)
        except statistics.StatisticsError:
            return signals  # 표준편차가 0인 경우
        
        if std_score == 0:
            return signals
        
        filtered_signals = []
        removed_count = 0
        
        for signal in signals:
            z_score = abs((signal.score - mean_score) / std_score)
            
            # 높은 신뢰도 신호는 덜 엄격하게 필터링
            threshold = self.outlier_z_threshold
            if signal.confidence > 0.8:
                threshold *= 1.2  # 20% 더 관대하게
            
            if z_score <= threshold:
                filtered_signals.append(signal)
            else:
                logger.debug(f"Removed outlier: {signal.source.value} score={signal.score:.3f} z={z_score:.2f}")
                removed_count += 1
        
        if removed_count > 0:
            self.stats["outliers_removed"] += removed_count
        
        return filtered_signals if filtered_signals else signals[:1]  # 최소 1개는 유지
    
    def _calculate_consensus_score(self, signals: List[SentimentSignalV2]) -> float:
        """신호 간 합의 점수 계산"""
        if len(signals) < 2:
            return 1.0
        
        scores = [s.score for s in signals]
        
        # 표준편차 기반 합의 점수 (낮을수록 높은 합의)
        try:
            std_dev = statistics.stdev(scores)
            # 0.5 표준편차에서 0점, 0 표준편차에서 1점
            consensus = max(0.0, 1.0 - (std_dev / 0.5))
        except statistics.StatisticsError:
            consensus = 1.0
        
        # 방향 일치도도 고려
        directions = [self._score_to_direction(s.score) for s in signals]
        direction_counts = defaultdict(int)
        for direction in directions:
            direction_counts[direction] += 1
        
        max_direction_count = max(direction_counts.values())
        direction_consensus = max_direction_count / len(directions)
        
        # 두 점수의 가중 평균
        final_consensus = (consensus * 0.6 + direction_consensus * 0.4)
        return max(0.0, min(1.0, final_consensus))
    
    def _score_to_direction(self, score: float) -> SentimentDirection:
        """점수를 방향으로 변환"""
        if score > 0.5:
            return SentimentDirection.VERY_BULLISH
        elif score > 0.1:
            return SentimentDirection.BULLISH
        elif score < -0.5:
            return SentimentDirection.VERY_BEARISH
        elif score < -0.1:
            return SentimentDirection.BEARISH
        else:
            return SentimentDirection.NEUTRAL
    
    def _calculate_intensity(self, score: float, confidence: float) -> float:
        """감정 강도 계산"""
        # 절댓값과 신뢰도를 결합
        abs_score = abs(score)
        return (abs_score * 0.7 + confidence * 0.3)
    
    def _execute_fusion_v2(self, 
                          signals: List[SentimentSignalV2], 
                          method: FusionMethod) -> Tuple[float, float]:
        """개선된 융합 실행"""
        
        if not signals:
            return 0.0, 0.0
        
        if len(signals) == 1:
            return signals[0].score, signals[0].confidence
        
        # 이상치 제거
        signals = self._remove_outliers_smart(signals)
        
        scores = [s.score for s in signals]
        confidences = [s.confidence for s in signals]
        
        if method == FusionMethod.SMART_FALLBACK:
            # 신호 품질에 따라 자동 선택
            avg_confidence = statistics.mean(confidences)
            if avg_confidence > 0.8 and len(signals) >= 3:
                method = FusionMethod.ENSEMBLE
            elif avg_confidence > 0.6:
                method = FusionMethod.CONFIDENCE_BASED
            else:
                method = FusionMethod.WEIGHTED_AVERAGE
        
        # 가중치 계산
        if method == FusionMethod.WEIGHTED_AVERAGE:
            weights = [1.0] * len(signals)
            
        elif method == FusionMethod.CONFIDENCE_BASED:
            weights = [self._sigmoid_weight(c) for c in confidences]
            
        elif method == FusionMethod.ADAPTIVE:
            adaptive_weights = self._calculate_adaptive_weights(signals)
            weights = [adaptive_weights.get(s.source, 0.5) for s in signals]
            
        elif method == FusionMethod.ENSEMBLE:
            # 복합 가중치 (소스 + 신뢰도 + 처리시간)
            weights = []
            for signal in signals:
                base_weight = self.base_weights.get(signal.source, 0.5)
                confidence_weight = self._sigmoid_weight(signal.confidence)
                
                # 처리 시간 패널티 (빠를수록 좋음)
                time_weight = max(0.5, 1.0 - (signal.processing_time / 10.0))
                
                combined_weight = base_weight * confidence_weight * time_weight
                weights.append(combined_weight)
        
        else:
            weights = [1.0] * len(signals)
        
        # 정규화
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0, 0.0
        
        normalized_weights = [w / total_weight for w in weights]
        
        # 최종 계산
        final_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        final_confidence = sum(conf * weight for conf, weight in zip(confidences, normalized_weights))
        
        # 범위 제한
        final_score = max(-1.0, min(1.0, final_score))
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return final_score, final_confidence
    
    def _sigmoid_weight(self, confidence: float) -> float:
        """시그모이드 가중치 계산"""
        # 0.5 이하는 가중치 감소, 0.5 이상은 증가
        return 1 / (1 + np.exp(-8 * (confidence - 0.5)))
    
    def _calculate_quality_score_v2(self, signals: List[SentimentSignalV2]) -> float:
        """개선된 품질 점수 계산"""
        if not signals:
            return 0.0
        
        factors = []
        
        # 1. 신호 개수 점수
        signal_count_score = min(1.0, len(signals) / 4.0)  # 4개가 최적
        factors.append(signal_count_score)
        
        # 2. 평균 신뢰도
        avg_confidence = statistics.mean(s.confidence for s in signals)
        factors.append(avg_confidence)
        
        # 3. 평균 신뢰성
        avg_reliability = statistics.mean(s.reliability for s in signals)
        factors.append(avg_reliability)
        
        # 4. 소스 다양성
        unique_sources = len(set(s.source for s in signals))
        diversity_score = min(1.0, unique_sources / 3.0)  # 3개 소스가 최적
        factors.append(diversity_score)
        
        # 5. 처리 시간 품질 (빠를수록 좋음)
        avg_processing_time = statistics.mean(s.processing_time for s in signals)
        time_quality = max(0.0, 1.0 - (avg_processing_time / 5.0))  # 5초 기준
        factors.append(time_quality)
        
        # 가중 평균
        weights = [0.2, 0.3, 0.2, 0.15, 0.15]
        quality_score = sum(factor * weight for factor, weight in zip(factors, weights))
        
        return max(0.0, min(1.0, quality_score))
    
    async def fuse_sentiment_v2(self,
                               content_hash: str,
                               signals: List[SentimentSignalV2],
                               method: FusionMethod = FusionMethod.SMART_FALLBACK,
                               ttl_override: Optional[int] = None) -> FusedSentimentV2:
        """개선된 감정 융합"""
        
        start_time = time.time()
        
        # 캐시 확인
        cached_result = self.cache_manager.get(content_hash)
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        self.stats["cache_misses"] += 1
        
        with self.fusion_lock:
            logger.debug(f"Fusing sentiment for: {content_hash} with {len(signals)} signals")
            
            if not signals:
                # 빈 신호 처리
                empty_result = FusedSentimentV2(
                    content_hash=content_hash,
                    final_score=0.0,
                    final_confidence=0.0,
                    direction=SentimentDirection.UNKNOWN,
                    intensity=0.0,
                    signals=[],
                    fusion_method=method,
                    quality_score=0.0,
                    reliability_score=0.0,
                    consensus_score=0.0,
                    processing_metadata={"signals_count": 0, "processing_time_total": 0.0},
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=ttl_override or 60)
                )
                self.cache_manager.put(content_hash, empty_result)
                return empty_result
            
            # 융합 실행
            final_score, final_confidence = self._execute_fusion_v2(signals, method)
            
            # 방향 및 강도 계산
            direction = self._score_to_direction(final_score)
            intensity = self._calculate_intensity(final_score, final_confidence)
            
            # 품질 및 신뢰성 점수
            quality_score = self._calculate_quality_score_v2(signals)
            reliability_score = statistics.mean(s.reliability for s in signals)
            consensus_score = self._calculate_consensus_score(signals)
            
            # 처리 메타데이터
            processing_metadata = {
                "signals_count": len(signals),
                "fusion_method_used": method.value,
                "processing_time_total": time.time() - start_time,
                "source_breakdown": {source.value: sum(1 for s in signals if s.source == source) 
                                   for source in set(s.source for s in signals)},
                "avg_processing_time": statistics.mean(s.processing_time for s in signals),
                "outliers_removed": self.stats.get("outliers_removed", 0)
            }
            
            # TTL 계산 (품질에 따라 조정)
            base_ttl = ttl_override or 300
            quality_multiplier = 0.5 + (quality_score * 0.5)  # 0.5 ~ 1.0
            adjusted_ttl = int(base_ttl * quality_multiplier)
            
            # 결과 생성
            fused_result = FusedSentimentV2(
                content_hash=content_hash,
                final_score=final_score,
                final_confidence=final_confidence,
                direction=direction,
                intensity=intensity,
                signals=signals,
                fusion_method=method,
                quality_score=quality_score,
                reliability_score=reliability_score,
                consensus_score=consensus_score,
                processing_metadata=processing_metadata,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=adjusted_ttl)
            )
            
            # 캐시 저장
            self.cache_manager.put(content_hash, fused_result)
            
            # 통계 업데이트
            self._update_stats_v2(signals, time.time() - start_time, direction)
            
            logger.debug(f"Sentiment fusion completed: {direction.value} "
                        f"({final_score:.3f}, {final_confidence:.3f}) "
                        f"from {len(signals)} signals in {time.time() - start_time:.3f}s")
            
            return fused_result
    
    def _update_stats_v2(self, signals: List[SentimentSignalV2], processing_time: float, direction: SentimentDirection):
        """개선된 통계 업데이트"""
        self.stats["total_fused"] += 1
        
        # 평균 처리 시간
        current_avg = self.stats["avg_fusion_time"]
        total_count = self.stats["total_fused"]
        self.stats["avg_fusion_time"] = (
            (current_avg * (total_count - 1) + processing_time) / total_count
        )
        
        # 평균 신호 수
        current_signal_avg = self.stats["avg_signals_per_fusion"]
        self.stats["avg_signals_per_fusion"] = (
            (current_signal_avg * (total_count - 1) + len(signals)) / total_count
        )
        
        # 소스별 사용량
        for signal in signals:
            self.stats["source_usage"][signal.source.value] += 1
        
        # 방향별 분포
        self.stats["direction_distribution"][direction.value] += 1
    
    async def get_symbol_sentiment(self,
                                 symbol: str,
                                 hours_back: int = 24,
                                 min_quality: float = 0.3) -> Dict[str, Any]:
        """심볼별 감정 집계 (캐시 활용)"""
        
        cache_key = f"{symbol}_{hours_back}_{min_quality}"
        
        # 심볼 캐시 확인
        if cache_key in self.symbol_cache:
            cache_data = self.symbol_cache[cache_key]
            if time.time() - cache_data['timestamp'] < self.symbol_cache_ttl:
                return cache_data['result']
        
        # 집계 계산
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        relevant_results = []
        
        # 캐시된 융합 결과에서 관련 항목 찾기
        for key, result in self.cache_manager.cache.items():
            if result.created_at >= cutoff_time and result.quality_score >= min_quality:
                # 심볼 매칭 로직 (메타데이터나 해시에서 추출)
                if self._is_symbol_relevant(result, symbol):
                    relevant_results.append(result)
        
        if not relevant_results:
            empty_result = {
                "symbol": symbol.upper(),
                "score": 0.0,
                "confidence": 0.0,
                "direction": SentimentDirection.NEUTRAL.value,
                "intensity": 0.0,
                "consensus": 0.0,
                "quality": 0.0,
                "count": 0,
                "trend": "stable",
                "last_updated": None,
                "time_range_hours": hours_back
            }
            
            # 빈 결과도 캐시 (빈번한 재계산 방지)
            self.symbol_cache[cache_key] = {
                'result': empty_result,
                'timestamp': time.time()
            }
            
            return empty_result
        
        # 집계 계산
        scores = [r.final_score for r in relevant_results]
        confidences = [r.final_confidence for r in relevant_results]
        intensities = [r.intensity for r in relevant_results]
        qualities = [r.quality_score for r in relevant_results]
        consensus_scores = [r.consensus_score for r in relevant_results]
        
        # 가중 평균 (최근 결과에 더 높은 가중치)
        now = datetime.now()
        weighted_scores = []
        weights = []
        
        for result in relevant_results:
            age_hours = (now - result.created_at).total_seconds() / 3600
            # 최근 1시간은 가중치 1.0, 24시간 후는 0.3
            weight = max(0.3, 1.0 - (age_hours / 24.0) * 0.7)
            weights.append(weight)
            weighted_scores.append(result.final_score * weight)
        
        total_weight = sum(weights)
        if total_weight > 0:
            avg_score = sum(weighted_scores) / total_weight
        else:
            avg_score = statistics.mean(scores)
        
        # 기타 통계
        avg_confidence = statistics.mean(confidences)
        avg_intensity = statistics.mean(intensities)
        avg_quality = statistics.mean(qualities)
        avg_consensus = statistics.mean(consensus_scores)
        
        # 트렌드 계산
        trend = self._calculate_trend(relevant_results)
        
        # 방향 결정
        direction = self._score_to_direction(avg_score)
        
        result = {
            "symbol": symbol.upper(),
            "score": round(avg_score, 3),
            "confidence": round(avg_confidence, 3),
            "direction": direction.value,
            "intensity": round(avg_intensity, 3),
            "consensus": round(avg_consensus, 3),
            "quality": round(avg_quality, 3),
            "count": len(relevant_results),
            "trend": trend,
            "last_updated": max(r.created_at for r in relevant_results).isoformat(),
            "time_range_hours": hours_back
        }
        
        # 캐시 저장
        self.symbol_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        return result
    
    def _is_symbol_relevant(self, result: FusedSentimentV2, symbol: str) -> bool:
        """결과가 특정 심볼과 관련있는지 확인"""
        # 간단한 구현 - 실제로는 더 정교한 매칭 로직 필요
        symbol_lower = symbol.lower()
        
        # 메타데이터에서 확인
        for signal in result.signals:
            metadata = signal.metadata
            if metadata.get('symbol', '').lower() == symbol_lower:
                return True
            
            # 키워드에서 확인
            keywords = metadata.get('matched_keywords', [])
            if any(symbol_lower in keyword.lower() for keyword in keywords):
                return True
        
        return False
    
    def _calculate_trend(self, results: List[FusedSentimentV2]) -> str:
        """트렌드 계산"""
        if len(results) < 4:  # 최소 4개 데이터포인트 필요
            return "stable"
        
        # 시간순 정렬
        sorted_results = sorted(results, key=lambda x: x.created_at)
        
        # 최근 절반 vs 이전 절반 비교
        mid_point = len(sorted_results) // 2
        older_scores = [r.final_score for r in sorted_results[:mid_point]]
        recent_scores = [r.final_score for r in sorted_results[mid_point:]]
        
        older_avg = statistics.mean(older_scores)
        recent_avg = statistics.mean(recent_scores)
        
        diff = recent_avg - older_avg
        
        if diff > 0.15:
            return "strongly_improving"
        elif diff > 0.05:
            return "improving"
        elif diff < -0.15:
            return "strongly_declining"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
    
    def get_fusion_stats_v2(self) -> Dict[str, Any]:
        """개선된 융합 통계"""
        cache_stats = self.cache_manager.get_stats()
        
        return {
            "fusion_manager": {
                **self.stats,
                "cache_size": len(self.cache_manager.cache),
                "symbol_cache_size": len(self.symbol_cache),
                "adaptive_weights_enabled": self.enable_adaptive_weights
            },
            "cache_performance": cache_stats,
            "configuration": {
                "confidence_threshold": self.confidence_threshold,
                "quality_threshold": self.quality_threshold,
                "outlier_z_threshold": self.outlier_z_threshold,
                "min_signals_for_outlier_removal": self.min_signals_for_outlier_removal,
                "base_weights": {k.value: v for k, v in self.base_weights.items()}
            }
        }
    
    def clear_cache(self):
        """캐시 정리"""
        with self.fusion_lock:
            cache_size = len(self.cache_manager.cache)
            symbol_cache_size = len(self.symbol_cache)
            
            self.cache_manager.cache.clear()
            self.cache_manager.access_count.clear()
            self.cache_manager.last_access.clear()
            self.symbol_cache.clear()
            
            logger.info(f"Cleared {cache_size} fusion cache entries and {symbol_cache_size} symbol cache entries")


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_fusion_manager_v2():
        """개선된 융합 매니저 테스트"""
        
        print("=== Sentiment Fusion Manager V2 테스트 ===")
        
        # 매니저 초기화
        manager = SentimentFusionManagerV2(
            cache_size=100,
            cache_ttl=300,
            enable_adaptive_weights=True
        )
        
        await manager.start()
        
        try:
            # 테스트 신호들
            test_signals = [
                SentimentSignalV2(
                    source=SignalSource.KEYWORD,
                    score=0.7,
                    confidence=0.8,
                    timestamp=datetime.now(),
                    processing_time=0.1,
                    reliability=0.7,
                    metadata={"symbol": "BTC", "matched_keywords": ["surge", "bullish"]}
                ),
                SentimentSignalV2(
                    source=SignalSource.FINBERT,
                    score=0.6,
                    confidence=0.9,
                    timestamp=datetime.now(),
                    processing_time=2.5,
                    reliability=0.9,
                    metadata={"confidence_raw": 0.87}
                ),
                SentimentSignalV2(
                    source=SignalSource.NEWS,
                    score=0.8,
                    confidence=0.75,
                    timestamp=datetime.now(),
                    processing_time=0.5,
                    reliability=0.8,
                    metadata={"source": "reuters"}
                )
            ]
            
            print(f"\n1. 감정 융합 테스트:")
            print(f"입력 신호: {len(test_signals)}개")
            for i, signal in enumerate(test_signals, 1):
                print(f"  {i}. {signal.source.value}: {signal.score:.3f} (신뢰도: {signal.confidence:.3f})")
            
            # 융합 실행
            result = await manager.fuse_sentiment_v2(
                content_hash="test_hash_1",
                signals=test_signals,
                method=FusionMethod.SMART_FALLBACK
            )
            
            print(f"\n융합 결과:")
            print(f"  최종 점수: {result.final_score:.3f}")
            print(f"  최종 신뢰도: {result.final_confidence:.3f}")
            print(f"  방향: {result.direction.value}")
            print(f"  강도: {result.intensity:.3f}")
            print(f"  합의 점수: {result.consensus_score:.3f}")
            print(f"  품질 점수: {result.quality_score:.3f}")
            print(f"  신뢰성 점수: {result.reliability_score:.3f}")
            print(f"  융합 방법: {result.fusion_method.value}")
            print(f"  처리 시간: {result.processing_metadata['processing_time_total']:.3f}초")
            
            # 캐시 테스트
            print(f"\n2. 캐시 테스트:")
            start_time = time.time()
            cached_result = await manager.fuse_sentiment_v2(
                content_hash="test_hash_1",
                signals=test_signals
            )
            cache_time = time.time() - start_time
            print(f"캐시된 결과 조회 시간: {cache_time:.3f}초")
            print(f"결과 일치: {result.final_score == cached_result.final_score}")
            
            # 심볼 감정 테스트
            print(f"\n3. 심볼 감정 집계 테스트:")
            symbol_result = await manager.get_symbol_sentiment("BTC", hours_back=1)
            print(f"BTC 감정:")
            for key, value in symbol_result.items():
                print(f"  {key}: {value}")
            
            # 통계
            print(f"\n4. 매니저 통계:")
            stats = manager.get_fusion_stats_v2()
            print(json.dumps(stats, indent=2, default=str))
            
            # 성능 테스트 (여러 번 실행)
            print(f"\n5. 성능 테스트:")
            iterations = 10
            start_time = time.time()
            
            for i in range(iterations):
                await manager.fuse_sentiment_v2(
                    content_hash=f"perf_test_{i}",
                    signals=test_signals[:2],  # 신호 수 줄여서 테스트
                    method=FusionMethod.CONFIDENCE_BASED
                )
            
            total_time = time.time() - start_time
            avg_time = total_time / iterations
            
            print(f"  {iterations}회 융합 실행")
            print(f"  총 시간: {total_time:.3f}초")
            print(f"  평균 시간: {avg_time:.3f}초")
            print(f"  초당 처리: {1/avg_time:.1f} 융합/초")
            
        finally:
            await manager.stop()
        
        print(f"\n=== 테스트 완료 ===")
    
    # 테스트 실행
    asyncio.run(test_fusion_manager_v2())