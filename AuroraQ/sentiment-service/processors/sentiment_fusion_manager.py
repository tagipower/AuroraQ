#!/usr/bin/env python3
"""
Sentiment Fusion Manager for AuroraQ Sentiment Service
실시간 키워드 스코어러와 FinBERT 배치 결과를 융합하여 최종 감정 점수 생성
"""

import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import numpy as np

# 로컬 임포트 (상대 import 문제 해결을 위해 조건부 import)
try:
    from ..models.keyword_scorer import KeywordScorer, KeywordScore, SentimentDirection
    from .finbert_batch_processor import FinBERTBatchProcessor, BatchResult
    from ..utils.content_cache_manager import ContentCacheManager, ContentMetadata
except ImportError:
    # 절대 import 대안
    try:
        from models.keyword_scorer import KeywordScorer, KeywordScore, SentimentDirection
        from processors.finbert_batch_processor import FinBERTBatchProcessor, BatchResult
        from utils.content_cache_manager import ContentCacheManager, ContentMetadata
    except ImportError:
        # 최소 기능을 위한 대안
        from typing import Any
        KeywordScorer = Any
        KeywordScore = Any
        SentimentDirection = Any
        FinBERTBatchProcessor = Any
        BatchResult = Any
        ContentCacheManager = Any
        ContentMetadata = Any

logger = logging.getLogger(__name__)

class FusionMethod(Enum):
    """융합 방법"""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    ADAPTIVE = "adaptive"
    ENSEMBLE = "ensemble"

@dataclass
class SentimentSignal:
    """감정 신호"""
    source: str  # "keyword", "finbert", "fusion"
    score: float  # -1.0 ~ 1.0
    confidence: float  # 0.0 ~ 1.0
    timestamp: datetime
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusedSentiment:
    """융합된 감정 결과"""
    content_hash: str
    final_score: float  # -1.0 ~ 1.0
    final_confidence: float  # 0.0 ~ 1.0
    direction: str  # "bullish", "bearish", "neutral"
    signals: List[SentimentSignal]
    fusion_method: FusionMethod
    quality_score: float
    reliability_score: float
    processing_metadata: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "content_hash": self.content_hash,
            "final_score": round(self.final_score, 4),
            "final_confidence": round(self.final_confidence, 4),
            "direction": self.direction,
            "signals": [
                {
                    "source": s.source,
                    "score": round(s.score, 4),
                    "confidence": round(s.confidence, 4),
                    "timestamp": s.timestamp.isoformat(),
                    "processing_time": round(s.processing_time, 3),
                    "metadata": s.metadata
                }
                for s in self.signals
            ],
            "fusion_method": self.fusion_method.value,
            "quality_score": round(self.quality_score, 4),
            "reliability_score": round(self.reliability_score, 4),
            "processing_metadata": self.processing_metadata,
            "created_at": self.created_at.isoformat()
        }

class SentimentFusionManager:
    """감정 융합 관리자"""
    
    def __init__(self,
                 keyword_scorer: KeywordScorer,
                 finbert_processor: FinBERTBatchProcessor,
                 cache_manager: Optional[ContentCacheManager] = None):
        """
        초기화
        
        Args:
            keyword_scorer: 실시간 키워드 스코어러
            finbert_processor: FinBERT 배치 프로세서
            cache_manager: 캐시 매니저
        """
        self.keyword_scorer = keyword_scorer
        self.finbert_processor = finbert_processor
        self.cache_manager = cache_manager
        
        # 융합 가중치 설정
        self.fusion_weights = {
            "news": 0.4,        # 뉴스 감정
            "social": 0.3,      # 소셜 감정
            "technical": 0.2,   # 기술적 지표
            "historical": 0.1   # 히스토리 트렌드
        }
        
        # 신뢰도 임계값
        self.confidence_threshold = 0.6
        self.quality_threshold = 0.5
        
        # 이상치 제거 설정
        self.outlier_z_threshold = 3.0
        
        # 융합 결과 캐시
        self.fusion_cache: Dict[str, FusedSentiment] = {}
        self.cache_ttl = 300  # 5분
        
        # 성능 통계
        self.stats = {
            "total_fused": 0,
            "keyword_only": 0,
            "finbert_only": 0,
            "both_sources": 0,
            "avg_fusion_time": 0.0,
            "fusion_accuracy": 0.0,
            "outliers_removed": 0
        }
    
    def _remove_outliers(self, scores: List[float]) -> List[float]:
        """이상치 제거 (Z-score 기반)"""
        
        if len(scores) < 3:
            return scores
        
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        
        if std_score == 0:
            return scores
        
        filtered_scores = []
        removed_count = 0
        
        for score in scores:
            z_score = abs((score - mean_score) / std_score)
            if z_score <= self.outlier_z_threshold:
                filtered_scores.append(score)
            else:
                removed_count += 1
        
        if removed_count > 0:
            self.stats["outliers_removed"] += removed_count
            logger.debug(f"Removed {removed_count} outliers from {len(scores)} scores")
        
        return filtered_scores if filtered_scores else scores
    
    def _calculate_confidence_weight(self, confidence: float) -> float:
        """신뢰도 기반 가중치 계산"""
        
        if confidence < self.confidence_threshold:
            return 0.1  # 낮은 신뢰도는 가중치 감소
        
        # 시그모이드 함수로 가중치 조정
        return 1 / (1 + np.exp(-5 * (confidence - 0.5)))
    
    def _determine_direction(self, score: float) -> str:
        """점수를 방향으로 변환"""
        
        if score > 0.1:
            return "bullish"
        elif score < -0.1:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_quality_score(self, signals: List[SentimentSignal]) -> float:
        """품질 점수 계산"""
        
        if not signals:
            return 0.0
        
        # 신호 개수 점수 (더 많은 신호 = 더 높은 품질)
        signal_count_score = min(1.0, len(signals) / 3.0)
        
        # 평균 신뢰도
        avg_confidence = statistics.mean(s.confidence for s in signals)
        
        # 신호 간 일관성 (표준편차 역수)
        scores = [s.score for s in signals]
        if len(scores) > 1:
            consistency = 1.0 / (1.0 + statistics.stdev(scores))
        else:
            consistency = 1.0
        
        # 소스 다양성 (서로 다른 소스 개수)
        unique_sources = len(set(s.source for s in signals))
        diversity_score = min(1.0, unique_sources / 2.0)
        
        # 가중 평균
        quality_score = (
            signal_count_score * 0.3 +
            avg_confidence * 0.4 +
            consistency * 0.2 +
            diversity_score * 0.1
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_reliability_score(self, 
                                   signals: List[SentimentSignal],
                                   content_metadata: Optional[ContentMetadata] = None) -> float:
        """신뢰성 점수 계산"""
        
        if not signals:
            return 0.0
        
        reliability_factors = []
        
        # 1. 신호 품질 평균
        avg_confidence = statistics.mean(s.confidence for s in signals)
        reliability_factors.append(avg_confidence)
        
        # 2. 처리 시간 기반 신뢰성 (빠를수록 좋음)
        avg_processing_time = statistics.mean(s.processing_time for s in signals)
        time_reliability = max(0.0, 1.0 - (avg_processing_time / 5.0))  # 5초 기준
        reliability_factors.append(time_reliability)
        
        # 3. 소스 신뢰성
        source_reliability = 0.0
        for signal in signals:
            if signal.source == "finbert":
                source_reliability += 0.9
            elif signal.source == "keyword":
                source_reliability += 0.7
            elif signal.source == "fusion":
                source_reliability += 0.8
        
        source_reliability = min(1.0, source_reliability / len(signals))
        reliability_factors.append(source_reliability)
        
        # 4. 메타데이터 기반 신뢰성
        if content_metadata:
            metadata_reliability = content_metadata.relevance_score
            reliability_factors.append(metadata_reliability)
        
        return statistics.mean(reliability_factors)
    
    async def fuse_sentiment(self,
                           content_hash: str,
                           text: str,
                           method: FusionMethod = FusionMethod.ADAPTIVE,
                           force_refresh: bool = False) -> FusedSentiment:
        """감정 융합"""
        
        start_time = time.time()
        
        # 캐시 확인
        if not force_refresh and content_hash in self.fusion_cache:
            cached_result = self.fusion_cache[content_hash]
            cache_age = (datetime.now() - cached_result.created_at).total_seconds()
            
            if cache_age < self.cache_ttl:
                logger.debug(f"Returning cached fusion result: {content_hash}")
                return cached_result
        
        logger.debug(f"Fusing sentiment for: {content_hash}")
        
        signals = []
        
        # 1. 실시간 키워드 분석
        try:
            keyword_result = self.keyword_scorer.analyze(text)
            
            keyword_signal = SentimentSignal(
                source="keyword",
                score=keyword_result.score,
                confidence=keyword_result.confidence,
                timestamp=datetime.now(),
                processing_time=keyword_result.processing_time / 1000.0,  # ms to s
                metadata={
                    "direction": keyword_result.direction.value,
                    "matched_keywords": keyword_result.matched_keywords,
                    "category_scores": keyword_result.category_scores
                }
            )
            signals.append(keyword_signal)
            
        except Exception as e:
            logger.error(f"Keyword analysis failed: {e}")
        
        # 2. FinBERT 배치 결과 확인
        finbert_result = None
        if self.cache_manager:
            try:
                metadata = await self.cache_manager.get_metadata(content_hash)
                if metadata and metadata.sentiment_score is not None:
                    finbert_signal = SentimentSignal(
                        source="finbert",
                        score=metadata.sentiment_score * 2 - 1,  # 0~1 to -1~1 변환
                        confidence=metadata.confidence or 0.7,
                        timestamp=metadata.processing_timestamp,
                        processing_time=0.5,  # 배치 처리 시간 추정
                        metadata={
                            "keywords": metadata.keywords,
                            "entities": metadata.entities
                        }
                    )
                    signals.append(finbert_signal)
                    finbert_result = metadata
                    
            except Exception as e:
                logger.error(f"FinBERT result retrieval failed: {e}")
        
        # 3. 융합 방법 선택
        if method == FusionMethod.ADAPTIVE:
            # 신호 개수와 품질에 따라 방법 선택
            if len(signals) >= 2:
                method = FusionMethod.CONFIDENCE_BASED
            else:
                method = FusionMethod.WEIGHTED_AVERAGE
        
        # 4. 융합 실행
        final_score, final_confidence = self._execute_fusion(signals, method)
        
        # 5. 품질 및 신뢰성 점수 계산
        quality_score = self._calculate_quality_score(signals)
        reliability_score = self._calculate_reliability_score(signals, finbert_result)
        
        # 6. 방향 결정
        direction = self._determine_direction(final_score)
        
        # 7. 처리 메타데이터
        processing_metadata = {
            "signals_count": len(signals),
            "fusion_method_used": method.value,
            "processing_time_total": time.time() - start_time,
            "has_finbert": any(s.source == "finbert" for s in signals),
            "has_keyword": any(s.source == "keyword" for s in signals),
            "outliers_removed": False  # 이상치 제거 여부는 융합 과정에서 설정
        }
        
        # 8. 결과 생성
        fused_result = FusedSentiment(
            content_hash=content_hash,
            final_score=final_score,
            final_confidence=final_confidence,
            direction=direction,
            signals=signals,
            fusion_method=method,
            quality_score=quality_score,
            reliability_score=reliability_score,
            processing_metadata=processing_metadata,
            created_at=datetime.now()
        )
        
        # 9. 캐시 저장
        self.fusion_cache[content_hash] = fused_result
        
        # 10. 통계 업데이트
        self._update_stats(signals, time.time() - start_time)
        
        logger.debug(f"Sentiment fusion completed: {direction} "
                    f"({final_score:.3f}, {final_confidence:.3f}) "
                    f"from {len(signals)} signals")
        
        return fused_result
    
    def _execute_fusion(self, 
                       signals: List[SentimentSignal], 
                       method: FusionMethod) -> Tuple[float, float]:
        """융합 실행"""
        
        if not signals:
            return 0.0, 0.0
        
        if len(signals) == 1:
            return signals[0].score, signals[0].confidence
        
        scores = [s.score for s in signals]
        confidences = [s.confidence for s in signals]
        
        if method == FusionMethod.WEIGHTED_AVERAGE:
            # 단순 가중 평균
            weights = [1.0] * len(signals)
            
        elif method == FusionMethod.CONFIDENCE_BASED:
            # 신뢰도 기반 가중치
            weights = [self._calculate_confidence_weight(c) for c in confidences]
            
        elif method == FusionMethod.ENSEMBLE:
            # 앙상블 방법 (복잡한 가중치)
            weights = []
            for signal in signals:
                if signal.source == "finbert":
                    weights.append(0.6)  # FinBERT 높은 가중치
                elif signal.source == "keyword":
                    weights.append(0.4)  # 키워드 낮은 가중치
                else:
                    weights.append(0.5)
        
        else:
            # 기본값: 균등 가중치
            weights = [1.0] * len(signals)
        
        # 이상치 제거
        filtered_scores = self._remove_outliers(scores)
        if len(filtered_scores) != len(scores):
            # 이상치가 제거된 경우 인덱스 조정
            filtered_indices = []
            for i, score in enumerate(scores):
                if score in filtered_scores:
                    filtered_indices.append(i)
            
            scores = [scores[i] for i in filtered_indices]
            confidences = [confidences[i] for i in filtered_indices]
            weights = [weights[i] for i in filtered_indices]
        
        # 가중 평균 계산
        if not scores:
            return 0.0, 0.0
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0, 0.0
        
        # 정규화된 가중치
        normalized_weights = [w / total_weight for w in weights]
        
        # 최종 점수 계산
        final_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        final_confidence = sum(conf * weight for conf, weight in zip(confidences, normalized_weights))
        
        # 범위 제한
        final_score = max(-1.0, min(1.0, final_score))
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return final_score, final_confidence
    
    def _update_stats(self, signals: List[SentimentSignal], processing_time: float):
        """통계 업데이트"""
        
        self.stats["total_fused"] += 1
        
        # 신호 소스별 카운트
        has_keyword = any(s.source == "keyword" for s in signals)
        has_finbert = any(s.source == "finbert" for s in signals)
        
        if has_keyword and has_finbert:
            self.stats["both_sources"] += 1
        elif has_keyword:
            self.stats["keyword_only"] += 1
        elif has_finbert:
            self.stats["finbert_only"] += 1
        
        # 평균 처리 시간 업데이트
        current_avg = self.stats["avg_fusion_time"]
        total_count = self.stats["total_fused"]
        
        self.stats["avg_fusion_time"] = (
            (current_avg * (total_count - 1) + processing_time) / total_count
        )
    
    async def get_realtime_sentiment(self,
                                   symbol: str,
                                   hours_back: int = 24) -> Dict[str, Any]:
        """실시간 심볼 감정 조회"""
        
        # 여기서는 캐시된 융합 결과들을 기반으로 심볼별 집계를 수행
        # 실제 구현에서는 데이터베이스나 시계열 저장소를 사용할 수 있음
        
        symbol_results = []
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=hours_back)
        
        # 캐시에서 해당 심볼 관련 결과 찾기
        for content_hash, result in self.fusion_cache.items():
            if result.created_at >= cutoff_time:
                # 메타데이터에서 심볼 확인 (실제로는 더 정교한 매칭 필요)
                symbol_results.append(result)
        
        if not symbol_results:
            return {
                "symbol": symbol.upper(),
                "score": 0.0,
                "confidence": 0.0,
                "trend": "neutral",
                "news_count": 0,
                "last_updated": None,
                "sources": []
            }
        
        # 집계 계산
        scores = [r.final_score for r in symbol_results]
        confidences = [r.final_confidence for r in symbol_results]
        
        avg_score = statistics.mean(scores)
        avg_confidence = statistics.mean(confidences)
        
        # 트렌드 계산 (최근 vs 이전 비교)
        if len(scores) >= 2:
            recent_scores = scores[-len(scores)//2:]  # 최근 절반
            older_scores = scores[:len(scores)//2]   # 이전 절반
            
            recent_avg = statistics.mean(recent_scores)
            older_avg = statistics.mean(older_scores)
            
            if recent_avg > older_avg + 0.1:
                trend = "improving"
            elif recent_avg < older_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "neutral"
        
        # 소스 정보
        sources = set()
        for result in symbol_results:
            for signal in result.signals:
                sources.add(signal.source)
        
        return {
            "symbol": symbol.upper(),
            "score": round(avg_score, 3),
            "confidence": round(avg_confidence, 3),
            "trend": trend,
            "news_count": len(symbol_results),
            "last_updated": max(r.created_at for r in symbol_results).isoformat(),
            "sources": list(sources)
        }
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """융합 통계 반환"""
        
        return {
            "fusion_cache_size": len(self.fusion_cache),
            "cache_ttl_seconds": self.cache_ttl,
            "confidence_threshold": self.confidence_threshold,
            "outlier_z_threshold": self.outlier_z_threshold,
            "fusion_weights": self.fusion_weights,
            **self.stats
        }
    
    def clear_cache(self):
        """캐시 정리"""
        cleared_count = len(self.fusion_cache)
        self.fusion_cache.clear()
        logger.info(f"Cleared {cleared_count} fusion cache entries")


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_fusion_manager():
        """융합 매니저 테스트"""
        
        # Mock 객체들 (실제 구현에서는 실제 객체 사용)
        keyword_scorer = KeywordScorer()
        
        # FinBERT 프로세서는 실제로 초기화하지 않고 Mock
        finbert_processor = None
        
        fusion_manager = SentimentFusionManager(
            keyword_scorer=keyword_scorer,
            finbert_processor=finbert_processor
        )
        
        print("=== 감정 융합 매니저 테스트 ===")
        
        # 테스트 텍스트들
        test_texts = [
            {
                "hash": "test1",
                "text": "Bitcoin surges to new all-time high as institutional adoption grows"
            },
            {
                "hash": "test2", 
                "text": "Market crash fears grip investors amid economic uncertainty"
            },
            {
                "hash": "test3",
                "text": "Neutral market conditions with mixed trading signals"
            }
        ]
        
        print(f"\n1. 감정 융합 테스트:")
        for item in test_texts:
            result = await fusion_manager.fuse_sentiment(
                content_hash=item["hash"],
                text=item["text"],
                method=FusionMethod.ADAPTIVE
            )
            
            print(f"\n텍스트: {item['text'][:50]}...")
            print(f"  최종 점수: {result.final_score:.3f}")
            print(f"  신뢰도: {result.final_confidence:.3f}")
            print(f"  방향: {result.direction}")
            print(f"  신호 수: {len(result.signals)}")
            print(f"  품질 점수: {result.quality_score:.3f}")
            print(f"  융합 방법: {result.fusion_method.value}")
        
        # 실시간 감정 조회 테스트
        print(f"\n2. 실시간 감정 조회 테스트:")
        realtime_result = await fusion_manager.get_realtime_sentiment("BTC")
        print(f"BTC 실시간 감정: {realtime_result}")
        
        # 통계
        print(f"\n3. 융합 통계:")
        stats = fusion_manager.get_fusion_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # 테스트 실행
    asyncio.run(test_fusion_manager())