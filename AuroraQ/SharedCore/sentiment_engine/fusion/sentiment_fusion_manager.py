# SharedCore/sentiment_engine/fusion/sentiment_fusion_manager.py

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from functools import lru_cache
from collections import defaultdict
import json
from pathlib import Path
import asyncio

# Import analyzer
from ..analyzers.finbert_analyzer import FinBERTAnalyzer, SentimentResult
from ...data_collection.base_collector import NewsArticle

logger = logging.getLogger(__name__)

@dataclass
class FusionConfig:
    """융합 설정"""
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "news": 0.4,
        "social": 0.3, 
        "technical": 0.2,
        "historical": 0.1
    })
    correction_factor: float = 0.1
    neutral_threshold: float = 0.05
    confidence_threshold: float = 0.6
    volatility_window: int = 20
    adaptive_weighting: bool = True
    outlier_detection: bool = True
    outlier_std_threshold: float = 3.0

@dataclass 
class FusedSentiment:
    """융합된 감정 분석 결과"""
    timestamp: datetime
    symbol: str
    fused_score: float
    raw_scores: Dict[str, float]
    weights_used: Dict[str, float]
    confidence: float
    volatility: float
    trend: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class SentimentFusionManager:
    """개선된 감정 점수 융합 관리자"""
    
    def __init__(self,
                 config: Optional[FusionConfig] = None,
                 analyzer: Optional[FinBERTAnalyzer] = None,
                 history_size: int = 1000):
        """
        Args:
            config: 융합 설정
            analyzer: 감정 분석기 인스턴스
            history_size: 히스토리 보관 크기
        """
        self.config = config or FusionConfig()
        self.analyzer = analyzer
        
        # 히스토리 관리
        self.history: List[FusedSentiment] = []
        self.history_size = history_size
        
        # 성능 추적
        self.performance_metrics = defaultdict(lambda: {
            'accuracy': [],
            'confidence': [], 
            'volatility': []
        })
        
        # 적응형 가중치 학습
        self.adaptive_weights = self.config.source_weights.copy()
        self.weight_update_count = 0
        
        # 캐시
        self._cache = {}
        self._cache_ttl = 300  # 5분
        
        logger.info("SentimentFusionManager initialized")
    
    async def initialize(self):
        """비동기 초기화"""
        if self.analyzer is None:
            from ..analyzers.finbert_analyzer import get_finbert_analyzer
            self.analyzer = await get_finbert_analyzer()
        elif not self.analyzer._initialized:
            await self.analyzer.initialize()
    
    async def fuse(self, 
                   sentiment_scores: Dict[str, float],
                   symbol: str = "BTCUSDT",
                   timestamp: Optional[datetime] = None) -> float:
        """
        여러 소스의 감정 점수를 융합
        
        Args:
            sentiment_scores: 소스별 감정 점수
            symbol: 심볼
            timestamp: 타임스탬프
            
        Returns:
            융합된 감정 점수 (0.0 ~ 1.0)
        """
        if not sentiment_scores:
            logger.warning("No sentiment scores provided")
            return 0.5
        
        timestamp = timestamp or datetime.utcnow()
        
        # 캐시 확인
        cache_key = self._get_cache_key(sentiment_scores, symbol, timestamp)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # 점수 검증 및 정규화
            validated_scores = self._validate_scores(sentiment_scores)
            
            # 이상치 제거
            if self.config.outlier_detection:
                filtered_scores = self._remove_outliers(validated_scores)
            else:
                filtered_scores = validated_scores
            
            # 가중치 결정
            if self.config.adaptive_weighting:
                weights = self._get_adaptive_weights(filtered_scores)
            else:
                weights = self._get_static_weights(filtered_scores)
            
            # 가중 평균 계산
            fused_score = self._calculate_weighted_average(filtered_scores, weights)
            
            # 보정 적용
            corrected_score = self._apply_corrections(fused_score, filtered_scores)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(filtered_scores, weights)
            
            # 변동성 계산
            volatility = self._calculate_volatility(symbol)
            
            # 트렌드 판단
            trend = self._determine_trend(corrected_score)
            
            # 결과 저장
            result = FusedSentiment(
                timestamp=timestamp,
                symbol=symbol,
                fused_score=corrected_score,
                raw_scores=sentiment_scores.copy(),
                weights_used=weights.copy(),
                confidence=confidence,
                volatility=volatility,
                trend=trend,
                metadata={
                    'filtered_count': len(sentiment_scores) - len(filtered_scores),
                    'correction_applied': abs(corrected_score - fused_score)
                }
            )
            
            self._update_history(result)
            self._cache_result(cache_key, corrected_score)
            
            return corrected_score
            
        except Exception as e:
            logger.error(f"Fusion failed: {e}", exc_info=True)
            return 0.5
    
    def _validate_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """점수 유효성 검증 및 정규화"""
        validated = {}
        
        for source, score in scores.items():
            if not isinstance(score, (int, float)):
                logger.warning(f"Invalid score type for {source}: {type(score)}")
                continue
            
            # NaN 체크
            if np.isnan(score):
                logger.warning(f"NaN score for {source}")
                continue
            
            # 범위 정규화 (다양한 범위 지원)
            if -1 <= score <= 1:
                # -1 ~ 1 범위를 0 ~ 1로 변환
                normalized = (score + 1) / 2
            elif 0 <= score <= 1:
                # 이미 0 ~ 1 범위
                normalized = score
            elif 0 <= score <= 100:
                # 0 ~ 100 범위를 0 ~ 1로 변환
                normalized = score / 100
            else:
                logger.warning(f"Score out of expected range for {source}: {score}")
                normalized = 0.5
            
            validated[source] = float(normalized)
        
        return validated
    
    def _remove_outliers(self, scores: Dict[str, float]) -> Dict[str, float]:
        """이상치 제거"""
        if len(scores) < 3:
            return scores
        
        values = list(scores.values())
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return scores
        
        filtered = {}
        for source, score in scores.items():
            z_score = abs((score - mean) / std)
            if z_score <= self.config.outlier_std_threshold:
                filtered[source] = score
            else:
                logger.debug(f"Outlier removed: {source}={score} (z-score: {z_score:.2f})")
        
        # 최소 1개 소스는 유지
        if not filtered:
            # 평균에 가장 가까운 값 선택
            closest_source = min(scores.keys(), key=lambda k: abs(scores[k] - mean))
            filtered[closest_source] = scores[closest_source]
        
        return filtered
    
    def _get_static_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """정적 가중치 반환"""
        weights = {}
        total_weight = 0.0
        
        for source in scores:
            weight = self.config.source_weights.get(source, 0.1)
            weights[source] = weight
            total_weight += weight
        
        # 정규화
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _get_adaptive_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """적응형 가중치 계산"""
        # 기본 가중치로 시작
        weights = self._get_static_weights(scores)
        
        # 성능 기반 조정
        for source in scores:
            if source in self.performance_metrics:
                metrics = self.performance_metrics[source]
                
                # 최근 정확도 기반 조정
                if metrics['accuracy']: 
                    recent_accuracy = np.mean(metrics['accuracy'][-10:])
                    adjustment = (recent_accuracy - 0.5) * 0.2  # ±0.1 범위 조정
                    weights[source] = max(0.05, min(0.95, weights[source] + adjustment))
        
        # 재정규화
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        self.adaptive_weights = weights.copy()
        return weights
    
    def _calculate_weighted_average(self, 
                                   scores: Dict[str, float], 
                                   weights: Dict[str, float]) -> float:
        """가중 평균 계산"""
        weighted_sum = sum(scores[k] * weights[k] for k in scores)
        return float(np.clip(weighted_sum, 0.0, 1.0))
    
    def _apply_corrections(self, 
                          base_score: float, 
                          raw_scores: Dict[str, float]) -> float:
        """점수 보정 적용"""
        corrected = base_score
        
        # 중립 근처 보정
        if abs(base_score - 0.5) < self.config.neutral_threshold:
            # 중립으로 수렴
            corrected = 0.5
        else:
            # 극단값 완화
            if base_score > 0.5:
                corrected = base_score - self.config.correction_factor * (base_score - 0.5)
            else:
                corrected = base_score + self.config.correction_factor * (0.5 - base_score)
        
        # 소스 간 불일치도가 높으면 추가 보정
        if len(raw_scores) > 1:
            disagreement = np.std(list(raw_scores.values()))
            if disagreement > 0.3:
                # 중립 방향으로 조정
                corrected = corrected * 0.8 + 0.5 * 0.2
        
        return float(np.clip(corrected, 0.0, 1.0))
    
    def _calculate_confidence(self, 
                            scores: Dict[str, float], 
                            weights: Dict[str, float]) -> float:
        """융합 신뢰도 계산"""
        if not scores:
            return 0.0
        
        # 요소 1: 소스 수
        source_confidence = min(len(scores) / 4.0, 1.0)  # 4개 소스면 최대
        
        # 요소 2: 일치도
        if len(scores) > 1:
            agreement = 1.0 - np.std(list(scores.values()))
        else:
            agreement = 0.5
        
        # 요소 3: 가중치 분산도
        weight_entropy = -sum(w * np.log(w + 1e-10) for w in weights.values() if w > 0)
        weight_confidence = 1.0 - (weight_entropy / np.log(len(weights)) if len(weights) > 1 else 0)
        
        # 종합 신뢰도
        confidence = (source_confidence * 0.3 + agreement * 0.5 + weight_confidence * 0.2)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_volatility(self, symbol: str, window: Optional[int] = None) -> float:
        """변동성 계산"""
        window = window or self.config.volatility_window
        
        # 심볼별 최근 점수 추출
        recent_scores = [
            h.fused_score for h in self.history[-window:]
            if h.symbol == symbol
        ]
        
        if len(recent_scores) < 2:
            return 0.0
        
        return float(np.std(recent_scores))
    
    def _determine_trend(self, score: float) -> str:
        """트렌드 판단"""
        if score >= 0.7:
            return "strong_bullish"
        elif score >= 0.6:
            return "bullish"
        elif score >= 0.4:
            return "neutral"
        elif score >= 0.3:
            return "bearish"
        else:
            return "strong_bearish"
    
    def _update_history(self, result: FusedSentiment):
        """히스토리 업데이트"""
        self.history.append(result)
        
        # 크기 제한
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # 적응형 가중치 업데이트 주기
        self.weight_update_count += 1
        if self.weight_update_count % 100 == 0:
            self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """성능 메트릭 업데이트"""
        # 실제 구현에서는 실제 시장 움직임과 비교하여 정확도 계산
        # 여기서는 시뮬레이션
        pass
    
    async def get_fused_scores(self, 
                              articles: List[Union[Dict[str, Any], NewsArticle]],
                              symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
        """
        기사 리스트에 대한 융합 감정 분석
        
        Args:
            articles: 기사 리스트
            symbol: 심볼
            
        Returns:
            융합된 감정 분석 결과 리스트
        """
        # 초기화 확인
        await self.initialize()
        
        results = []
        
        for article in articles:
            try:
                # 개별 분석
                detail = await self.analyzer.analyze_detailed(article)
                
                # 단일 소스로 융합 (확장 가능)
                fused_score = await self.fuse(
                    {"news": detail.sentiment_score},
                    symbol=symbol,
                    timestamp=getattr(article, 'datetime', None) or 
                             article.get("datetime", datetime.utcnow()) if isinstance(article, dict) else datetime.utcnow()
                )
                
                # 결과 구성
                result = {
                    "datetime": getattr(article, 'datetime', None) or 
                               article.get("datetime", datetime.utcnow()) if isinstance(article, dict) else datetime.utcnow(),
                    "symbol": symbol,
                    "sentiment_score": fused_score,
                    "label": detail.label.value,
                    "confidence": detail.confidence,
                    "keywords": detail.keywords,
                    "scenario_tag": detail.scenario_tag,
                    "volatility": self._calculate_volatility(symbol),
                    "trend": self._determine_trend(fused_score)
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process article: {e}")
                continue
        
        return results
    
    def update_weights(self, new_weights: Dict[str, float]):
        """가중치 업데이트"""
        # 검증
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.config.source_weights = new_weights.copy()
        logger.info(f"Weights updated: {new_weights}")
    
    def get_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """통계 정보 반환"""
        if symbol:
            history = [h for h in self.history if h.symbol == symbol]
        else:
            history = self.history
        
        if not history:
            return {}
        
        scores = [h.fused_score for h in history]
        
        return {
            "count": len(history),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "current_score": scores[-1] if scores else None,
            "trend_distribution": self._get_trend_distribution(history),
            "source_weights": self.adaptive_weights if self.config.adaptive_weighting else self.config.source_weights,
            "average_confidence": np.mean([h.confidence for h in history])
        }
    
    def _get_trend_distribution(self, history: List[FusedSentiment]) -> Dict[str, float]:
        """트렌드 분포 계산"""
        trends = [h.trend for h in history]
        total = len(trends)
        
        distribution = {}
        for trend in ["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"]:
            count = trends.count(trend)
            distribution[trend] = count / total if total > 0 else 0.0
        
        return distribution
    
    # 캐시 관련 메서드
    def _get_cache_key(self, scores: Dict[str, float], symbol: str, timestamp: datetime) -> str:
        """캐시 키 생성"""
        # 점수를 정렬하여 일관된 키 생성
        sorted_scores = sorted(scores.items())
        scores_str = json.dumps(sorted_scores)
        
        # 분 단위로 반올림 (같은 분 내에서는 같은 캐시 사용)
        rounded_time = timestamp.replace(second=0, microsecond=0)
        
        return f"{symbol}_{rounded_time.isoformat()}_{hash(scores_str)}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[float]:
        """캐시에서 결과 조회"""
        if cache_key in self._cache:
            cached_time, cached_score = self._cache[cache_key]
            if (datetime.utcnow() - cached_time).seconds < self._cache_ttl:
                return cached_score
            else:
                del self._cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, score: float):
        """결과 캐싱"""
        self._cache[cache_key] = (datetime.utcnow(), score)
        
        # 캐시 크기 제한
        if len(self._cache) > 1000:
            # 가장 오래된 항목 제거
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
    
    async def close(self):
        """리소스 정리"""
        if self.analyzer:
            await self.analyzer.close()
        
        logger.info("SentimentFusionManager resources cleaned up")


# 전역 인스턴스 관리
_global_fusion_manager: Optional[SentimentFusionManager] = None

async def get_fusion_manager() -> SentimentFusionManager:
    """전역 융합 관리자 인스턴스 반환"""
    global _global_fusion_manager
    
    if _global_fusion_manager is None:
        _global_fusion_manager = SentimentFusionManager()
        await _global_fusion_manager.initialize()
    
    return _global_fusion_manager


if __name__ == "__main__":
    import asyncio
    import random
    
    async def test_fusion_manager():
        """테스트 실행"""
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 융합 관리자 생성
        fusion_manager = SentimentFusionManager()
        await fusion_manager.initialize()
        
        print("=== Sentiment Fusion Test ===\n")
        
        # 테스트 1: 단일 소스 융합
        print("1. Single Source Fusion:")
        score1 = await fusion_manager.fuse({"news": 0.7})
        print(f"   News (0.7) → Fused: {score1:.4f}")
        
        # 테스트 2: 다중 소스 융합
        print("\n2. Multi-Source Fusion:")
        multi_scores = {
            "news": 0.8,
            "social": 0.6,
            "technical": 0.5,
            "historical": 0.7
        }
        score2 = await fusion_manager.fuse(multi_scores)
        print(f"   {multi_scores}")
        print(f"   → Fused: {score2:.4f}")
        
        # 테스트 3: 이상치 처리
        print("\n3. Outlier Handling:")
        outlier_scores = {
            "news": 0.7,
            "social": 0.6,
            "technical": 0.1,  # 이상치
            "historical": 0.65
        }
        score3 = await fusion_manager.fuse(outlier_scores)
        print(f"   {outlier_scores}")
        print(f"   → Fused: {score3:.4f} (outlier removed)")
        
        # 테스트 4: 시계열 시뮬레이션
        print("\n4. Time Series Simulation:")
        symbol = "BTCUSDT"
        
        for i in range(10):
            # 랜덤 점수 생성
            scores = {
                "news": 0.5 + 0.3 * np.sin(i/3) + random.uniform(-0.1, 0.1),
                "social": 0.5 + 0.2 * np.cos(i/2) + random.uniform(-0.1, 0.1),
                "technical": 0.5 + random.uniform(-0.2, 0.2)
            }
            
            # 점수 클리핑
            scores = {k: max(0, min(1, v)) for k, v in scores.items()}
            
            fused = await fusion_manager.fuse(scores, symbol=symbol)
            
            if i % 3 == 0:
                print(f"   T{i}: {scores} → {fused:.4f}")
        
        # 테스트 5: 통계
        print("\n5. Statistics:")
        stats = fusion_manager.get_statistics(symbol)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")
            else:
                print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
        
        await fusion_manager.close()
        print("\n=== Test Complete ===")
    
    # 테스트 실행
    asyncio.run(test_fusion_manager())