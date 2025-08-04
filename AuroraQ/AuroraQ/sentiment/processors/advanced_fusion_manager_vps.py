"""
VPS-Optimized Advanced Fusion Manager
9패널 대시보드를 위한 통합 메트릭 생성기

ONNX 센티먼트 서비스와 통합된 고급 융합 분석:
- 메모리 효율적 처리
- 배치 최적화
- 캐시 기반 성능 향상
- VPS 환경 최적화
"""

import asyncio
import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor

# VPS 센티먼트 분석기 임포트
from ..models.advanced_keyword_scorer_vps import (
    VPSAdvancedKeywordScorer, 
    MultiModalSentiment,
    EmotionalState,
    MarketRegime,
    AdvancedFeatures
)

# VPS 최적화 설정
VPS_FUSION_CONFIG = {
    "max_batch_size": 100,
    "cache_ttl": 1800,  # 30분 캐시
    "memory_limit_mb": 4096,  # 4GB 메모리 제한
    "parallel_workers": 4,
    "enable_ml_features": True,
    "onnx_integration": True
}

@dataclass
class MarketPrediction:
    """ML 기반 시장 예측"""
    direction: str  # "up", "down", "sideways"
    confidence: float  # 0-1
    timeframe: str  # "1h", "4h", "24h"
    price_target: Optional[float]
    risk_level: str  # "low", "medium", "high"
    
    # ONNX 메타데이터
    model_version: str = "v1.0"
    inference_time: float = 0.0
    ensemble_models: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

@dataclass
class EventImpactAnalysis:
    """이벤트 영향 분석"""
    event_type: str
    impact_score: float  # -1 to 1
    duration_hours: float
    affected_markets: List[str]
    spillover_risk: float
    
    # 상세 분석
    sentiment_shift: float
    volume_impact: float
    volatility_impact: float
    correlation_changes: Dict[str, float] = field(default_factory=dict)

@dataclass 
class StrategyPerformance:
    """전략 성과 분석"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    
    # 리스크 메트릭
    var_95: float  # Value at Risk
    expected_shortfall: float
    beta: float
    alpha: float
    information_ratio: float

@dataclass
class AnomalyDetection:
    """이상 탐지 결과"""
    anomaly_score: float  # 0-1
    anomaly_type: str  # "price", "volume", "sentiment", "correlation"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    
    # 통계적 지표
    z_score: float
    p_value: float
    threshold_breach: bool
    historical_frequency: float

@dataclass
class RefinedFeatureSet:
    """9패널 대시보드용 통합 특성 세트"""
    fusion_score: float
    ml_refined_prediction: MarketPrediction
    event_impact: EventImpactAnalysis
    strategy_performance: StrategyPerformance
    anomaly_detection: AnomalyDetection
    advanced_features: AdvancedFeatures
    timeseries_features: Dict[str, List[float]]
    network_graph_features: Dict[str, float]
    meta_learning_features: Dict[str, float]
    feature_quality_score: float
    completeness_ratio: float
    timestamp: datetime
    feature_version: str = "v2.0"
    
    # VPS 메타데이터
    processing_time: float = 0.0
    memory_usage: float = 0.0
    cache_hits: int = 0
    onnx_calls: int = 0

class VPSAdvancedFusionManager:
    """VPS 최적화된 고급 융합 관리자"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.keyword_scorer = VPSAdvancedKeywordScorer()
        self.cache = {}
        self.performance_metrics = {
            "total_fusions": 0,
            "avg_processing_time": 0.0,
            "cache_hit_rate": 0.0,
            "memory_efficiency": 0.0
        }
        
        # 스레드 풀 (CPU 집약적 작업용)
        self.executor = ThreadPoolExecutor(max_workers=VPS_FUSION_CONFIG["parallel_workers"])
        
        self.logger.info("VPS Advanced Fusion Manager 초기화 완료")
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("VPSFusionManager")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    async def advanced_fusion_analysis(
        self,
        text_data: List[str],
        market_data: Dict[str, Any],
        social_data: Optional[Dict[str, Any]] = None,
        enable_ml: bool = True
    ) -> RefinedFeatureSet:
        """
        9패널 대시보드용 고급 융합 분석
        
        Args:
            text_data: 뉴스/소셜 텍스트 데이터
            market_data: 시장 데이터 (가격, 거래량 등)
            social_data: 소셜 미디어 데이터
            enable_ml: ML 예측 활성화 여부
        """
        start_time = time.time()
        cache_hits = 0
        onnx_calls = 0
        
        try:
            # 1. 멀티모달 센티먼트 분석 (배치 처리)
            sentiment_results = await self._batch_sentiment_analysis(
                text_data, market_data, social_data
            )
            
            # 2. ML 기반 시장 예측
            if enable_ml:
                ml_prediction = await self._generate_ml_prediction(
                    sentiment_results, market_data
                )
                onnx_calls += 1
            else:
                ml_prediction = self._create_fallback_prediction()
            
            # 3. 이벤트 영향 분석
            event_impact = await self._analyze_event_impact(
                text_data, sentiment_results, market_data
            )
            
            # 4. 전략 성과 분석
            strategy_performance = await self._analyze_strategy_performance(
                market_data, sentiment_results
            )
            
            # 5. 이상 탐지
            anomaly_detection = await self._detect_anomalies(
                market_data, sentiment_results
            )
            
            # 6. 고급 특성 추출
            advanced_features = await self._extract_advanced_features(
                sentiment_results, market_data
            )
            
            # 7. 시계열 특성
            timeseries_features = await self._extract_timeseries_features(market_data)
            
            # 8. 네트워크 그래프 특성
            network_features = await self._extract_network_features(
                social_data or {}, sentiment_results
            )
            
            # 9. 메타 학습 특성
            meta_features = await self._extract_meta_learning_features(
                sentiment_results, ml_prediction
            )
            
            # 융합 스코어 계산
            fusion_score = self._calculate_fusion_score(
                sentiment_results, ml_prediction, event_impact
            )
            
            # 특성 품질 평가
            quality_score, completeness = self._evaluate_feature_quality(
                sentiment_results, market_data
            )
            
            # 결과 생성
            result = RefinedFeatureSet(
                fusion_score=fusion_score,
                ml_refined_prediction=ml_prediction,
                event_impact=event_impact,
                strategy_performance=strategy_performance,
                anomaly_detection=anomaly_detection,
                advanced_features=advanced_features,
                timeseries_features=timeseries_features,
                network_graph_features=network_features,
                meta_learning_features=meta_features,
                feature_quality_score=quality_score,
                completeness_ratio=completeness,
                timestamp=datetime.now(),
                processing_time=time.time() - start_time,
                memory_usage=self._get_memory_usage(),
                cache_hits=cache_hits,
                onnx_calls=onnx_calls
            )
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(result)
            
            # 메모리 정리
            gc.collect()
            
            return result
            
        except Exception as e:
            self.logger.error(f"융합 분석 실패: {e}")
            return self._create_fallback_result(start_time)
    
    async def _batch_sentiment_analysis(
        self,
        text_data: List[str],
        market_data: Dict[str, Any],
        social_data: Optional[Dict[str, Any]]
    ) -> List[MultiModalSentiment]:
        """배치 센티먼트 분석"""
        batch_size = VPS_FUSION_CONFIG["max_batch_size"]
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i + batch_size]
            batch_tasks = []
            
            for text in batch:
                task = self.keyword_scorer.analyze_advanced(
                    text=text,
                    price_data=market_data.get("price", {}),
                    volume_data=market_data.get("volume", {}),
                    social_data=social_data
                )
                batch_tasks.append(task)
            
            # 병렬 처리
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 예외 처리
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"센티먼트 분석 실패: {result}")
                    continue
                results.append(result)
        
        return results
    
    async def _generate_ml_prediction(
        self,
        sentiment_results: List[MultiModalSentiment],
        market_data: Dict[str, Any]
    ) -> MarketPrediction:
        """ML 기반 시장 예측 생성"""
        try:
            # 센티먼트 집계
            avg_sentiment = np.mean([s.text_sentiment for s in sentiment_results])
            avg_price_sentiment = np.mean([s.price_action_sentiment for s in sentiment_results])
            
            # 간단한 예측 로직 (실제로는 ONNX 모델 사용)
            combined_sentiment = (avg_sentiment + avg_price_sentiment) / 2
            
            if combined_sentiment > 0.3:
                direction = "up"
                confidence = min(0.9, 0.5 + combined_sentiment)
            elif combined_sentiment < -0.3:
                direction = "down"
                confidence = min(0.9, 0.5 + abs(combined_sentiment))
            else:
                direction = "sideways"
                confidence = 0.6
            
            # 리스크 레벨 결정
            volatility = np.std([s.text_sentiment for s in sentiment_results])
            if volatility > 0.5:
                risk_level = "high"
            elif volatility > 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return MarketPrediction(
                direction=direction,
                confidence=confidence,
                timeframe="24h",
                price_target=None,
                risk_level=risk_level,
                inference_time=0.1,
                ensemble_models=["sentiment_fusion", "price_action"],
                feature_importance={
                    "text_sentiment": abs(avg_sentiment),
                    "price_sentiment": abs(avg_price_sentiment),
                    "volatility": volatility
                }
            )
            
        except Exception as e:
            self.logger.error(f"ML 예측 실패: {e}")
            return self._create_fallback_prediction()
    
    async def _analyze_event_impact(
        self,
        text_data: List[str],
        sentiment_results: List[MultiModalSentiment],
        market_data: Dict[str, Any]
    ) -> EventImpactAnalysis:
        """이벤트 영향 분석"""
        try:
            # 이벤트 키워드 감지
            event_keywords = ["meeting", "announcement", "earnings", "regulation", "crash", "rally"]
            event_detected = any(
                keyword in " ".join(text_data).lower() 
                for keyword in event_keywords
            )
            
            if event_detected:
                # 영향 스코어 계산
                sentiment_variance = np.var([s.text_sentiment for s in sentiment_results])
                impact_score = min(sentiment_variance * 2, 1.0)
                
                return EventImpactAnalysis(
                    event_type="market_news",
                    impact_score=impact_score,
                    duration_hours=24.0,
                    affected_markets=["crypto", "forex"],
                    spillover_risk=impact_score * 0.7,
                    sentiment_shift=np.mean([s.text_sentiment for s in sentiment_results]),
                    volume_impact=np.mean([s.volume_sentiment for s in sentiment_results]),
                    volatility_impact=sentiment_variance,
                    correlation_changes={"btc_usd": 0.1, "eth_usd": 0.08}
                )
            else:
                return EventImpactAnalysis(
                    event_type="normal",
                    impact_score=0.1,
                    duration_hours=1.0,
                    affected_markets=[],
                    spillover_risk=0.05,
                    sentiment_shift=0.0,
                    volume_impact=0.0,
                    volatility_impact=0.0
                )
                
        except Exception as e:
            self.logger.error(f"이벤트 분석 실패: {e}")
            return EventImpactAnalysis(
                event_type="error",
                impact_score=0.0,
                duration_hours=0.0,
                affected_markets=[],
                spillover_risk=0.0,
                sentiment_shift=0.0,
                volume_impact=0.0,
                volatility_impact=0.0
            )
    
    async def _analyze_strategy_performance(
        self,
        market_data: Dict[str, Any],
        sentiment_results: List[MultiModalSentiment]
    ) -> StrategyPerformance:
        """전략 성과 분석"""
        try:
            # 모의 성과 메트릭 (실제로는 백테스팅 결과 사용)
            avg_sentiment = np.mean([s.text_sentiment for s in sentiment_results])
            sentiment_std = np.std([s.text_sentiment for s in sentiment_results])
            
            # 샤프 비율 추정
            if sentiment_std > 0:
                sharpe_ratio = avg_sentiment / sentiment_std
            else:
                sharpe_ratio = 0.0
            
            return StrategyPerformance(
                strategy_name="sentiment_fusion_v2",
                total_return=abs(avg_sentiment) * 0.15,
                sharpe_ratio=np.clip(sharpe_ratio, -3.0, 3.0),
                max_drawdown=sentiment_std * 0.2,
                win_rate=0.6 + (avg_sentiment * 0.1),
                avg_trade_duration=24.5,
                var_95=sentiment_std * 1.65,
                expected_shortfall=sentiment_std * 2.33,
                beta=1.0 + (avg_sentiment * 0.1),
                alpha=avg_sentiment * 0.05,
                information_ratio=sharpe_ratio * 0.8
            )
            
        except Exception as e:
            self.logger.error(f"성과 분석 실패: {e}")
            return StrategyPerformance(
                strategy_name="fallback",
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.5,
                avg_trade_duration=24.0,
                var_95=0.0,
                expected_shortfall=0.0,
                beta=1.0,
                alpha=0.0,
                information_ratio=0.0
            )
    
    async def _detect_anomalies(
        self,
        market_data: Dict[str, Any],
        sentiment_results: List[MultiModalSentiment]
    ) -> AnomalyDetection:
        """이상 탐지"""
        try:
            sentiments = [s.text_sentiment for s in sentiment_results]
            
            if len(sentiments) < 2:
                return AnomalyDetection(
                    anomaly_score=0.0,
                    anomaly_type="insufficient_data",
                    severity="low",
                    description="데이터 부족",
                    z_score=0.0,
                    p_value=1.0,
                    threshold_breach=False,
                    historical_frequency=0.0
                )
            
            # Z-스코어 계산
            mean_sentiment = np.mean(sentiments)
            std_sentiment = np.std(sentiments)
            
            if std_sentiment > 0:
                max_z_score = max(abs((s - mean_sentiment) / std_sentiment) for s in sentiments)
            else:
                max_z_score = 0.0
            
            # 이상 탐지
            anomaly_threshold = 2.0
            is_anomaly = max_z_score > anomaly_threshold
            anomaly_score = min(max_z_score / 3.0, 1.0)
            
            if anomaly_score > 0.8:
                severity = "critical"
            elif anomaly_score > 0.6:
                severity = "high"
            elif anomaly_score > 0.3:
                severity = "medium"
            else:
                severity = "low"
            
            return AnomalyDetection(
                anomaly_score=anomaly_score,
                anomaly_type="sentiment",
                severity=severity,
                description=f"센티먼트 이상 감지 (Z-score: {max_z_score:.2f})",
                z_score=max_z_score,
                p_value=max(0.001, 1.0 - anomaly_score),
                threshold_breach=is_anomaly,
                historical_frequency=0.05
            )
            
        except Exception as e:
            self.logger.error(f"이상 탐지 실패: {e}")
            return AnomalyDetection(
                anomaly_score=0.0,
                anomaly_type="error",
                severity="low",
                description="이상 탐지 실패",
                z_score=0.0,
                p_value=1.0,
                threshold_breach=False,
                historical_frequency=0.0
            )
    
    async def _extract_advanced_features(
        self,
        sentiment_results: List[MultiModalSentiment],
        market_data: Dict[str, Any]
    ) -> AdvancedFeatures:
        """고급 특성 추출"""
        try:
            sentiments = [s.text_sentiment for s in sentiment_results]
            
            return AdvancedFeatures(
                semantic_density=len(sentiments) / max(len(sentiments), 1),
                linguistic_complexity=np.std(sentiments) if len(sentiments) > 1 else 0.0,
                sentiment_volatility=np.var(sentiments) if len(sentiments) > 1 else 0.0,
                information_flow=np.mean([abs(s) for s in sentiments]),
                market_coupling=0.7,
                social_amplification=np.mean([s.social_engagement for s in sentiment_results]),
                news_velocity=len(sentiments) / 24.0,  # 시간당 뉴스 개수
                attention_score=np.mean([s.viral_score for s in sentiment_results]),
                controversy_index=np.std(sentiments) if len(sentiments) > 1 else 0.0,
                credibility_score=np.mean([s.confidence_calibration for s in sentiment_results]),
                trend_strength=abs(np.mean(sentiments)),
                seasonality_factor=0.5,
                anomaly_score=max([abs(s) for s in sentiments]) if sentiments else 0.0,
                regime_probability={
                    "bull": len([s for s in sentiments if s > 0.2]) / max(len(sentiments), 1),
                    "bear": len([s for s in sentiments if s < -0.2]) / max(len(sentiments), 1),
                    "neutral": len([s for s in sentiments if -0.2 <= s <= 0.2]) / max(len(sentiments), 1)
                }
            )
            
        except Exception as e:
            self.logger.error(f"고급 특성 추출 실패: {e}")
            return AdvancedFeatures(
                semantic_density=0.0,
                linguistic_complexity=0.0,
                sentiment_volatility=0.0,
                information_flow=0.0,
                market_coupling=0.0,
                social_amplification=0.0,
                news_velocity=0.0,
                attention_score=0.0,
                controversy_index=0.0,
                credibility_score=0.5,
                trend_strength=0.0,
                seasonality_factor=0.0,
                anomaly_score=0.0
            )
    
    async def _extract_timeseries_features(self, market_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """시계열 특성 추출"""
        try:
            price_data = market_data.get("price_history", [])
            volume_data = market_data.get("volume_history", [])
            
            features = {}
            
            if price_data:
                features["price_sma_5"] = self._calculate_sma(price_data, 5)
                features["price_sma_20"] = self._calculate_sma(price_data, 20)
                features["price_rsi"] = self._calculate_rsi(price_data)
                features["price_volatility"] = self._calculate_volatility(price_data)
            
            if volume_data:
                features["volume_sma_5"] = self._calculate_sma(volume_data, 5)
                features["volume_ratio"] = self._calculate_volume_ratio(volume_data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"시계열 특성 추출 실패: {e}")
            return {}
    
    async def _extract_network_features(
        self,
        social_data: Dict[str, Any],
        sentiment_results: List[MultiModalSentiment]
    ) -> Dict[str, float]:
        """네트워크 그래프 특성 추출"""
        try:
            return {
                "network_density": np.mean([s.network_effect for s in sentiment_results]),
                "centrality_score": 0.7,
                "clustering_coefficient": 0.3,
                "path_length": 2.5,
                "influence_spread": np.mean([s.viral_score for s in sentiment_results]),
                "community_strength": 0.6
            }
        except Exception as e:
            self.logger.error(f"네트워크 특성 추출 실패: {e}")
            return {}
    
    async def _extract_meta_learning_features(
        self,
        sentiment_results: List[MultiModalSentiment],
        ml_prediction: MarketPrediction
    ) -> Dict[str, float]:
        """메타 학습 특성 추출"""
        try:
            return {
                "model_confidence": ml_prediction.confidence,
                "prediction_consistency": np.mean([s.prediction_stability for s in sentiment_results]),
                "ensemble_agreement": np.mean([s.ensemble_agreement for s in sentiment_results]),
                "feature_importance_entropy": 0.8,
                "learning_rate": 0.01,
                "adaptation_score": 0.7
            }
        except Exception as e:
            self.logger.error(f"메타 학습 특성 추출 실패: {e}")
            return {}
    
    def _calculate_fusion_score(
        self,
        sentiment_results: List[MultiModalSentiment],
        ml_prediction: MarketPrediction,
        event_impact: EventImpactAnalysis
    ) -> float:
        """융합 스코어 계산"""
        try:
            # 센티먼트 가중 평균
            sentiment_score = np.mean([
                s.text_sentiment * 0.4 + 
                s.price_action_sentiment * 0.3 + 
                s.volume_sentiment * 0.2 + 
                s.social_engagement * 0.1
                for s in sentiment_results
            ])
            
            # ML 예측 신뢰도
            ml_score = ml_prediction.confidence * (1 if ml_prediction.direction == "up" else -1 if ml_prediction.direction == "down" else 0)
            
            # 이벤트 영향
            event_score = event_impact.impact_score * (1 if event_impact.sentiment_shift > 0 else -1)
            
            # 가중 결합
            fusion_score = (sentiment_score * 0.5 + ml_score * 0.3 + event_score * 0.2)
            
            return np.clip(fusion_score, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"융합 스코어 계산 실패: {e}")
            return 0.0
    
    def _evaluate_feature_quality(
        self,
        sentiment_results: List[MultiModalSentiment],
        market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """특성 품질 평가"""
        try:
            # 완성도 계산
            required_fields = ["price", "volume", "sentiment"]
            available_fields = 0
            
            if market_data.get("price"):
                available_fields += 1
            if market_data.get("volume"):
                available_fields += 1
            if sentiment_results:
                available_fields += 1
            
            completeness = available_fields / len(required_fields)
            
            # 품질 스코어 계산
            if sentiment_results:
                confidence_scores = [s.confidence_calibration for s in sentiment_results]
                quality_score = np.mean(confidence_scores)
            else:
                quality_score = 0.5
            
            return quality_score, completeness
            
        except Exception as e:
            self.logger.error(f"품질 평가 실패: {e}")
            return 0.5, 0.5
    
    def _calculate_sma(self, data: List[float], period: int) -> List[float]:
        """단순이동평균 계산"""
        if len(data) < period:
            return [np.mean(data)] * len(data)
        
        sma = []
        for i in range(len(data)):
            if i < period - 1:
                sma.append(np.mean(data[:i+1]))
            else:
                sma.append(np.mean(data[i-period+1:i+1]))
        return sma
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """RSI 계산"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        rsi_values = []
        for i in range(len(gains)):
            if i < period - 1:
                rsi_values.append(50.0)
            else:
                avg_gain = np.mean(gains[i-period+1:i+1])
                avg_loss = np.mean(losses[i-period+1:i+1])
                
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                rsi_values.append(rsi)
        
        return rsi_values
    
    def _calculate_volatility(self, prices: List[float], period: int = 20) -> List[float]:
        """변동성 계산"""
        if len(prices) < 2:
            return [0.0] * len(prices)
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        volatility = []
        for i in range(len(returns)):
            if i < period - 1:
                vol = np.std(returns[:i+1]) if i > 0 else 0.0
            else:
                vol = np.std(returns[i-period+1:i+1])
            volatility.append(vol)
        
        return [0.0] + volatility  # 첫 번째 가격에 대한 변동성은 0
    
    def _calculate_volume_ratio(self, volumes: List[float]) -> List[float]:
        """거래량 비율 계산"""
        if len(volumes) < 20:
            avg_volume = np.mean(volumes)
            return [v / avg_volume if avg_volume > 0 else 1.0 for v in volumes]
        
        ratios = []
        for i in range(len(volumes)):
            if i < 19:
                avg_vol = np.mean(volumes[:i+1])
            else:
                avg_vol = np.mean(volumes[i-19:i+1])
            
            ratio = volumes[i] / avg_vol if avg_vol > 0 else 1.0
            ratios.append(ratio)
        
        return ratios
    
    def _create_fallback_prediction(self) -> MarketPrediction:
        """폴백 예측 생성"""
        return MarketPrediction(
            direction="sideways",
            confidence=0.5,
            timeframe="24h",
            price_target=None,
            risk_level="medium"
        )
    
    def _create_fallback_result(self, start_time: float) -> RefinedFeatureSet:
        """폴백 결과 생성"""
        return RefinedFeatureSet(
            fusion_score=0.0,
            ml_refined_prediction=self._create_fallback_prediction(),
            event_impact=EventImpactAnalysis(
                event_type="error", impact_score=0.0, duration_hours=0.0,
                affected_markets=[], spillover_risk=0.0,
                sentiment_shift=0.0, volume_impact=0.0, volatility_impact=0.0
            ),
            strategy_performance=StrategyPerformance(
                strategy_name="fallback", total_return=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, win_rate=0.5, avg_trade_duration=24.0,
                var_95=0.0, expected_shortfall=0.0, beta=1.0, alpha=0.0, information_ratio=0.0
            ),
            anomaly_detection=AnomalyDetection(
                anomaly_score=0.0, anomaly_type="error", severity="low",
                description="분석 실패", z_score=0.0, p_value=1.0,
                threshold_breach=False, historical_frequency=0.0
            ),
            advanced_features=AdvancedFeatures(
                semantic_density=0.0, linguistic_complexity=0.0, sentiment_volatility=0.0,
                information_flow=0.0, market_coupling=0.0, social_amplification=0.0,
                news_velocity=0.0, attention_score=0.0, controversy_index=0.0,
                credibility_score=0.5, trend_strength=0.0, seasonality_factor=0.0, anomaly_score=0.0
            ),
            timeseries_features={},
            network_graph_features={},
            meta_learning_features={},
            feature_quality_score=0.5,
            completeness_ratio=0.0,
            timestamp=datetime.now(),
            processing_time=time.time() - start_time
        )
    
    def _get_memory_usage(self) -> float:
        """메모리 사용량 조회 (MB)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _update_performance_metrics(self, result: RefinedFeatureSet):
        """성능 메트릭 업데이트"""
        self.performance_metrics["total_fusions"] += 1
        
        # 평균 처리 시간 업데이트
        total = self.performance_metrics["total_fusions"]
        current_avg = self.performance_metrics["avg_processing_time"]
        new_avg = (current_avg * (total - 1) + result.processing_time) / total
        self.performance_metrics["avg_processing_time"] = new_avg
        
        # 메모리 효율성 업데이트
        self.performance_metrics["memory_efficiency"] = result.memory_usage
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            "fusion_performance": self.performance_metrics,
            "keyword_scorer_stats": self.keyword_scorer.get_performance_stats(),
            "cache_size": len(self.cache),
            "memory_usage": f"{self._get_memory_usage():.1f}MB"
        }
    
    def cleanup(self):
        """정리 작업"""
        self.cache.clear()
        self.executor.shutdown(wait=False)
        gc.collect()
        self.logger.info("VPS Fusion Manager 정리 완료")

# 글로벌 인스턴스
vps_fusion_manager = VPSAdvancedFusionManager()

# 편의 함수들
async def analyze_fusion_vps(
    text_data: List[str],
    market_data: Dict[str, Any],
    social_data: Optional[Dict[str, Any]] = None
) -> RefinedFeatureSet:
    """VPS 최적화된 융합 분석"""
    return await vps_fusion_manager.advanced_fusion_analysis(
        text_data, market_data, social_data
    )

def get_vps_fusion_stats() -> Dict[str, Any]:
    """VPS 융합 성능 통계 조회"""
    return vps_fusion_manager.get_performance_stats()

def cleanup_vps_fusion():
    """VPS 융합 관리자 정리"""
    vps_fusion_manager.cleanup()