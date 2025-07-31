#!/usr/bin/env python3
"""
Advanced Sentiment Fusion Manager for AuroraQ
고도화된 멀티모달 감정 융합 및 ML 피처 생성
"""

import asyncio
import time
import statistics
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from collections import deque
import hashlib

# 로컬 임포트 (상대 import 문제 해결을 위해 조건부 import)
try:
    from .sentiment_fusion_manager import SentimentFusionManager, FusedSentiment, SentimentSignal, FusionMethod
    from ..models.advanced_keyword_scorer import AdvancedKeywordScorer, MultiModalSentiment, AdvancedFeatures
    from ..utils.content_cache_manager import ContentCacheManager, ContentMetadata
except ImportError:
    # 절대 import 대안
    from sentiment_fusion_manager import SentimentFusionManager, FusedSentiment, SentimentSignal, FusionMethod
    from models.advanced_keyword_scorer import AdvancedKeywordScorer, MultiModalSentiment, AdvancedFeatures
    from utils.content_cache_manager import ContentCacheManager, ContentMetadata

logger = logging.getLogger(__name__)

class MLRefinementMethod(Enum):
    """ML 리파인 방법"""
    ENSEMBLE_VOTING = "ensemble_voting"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"

class PredictionConfidence(Enum):
    """예측 신뢰도 레벨"""
    VERY_HIGH = "very_high"  # 95%+
    HIGH = "high"           # 85-95%
    MEDIUM = "medium"       # 70-85%
    LOW = "low"             # 50-70%
    VERY_LOW = "very_low"   # <50%

@dataclass
class MarketPrediction:
    """시장 예측 결과"""
    direction: str  # "bullish", "bearish", "neutral"
    probability: float  # 0.0 ~ 1.0
    volatility_forecast: float  # 예상 변동성
    confidence_level: PredictionConfidence
    time_horizon: str  # "1h", "4h", "24h", "1w"
    
    # 고급 예측 지표
    trend_strength: float
    momentum_persistence: float
    regime_stability: float
    black_swan_risk: float
    
    # 메타 정보
    model_ensemble_size: int
    prediction_uncertainty: float
    feature_importance: Dict[str, float]
    created_at: datetime

@dataclass
class EventImpactAnalysis:
    """이벤트 영향도 분석"""
    event_type: str  # "news", "social", "technical", "regulatory"
    impact_score: float  # 0.0 ~ 1.0
    lag_estimate: float  # 예상 지연시간 (분)
    duration_estimate: float  # 예상 지속시간 (시간)
    
    # 파급효과 분석
    market_spillover: float  # 시장 파급효과
    cross_asset_correlation: float  # 자산간 상관관계 변화
    volatility_impact: float  # 변동성 영향
    liquidity_impact: float  # 유동성 영향
    
    # 신뢰도
    prediction_accuracy: float
    historical_precedence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyPerformance:
    """전략 실시간 성과 지표"""
    strategy_name: str
    roi: float  # Return on Investment
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    # 고급 리스크 지표
    var_95: float  # Value at Risk 95%
    expected_shortfall: float  # 기대 손실
    beta: float  # 시장 베타
    alpha: float  # 알파
    
    # 최근 성과
    last_7_days_return: float
    last_30_days_return: float
    ytd_return: float
    
    # 포지션 정보
    current_positions: Dict[str, float]
    exposure: float
    leverage: float
    
    updated_at: datetime

@dataclass
class AnomalyDetection:
    """이상 탐지 결과"""
    anomaly_flag: bool
    anomaly_score: float  # 0.0 ~ 1.0
    anomaly_type: str  # "price", "volume", "sentiment", "correlation"
    severity: str  # "low", "medium", "high", "critical"
    
    # 이벤트 태깅
    event_tag: Optional[str]  # "earnings", "news", "manipulation", "technical"
    related_events: List[str]
    
    # 상세 분석
    statistical_significance: float
    historical_comparison: float
    deviation_magnitude: float
    
    # 대응 권고
    recommended_action: str  # "monitor", "investigate", "alert", "hedge"
    confidence: float
    
    detected_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RefinedFeatureSet:
    """PPO/백테스트용 정제 피처 세트"""
    # 핵심 신호
    fusion_score: float
    ml_refined_prediction: MarketPrediction
    event_impact: EventImpactAnalysis
    strategy_performance: StrategyPerformance
    anomaly_detection: AnomalyDetection
    
    # 고급 피처
    advanced_features: AdvancedFeatures
    
    # 시계열 구조화 데이터
    timeseries_features: Dict[str, List[float]]
    
    # 네트워크 그래프 피처
    network_graph_features: Dict[str, float]
    
    # 메타 학습 피처
    meta_learning_features: Dict[str, float]
    
    # 품질 지표
    feature_quality_score: float
    completeness_ratio: float
    
    # 타임스탬프
    timestamp: datetime
    feature_version: str = "v2.0"

class AdvancedFusionManager(SentimentFusionManager):
    """고도화된 감정 융합 매니저"""
    
    def __init__(self,
                 advanced_scorer: AdvancedKeywordScorer,
                 cache_manager: Optional[ContentCacheManager] = None,
                 ml_refinement_enabled: bool = True):
        """
        초기화
        
        Args:
            advanced_scorer: 고급 키워드 스코어러
            cache_manager: 캐시 매니저
            ml_refinement_enabled: ML 리파인 활성화
        """
        # 기본 초기화 (부모 클래스 초기화를 위해 임시로)
        super().__init__(
            keyword_scorer=advanced_scorer,
            finbert_processor=None,  # 나중에 설정
            cache_manager=cache_manager
        )
        
        self.advanced_scorer = advanced_scorer
        self.ml_refinement_enabled = ml_refinement_enabled
        
        # 고급 설정
        self._initialize_advanced_models()
        self._initialize_ml_pipeline()
        self._initialize_anomaly_detection()
        
        # 데이터 저장소
        self.prediction_history = deque(maxlen=10000)
        self.event_history = deque(maxlen=5000)
        self.strategy_performance_cache = {}
        self.anomaly_cache = deque(maxlen=1000)
        
        # 성능 통계
        self.advanced_stats = {
            "ml_predictions": 0,
            "anomalies_detected": 0,
            "strategies_tracked": 0,
            "feature_extractions": 0,
            "avg_prediction_accuracy": 0.0,
            "model_confidence_avg": 0.0
        }
        
        logger.info("AdvancedFusionManager initialized successfully")
    
    def _initialize_advanced_models(self):
        """고급 모델 초기화"""
        # ML 모델 앙상블 설정
        self.ml_models = {
            "gradient_boosting": None,  # 실제 모델로 교체 필요
            "neural_network": None,
            "transformer": None,
            "ensemble_meta": None
        }
        
        # 예측 파라미터
        self.prediction_params = {
            "time_horizons": ["1h", "4h", "24h", "1w"],
            "confidence_thresholds": {
                "very_high": 0.95,
                "high": 0.85,
                "medium": 0.70,
                "low": 0.50
            },
            "ensemble_weights": {
                "sentiment": 0.35,
                "technical": 0.25,
                "fundamental": 0.20,
                "social": 0.15,
                "macro": 0.05
            }
        }
    
    def _initialize_ml_pipeline(self):
        """ML 파이프라인 초기화"""
        self.feature_engineering = {
            "polynomial_features": True,
            "interaction_terms": True,
            "time_series_decomposition": True,
            "fourier_transforms": True,
            "wavelet_transforms": False  # 고급 기능
        }
        
        self.model_selection = {
            "cross_validation_folds": 5,
            "hyperparameter_optimization": True,
            "early_stopping": True,
            "regularization": True
        }
    
    def _initialize_anomaly_detection(self):
        """이상 탐지 시스템 초기화"""
        self.anomaly_detectors = {
            "isolation_forest": None,  # 실제 모델로 교체 필요
            "local_outlier_factor": None,
            "one_class_svm": None,
            "statistical_tests": None
        }
        
        self.anomaly_thresholds = {
            "price_anomaly": 3.0,      # 3σ
            "volume_anomaly": 2.5,     # 2.5σ
            "sentiment_anomaly": 2.0,  # 2σ
            "correlation_anomaly": 0.8  # 상관관계 임계값
        }
    
    async def advanced_fusion_analysis(self,
                                     content_hash: str,
                                     text: str,
                                     market_data: Optional[Dict] = None,
                                     social_data: Optional[Dict] = None,
                                     force_refresh: bool = False) -> RefinedFeatureSet:
        """고도화된 융합 분석"""
        start_time = time.time()
        
        try:
            # 1. 기본 멀티모달 감정 분석
            multimodal_result = self.advanced_scorer.analyze_advanced(
                text=text,
                price_data=market_data.get('price_data') if market_data else None,
                volume_data=market_data.get('volume_data') if market_data else None,
                social_data=social_data
            )
            
            # 2. 고급 피처 추출
            advanced_features = self.advanced_scorer.extract_advanced_features(multimodal_result)
            
            # 3. ML 리파인 예측
            ml_prediction = await self._generate_ml_prediction(multimodal_result, market_data)
            
            # 4. 이벤트 영향도 분석
            event_impact = await self._analyze_event_impact(text, multimodal_result, market_data)
            
            # 5. 전략 성과 분석
            strategy_performance = await self._analyze_strategy_performance(market_data)
            
            # 6. 이상 탐지
            anomaly_detection = await self._detect_anomalies(multimodal_result, market_data)
            
            # 7. 시계열 피처 생성
            timeseries_features = self._generate_timeseries_features(market_data)
            
            # 8. 네트워크 그래프 피처
            network_features = self._generate_network_features(social_data, multimodal_result)
            
            # 9. 메타 학습 피처
            meta_features = self._generate_meta_learning_features(multimodal_result, ml_prediction)
            
            # 10. 품질 점수 계산
            quality_score, completeness = self._calculate_feature_quality(
                advanced_features, ml_prediction, event_impact
            )
            
            # 11. 통합 피처 세트 생성
            refined_features = RefinedFeatureSet(
                fusion_score=multimodal_result.text_sentiment,
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
                timestamp=datetime.now()
            )
            
            # 12. 히스토리 업데이트
            self._update_analysis_history(refined_features)
            
            # 13. 통계 업데이트
            self._update_advanced_stats(time.time() - start_time)
            
            logger.info(f"Advanced fusion analysis completed: {quality_score:.3f} quality score")
            
            return refined_features
            
        except Exception as e:
            logger.error(f"Advanced fusion analysis failed: {e}", exc_info=True)
            return self._create_default_refined_features(start_time)
    
    async def _generate_ml_prediction(self,
                                    multimodal_result: MultiModalSentiment,
                                    market_data: Optional[Dict]) -> MarketPrediction:
        """ML 리파인 예측 생성"""
        try:
            # 특성 벡터 구성
            features = self._build_feature_vector(multimodal_result, market_data)
            
            # 앙상블 예측
            predictions = {}
            confidences = {}
            
            # 각 모델별 예측 (실제 구현에서는 훈련된 모델 사용)
            if self.ml_refinement_enabled:
                # 그래디언트 부스팅 예측
                predictions['gb'] = self._predict_gradient_boosting(features)
                confidences['gb'] = 0.85
                
                # 신경망 예측
                predictions['nn'] = self._predict_neural_network(features)
                confidences['nn'] = 0.78
                
                # 트랜스포머 예측
                predictions['transformer'] = self._predict_transformer(features)
                confidences['transformer'] = 0.82
            else:
                # 간단한 룰 기반 예측
                predictions['rule_based'] = self._predict_rule_based(multimodal_result)
                confidences['rule_based'] = 0.70
            
            # 앙상블 결합
            final_prediction = self._ensemble_predictions(predictions, confidences)
            
            # 신뢰도 레벨 결정
            avg_confidence = statistics.mean(confidences.values())
            confidence_level = self._determine_confidence_level(avg_confidence)
            
            # 고급 지표 계산
            trend_strength = abs(final_prediction['direction_score'])
            momentum_persistence = self._calculate_momentum_persistence(market_data)
            regime_stability = self._calculate_regime_stability(market_data)
            black_swan_risk = multimodal_result.black_swan_probability
            
            return MarketPrediction(
                direction=final_prediction['direction'],
                probability=final_prediction['probability'],
                volatility_forecast=final_prediction['volatility'],
                confidence_level=confidence_level,
                time_horizon="4h",  # 기본값
                trend_strength=trend_strength,
                momentum_persistence=momentum_persistence,
                regime_stability=regime_stability,
                black_swan_risk=black_swan_risk,
                model_ensemble_size=len(predictions),
                prediction_uncertainty=1.0 - avg_confidence,
                feature_importance=self._calculate_feature_importance(features),
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"ML prediction generation failed: {e}")
            return self._create_default_prediction()
    
    async def _analyze_event_impact(self,
                                  text: str,
                                  multimodal_result: MultiModalSentiment,
                                  market_data: Optional[Dict]) -> EventImpactAnalysis:
        """이벤트 영향도 분석"""
        try:
            # 이벤트 타입 분류
            event_type = self._classify_event_type(text)
            
            # 영향도 스코어 계산
            impact_score = self._calculate_impact_score(text, multimodal_result)
            
            # 지연시간 추정
            lag_estimate = self._estimate_reaction_lag(event_type, impact_score)
            
            # 지속시간 추정
            duration_estimate = self._estimate_impact_duration(event_type, impact_score)
            
            # 파급효과 분석
            spillover = self._analyze_market_spillover(event_type, impact_score)
            correlation_change = self._analyze_correlation_impact(event_type)
            volatility_impact = impact_score * multimodal_result.viral_score
            liquidity_impact = self._estimate_liquidity_impact(event_type, impact_score)
            
            # 예측 정확도 (과거 유사 이벤트 기반)
            prediction_accuracy = self._estimate_prediction_accuracy(event_type)
            historical_precedence = self._find_historical_precedence(text, event_type)
            
            return EventImpactAnalysis(
                event_type=event_type,
                impact_score=impact_score,
                lag_estimate=lag_estimate,
                duration_estimate=duration_estimate,
                market_spillover=spillover,
                cross_asset_correlation=correlation_change,
                volatility_impact=volatility_impact,
                liquidity_impact=liquidity_impact,
                prediction_accuracy=prediction_accuracy,
                historical_precedence=historical_precedence,
                metadata={
                    "text_snippet": text[:100],
                    "viral_score": multimodal_result.viral_score,
                    "network_effect": multimodal_result.network_effect
                }
            )
            
        except Exception as e:
            logger.error(f"Event impact analysis failed: {e}")
            return self._create_default_event_impact()
    
    async def _analyze_strategy_performance(self, market_data: Optional[Dict]) -> StrategyPerformance:
        """전략 성과 분석"""
        try:
            # 기본 전략 성과 (모의)
            strategy_name = "AuroraQ_Sentiment_Strategy"
            
            # 수익률 계산 (실제 구현에서는 실제 포트폴리오 데이터 사용)
            roi = np.random.normal(0.08, 0.15)  # 8% 평균, 15% 변동성
            sharpe_ratio = np.random.uniform(0.5, 2.0)
            max_drawdown = np.random.uniform(0.05, 0.25)
            win_rate = np.random.uniform(0.45, 0.65)
            
            # 위험 지표
            var_95 = abs(np.random.normal(0.02, 0.01))
            expected_shortfall = var_95 * 1.3
            beta = np.random.uniform(0.8, 1.2)
            alpha = roi - beta * 0.05  # 시장 수익률 5% 가정
            
            # 최근 성과
            last_7_days = np.random.normal(0.01, 0.03)
            last_30_days = np.random.normal(0.03, 0.08)
            ytd_return = np.random.normal(0.12, 0.20)
            
            # 포지션 정보
            current_positions = {
                "BTC": 0.4,
                "ETH": 0.3,
                "CASH": 0.3
            }
            exposure = 0.7
            leverage = 1.2
            
            return StrategyPerformance(
                strategy_name=strategy_name,
                roi=roi,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                beta=beta,
                alpha=alpha,
                last_7_days_return=last_7_days,
                last_30_days_return=last_30_days,
                ytd_return=ytd_return,
                current_positions=current_positions,
                exposure=exposure,
                leverage=leverage,
                updated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Strategy performance analysis failed: {e}")
            return self._create_default_strategy_performance()
    
    async def _detect_anomalies(self,
                              multimodal_result: MultiModalSentiment,
                              market_data: Optional[Dict]) -> AnomalyDetection:
        """이상 탐지"""
        try:
            anomalies = []
            max_score = 0.0
            primary_type = "normal"
            
            # 감정 이상 탐지
            sentiment_anomaly = self._detect_sentiment_anomaly(multimodal_result)
            if sentiment_anomaly['is_anomaly']:
                anomalies.append(sentiment_anomaly)
                if sentiment_anomaly['score'] > max_score:
                    max_score = sentiment_anomaly['score']
                    primary_type = "sentiment"
            
            # 가격 이상 탐지
            if market_data and 'price_data' in market_data:
                price_anomaly = self._detect_price_anomaly(market_data['price_data'])
                if price_anomaly['is_anomaly']:
                    anomalies.append(price_anomaly)
                    if price_anomaly['score'] > max_score:
                        max_score = price_anomaly['score']
                        primary_type = "price"
            
            # 거래량 이상 탐지
            if market_data and 'volume_data' in market_data:
                volume_anomaly = self._detect_volume_anomaly(market_data['volume_data'])
                if volume_anomaly['is_anomaly']:
                    anomalies.append(volume_anomaly)
                    if volume_anomaly['score'] > max_score:
                        max_score = volume_anomaly['score']
                        primary_type = "volume"
            
            # 상관관계 이상 탐지
            correlation_anomaly = self._detect_correlation_anomaly(multimodal_result)
            if correlation_anomaly['is_anomaly']:
                anomalies.append(correlation_anomaly)
                if correlation_anomaly['score'] > max_score:
                    max_score = correlation_anomaly['score']
                    primary_type = "correlation"
            
            # 종합 이상 플래그
            anomaly_flag = len(anomalies) > 0
            
            # 심각도 결정
            severity = self._determine_anomaly_severity(max_score)
            
            # 이벤트 태깅
            event_tag = self._tag_anomaly_event(anomalies, multimodal_result) if anomaly_flag else None
            
            # 관련 이벤트
            related_events = self._find_related_events(anomalies) if anomaly_flag else []
            
            # 통계적 유의성
            statistical_significance = max_score if anomaly_flag else 0.0
            
            # 과거 비교
            historical_comparison = self._compare_with_history(anomalies) if anomaly_flag else 0.0
            
            # 편차 크기
            deviation_magnitude = max_score
            
            # 대응 권고
            recommended_action = self._recommend_action(severity, primary_type)
            
            # 신뢰도
            confidence = min(max_score + 0.2, 1.0) if anomaly_flag else 0.8
            
            return AnomalyDetection(
                anomaly_flag=anomaly_flag,
                anomaly_score=max_score,
                anomaly_type=primary_type,
                severity=severity,
                event_tag=event_tag,
                related_events=related_events,
                statistical_significance=statistical_significance,
                historical_comparison=historical_comparison,
                deviation_magnitude=deviation_magnitude,
                recommended_action=recommended_action,
                confidence=confidence,
                detected_at=datetime.now(),
                metadata={
                    "total_anomalies": len(anomalies),
                    "anomaly_details": anomalies
                }
            )
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return self._create_default_anomaly_detection()
    
    def _generate_timeseries_features(self, market_data: Optional[Dict]) -> Dict[str, List[float]]:
        """시계열 피처 생성"""
        try:
            if not market_data:
                return {}
            
            features = {}
            
            # 가격 기반 피처
            if 'price_data' in market_data and 'prices' in market_data['price_data']:
                prices = market_data['price_data']['prices'][-100:]  # 최근 100개
                
                # 수익률 시계열
                returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else []
                features['returns'] = returns.tolist() if len(returns) > 0 else []
                
                # 누적 수익률
                cumulative_returns = np.cumprod(1 + np.array(returns)) - 1 if len(returns) > 0 else []
                features['cumulative_returns'] = cumulative_returns.tolist() if len(cumulative_returns) > 0 else []
                
                # 이동평균
                if len(prices) >= 20:
                    ma_20 = np.convolve(prices, np.ones(20)/20, mode='valid')
                    features['ma_20'] = ma_20.tolist()
                
                # 볼린저 밴드
                if len(prices) >= 20:
                    rolling_std = np.array([np.std(prices[max(0, i-19):i+1]) for i in range(19, len(prices))])
                    upper_band = ma_20 + 2 * rolling_std
                    lower_band = ma_20 - 2 * rolling_std
                    features['bollinger_upper'] = upper_band.tolist()
                    features['bollinger_lower'] = lower_band.tolist()
            
            # 거래량 기반 피처
            if 'volume_data' in market_data and 'volumes' in market_data['volume_data']:
                volumes = market_data['volume_data']['volumes'][-100:]
                
                # 거래량 변화율
                volume_changes = np.diff(volumes) / (np.array(volumes[:-1]) + 1e-8) if len(volumes) > 1 else []
                features['volume_changes'] = volume_changes.tolist() if len(volume_changes) > 0 else []
                
                # 거래량 이동평균
                if len(volumes) >= 20:
                    volume_ma = np.convolve(volumes, np.ones(20)/20, mode='valid')
                    features['volume_ma_20'] = volume_ma.tolist()
            
            return features
            
        except Exception as e:
            logger.error(f"Timeseries feature generation failed: {e}")
            return {}
    
    def _generate_network_features(self, social_data: Optional[Dict], multimodal_result: MultiModalSentiment) -> Dict[str, float]:
        """네트워크 그래프 피처 생성"""
        try:
            features = {}
            
            # 기본 네트워크 지표
            features['network_centrality'] = multimodal_result.network_effect
            features['viral_coefficient'] = multimodal_result.viral_score
            features['social_clustering'] = multimodal_result.herding_behavior
            
            # 소셜 데이터 기반 네트워크 피처
            if social_data:
                # 플랫폼 간 연결성
                platforms = ['twitter', 'reddit', 'other_platforms']
                active_platforms = sum(1 for p in platforms if p in social_data)
                features['platform_connectivity'] = active_platforms / len(platforms)
                
                # 정보 확산 속도
                if 'retweet_velocity' in social_data:
                    features['information_diffusion_rate'] = min(social_data['retweet_velocity'] / 100, 1.0)
                
                # 네트워크 밀도
                total_interactions = 0
                for platform in platforms:
                    if platform in social_data:
                        platform_data = social_data[platform]
                        if isinstance(platform_data, dict):
                            total_interactions += sum(platform_data.values())
                
                features['network_density'] = min(np.log1p(total_interactions) / 10, 1.0)
                
                # 영향력 분포
                if 'influencer_amplification' in social_data:
                    features['influence_distribution'] = social_data['influencer_amplification']
                else:
                    features['influence_distribution'] = 0.5  # 기본값
            
            # 시간적 네트워크 특성
            features['temporal_persistence'] = multimodal_result.prediction_stability
            features['cascade_potential'] = (features.get('network_density', 0.5) * 
                                           features.get('viral_coefficient', 0.0))
            
            return features
            
        except Exception as e:
            logger.error(f"Network feature generation failed: {e}")
            return {}
    
    def _generate_meta_learning_features(self, multimodal_result: MultiModalSentiment, ml_prediction: MarketPrediction) -> Dict[str, float]:
        """메타 학습 피처 생성"""
        try:
            features = {}
            
            # 모델 성능 메타 피처
            features['model_confidence'] = multimodal_result.confidence_calibration
            features['prediction_uncertainty'] = ml_prediction.prediction_uncertainty
            features['ensemble_agreement'] = multimodal_result.ensemble_agreement
            features['model_stability'] = multimodal_result.prediction_stability
            
            # 학습 동역학
            features['learning_rate_indicator'] = 0.1  # 모델 적응 속도
            features['concept_drift_score'] = 0.05     # 개념 드리프트 정도
            features['model_complexity'] = ml_prediction.model_ensemble_size / 10.0
            
            # 히스토리 기반 메타 피처
            if len(self.prediction_history) > 5:
                recent_predictions = list(self.prediction_history)[-5:]
                
                # 예측 일관성
                prediction_scores = [p.ml_refined_prediction.probability for p in recent_predictions]
                features['prediction_consistency'] = 1.0 - np.std(prediction_scores)
                
                # 정확도 트렌드
                # 실제 구현에서는 실제 결과와 비교하여 계산
                features['accuracy_trend'] = 0.75  # 모의값
                
                # 과신 편향
                confidence_scores = [p.ml_refined_prediction.confidence_level.value for p in recent_predictions]
                features['overconfidence_bias'] = len(set(confidence_scores)) / len(confidence_scores)
            else:
                features['prediction_consistency'] = 0.5
                features['accuracy_trend'] = 0.5
                features['overconfidence_bias'] = 0.5
            
            # 컨텍스트 적응
            features['context_adaptation_score'] = (
                multimodal_result.emotional_state.value.count('_') + 1  # 복잡성 지표
            ) / 3.0  # 정규화
            
            # 불확실성 정량화
            features['epistemic_uncertainty'] = ml_prediction.prediction_uncertainty * 0.7
            features['aleatoric_uncertainty'] = ml_prediction.prediction_uncertainty * 0.3
            
            return features
            
        except Exception as e:
            logger.error(f"Meta learning feature generation failed: {e}")
            return {}
    
    # 유틸리티 메서드들
    def _build_feature_vector(self, multimodal_result: MultiModalSentiment, market_data: Optional[Dict]) -> np.ndarray:
        """특성 벡터 구성"""
        features = []
        
        # 멀티모달 감정 피처
        features.extend([
            multimodal_result.text_sentiment,
            multimodal_result.price_action_sentiment,
            multimodal_result.volume_sentiment,
            multimodal_result.social_engagement
        ])
        
        # 시계열 모멘텀 피처
        features.extend([
            multimodal_result.momentum_1h,
            multimodal_result.momentum_4h,
            multimodal_result.momentum_24h
        ])
        
        # 위험 지표
        features.extend([
            multimodal_result.black_swan_probability,
            multimodal_result.tail_risk,
            multimodal_result.herding_behavior,
            multimodal_result.panic_indicator
        ])
        
        # 네트워크 효과
        features.extend([
            multimodal_result.viral_score,
            multimodal_result.network_effect
        ])
        
        # 메타 피처
        features.extend([
            multimodal_result.confidence_calibration,
            multimodal_result.prediction_stability,
            multimodal_result.ensemble_agreement
        ])
        
        return np.array(features)
    
    def _predict_gradient_boosting(self, features: np.ndarray) -> Dict:
        """그래디언트 부스팅 예측 (모의)"""
        # 실제 구현에서는 훈련된 모델 사용
        weighted_sum = np.sum(features * np.random.uniform(0.5, 1.5, len(features)))
        probability = 1 / (1 + np.exp(-weighted_sum))  # 시그모이드
        direction = "bullish" if probability > 0.6 else "bearish" if probability < 0.4 else "neutral"
        
        return {
            "direction": direction,
            "probability": probability,
            "volatility": min(abs(weighted_sum) * 0.1, 0.5),
            "direction_score": weighted_sum
        }
    
    def _predict_neural_network(self, features: np.ndarray) -> Dict:
        """신경망 예측 (모의)"""
        # 모의 신경망 출력
        hidden = np.tanh(features.dot(np.random.randn(len(features), 5)))
        output = hidden.dot(np.random.randn(5, 3))  # 3개 클래스
        probabilities = np.exp(output) / np.sum(np.exp(output))  # 소프트맥스
        
        directions = ["bearish", "neutral", "bullish"]
        max_idx = np.argmax(probabilities)
        
        return {
            "direction": directions[max_idx],
            "probability": probabilities[max_idx],
            "volatility": np.std(probabilities) * 2,
            "direction_score": output[max_idx]
        }
    
    def _predict_transformer(self, features: np.ndarray) -> Dict:
        """트랜스포머 예측 (모의)"""
        # 모의 어텐션 메커니즘
        attention_weights = np.exp(features) / np.sum(np.exp(features))
        weighted_features = features * attention_weights
        
        prediction_score = np.sum(weighted_features)
        probability = 1 / (1 + np.exp(-prediction_score * 2))
        
        direction = "bullish" if probability > 0.65 else "bearish" if probability < 0.35 else "neutral"
        
        return {
            "direction": direction,
            "probability": probability,
            "volatility": np.std(attention_weights) * 0.5,
            "direction_score": prediction_score
        }
    
    def _predict_rule_based(self, multimodal_result: MultiModalSentiment) -> Dict:
        """룰 기반 예측"""
        sentiment_score = multimodal_result.text_sentiment
        momentum = (multimodal_result.momentum_1h + multimodal_result.momentum_4h) / 2
        risk_factor = (multimodal_result.black_swan_probability + multimodal_result.tail_risk) / 2
        
        combined_score = sentiment_score * 0.5 + momentum * 0.3 - risk_factor * 0.2
        
        probability = 1 / (1 + np.exp(-combined_score * 3))
        direction = "bullish" if combined_score > 0.2 else "bearish" if combined_score < -0.2 else "neutral"
        
        return {
            "direction": direction,
            "probability": probability,
            "volatility": risk_factor * 2,
            "direction_score": combined_score
        }
    
    def _ensemble_predictions(self, predictions: Dict, confidences: Dict) -> Dict:
        """앙상블 예측 결합"""
        # 가중 평균
        total_weight = sum(confidences.values())
        weighted_probabilities = []
        weighted_volatilities = []
        direction_scores = []
        
        for model, pred in predictions.items():
            weight = confidences[model] / total_weight
            weighted_probabilities.append(pred['probability'] * weight)
            weighted_volatilities.append(pred['volatility'] * weight)
            direction_scores.append(pred['direction_score'] * weight)
        
        final_probability = sum(weighted_probabilities)
        final_volatility = sum(weighted_volatilities)
        final_direction_score = sum(direction_scores)
        
        # 최종 방향 결정
        final_direction = "bullish" if final_direction_score > 0.1 else "bearish" if final_direction_score < -0.1 else "neutral"
        
        return {
            "direction": final_direction,
            "probability": final_probability,
            "volatility": final_volatility,
            "direction_score": final_direction_score
        }
    
    def _determine_confidence_level(self, confidence: float) -> PredictionConfidence:
        """신뢰도 레벨 결정"""
        if confidence >= 0.95:
            return PredictionConfidence.VERY_HIGH
        elif confidence >= 0.85:
            return PredictionConfidence.HIGH
        elif confidence >= 0.70:
            return PredictionConfidence.MEDIUM
        elif confidence >= 0.50:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    # 추가 유틸리티 메서드들 (공간 절약을 위해 간략화)
    def _calculate_momentum_persistence(self, market_data: Optional[Dict]) -> float:
        return np.random.uniform(0.3, 0.8)  # 모의값
    
    def _calculate_regime_stability(self, market_data: Optional[Dict]) -> float:
        return np.random.uniform(0.4, 0.9)  # 모의값
    
    def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        feature_names = ['text_sentiment', 'price_sentiment', 'volume_sentiment', 'social_engagement',
                        'momentum_1h', 'momentum_4h', 'momentum_24h', 'black_swan_prob', 'tail_risk',
                        'herding', 'panic', 'viral_score', 'network_effect', 'confidence_cal',
                        'pred_stability', 'ensemble_agreement']
        
        importances = np.abs(features) / np.sum(np.abs(features))
        return dict(zip(feature_names[:len(importances)], importances.tolist()))
    
    def _classify_event_type(self, text: str) -> str:
        """이벤트 타입 분류"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['earnings', 'financial', 'revenue', 'profit']):
            return 'earnings'
        elif any(word in text_lower for word in ['regulation', 'regulatory', 'ban', 'legal']):
            return 'regulatory'
        elif any(word in text_lower for word in ['technical', 'chart', 'breakout', 'support', 'resistance']):
            return 'technical'
        elif any(word in text_lower for word in ['news', 'announcement', 'partnership', 'adoption']):
            return 'news'
        else:
            return 'social'
    
    def _calculate_impact_score(self, text: str, multimodal_result: MultiModalSentiment) -> float:
        """영향도 스코어 계산"""
        base_score = abs(multimodal_result.text_sentiment) * 0.5
        viral_bonus = multimodal_result.viral_score * 0.3
        network_bonus = multimodal_result.network_effect * 0.2
        return min(base_score + viral_bonus + network_bonus, 1.0)
    
    def _estimate_reaction_lag(self, event_type: str, impact_score: float) -> float:
        """반응 지연시간 추정 (분)"""
        base_lags = {
            'technical': 5,
            'social': 15,
            'news': 30,
            'regulatory': 60,
            'earnings': 120
        }
        base_lag = base_lags.get(event_type, 30)
        return base_lag * (1.0 - impact_score * 0.5)  # 영향도가 높을수록 빠른 반응
    
    def _estimate_impact_duration(self, event_type: str, impact_score: float) -> float:
        """영향 지속시간 추정 (시간)"""
        base_durations = {
            'technical': 2,
            'social': 6,
            'news': 24,
            'regulatory': 168,  # 1주
            'earnings': 72
        }
        base_duration = base_durations.get(event_type, 12)
        return base_duration * (0.5 + impact_score * 1.5)  # 영향도에 따른 조정
    
    # 기본값 생성 메서드들
    def _create_default_prediction(self) -> MarketPrediction:
        """기본 예측 생성"""
        return MarketPrediction(
            direction="neutral",
            probability=0.5,
            volatility_forecast=0.1,
            confidence_level=PredictionConfidence.MEDIUM,
            time_horizon="4h",
            trend_strength=0.0,
            momentum_persistence=0.5,
            regime_stability=0.5,
            black_swan_risk=0.01,
            model_ensemble_size=1,
            prediction_uncertainty=0.5,
            feature_importance={},
            created_at=datetime.now()
        )
    
    def _create_default_event_impact(self) -> EventImpactAnalysis:
        """기본 이벤트 영향도 생성"""
        return EventImpactAnalysis(
            event_type="unknown",
            impact_score=0.0,
            lag_estimate=30.0,
            duration_estimate=12.0,
            market_spillover=0.0,
            cross_asset_correlation=0.0,
            volatility_impact=0.0,
            liquidity_impact=0.0,
            prediction_accuracy=0.5,
            historical_precedence=0.0
        )
    
    def _create_default_strategy_performance(self) -> StrategyPerformance:
        """기본 전략 성과 생성"""
        return StrategyPerformance(
            strategy_name="Default_Strategy",
            roi=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.5,
            var_95=0.02,
            expected_shortfall=0.03,
            beta=1.0,
            alpha=0.0,
            last_7_days_return=0.0,
            last_30_days_return=0.0,
            ytd_return=0.0,
            current_positions={},
            exposure=0.0,
            leverage=1.0,
            updated_at=datetime.now()
        )
    
    def _create_default_anomaly_detection(self) -> AnomalyDetection:
        """기본 이상 탐지 생성"""
        return AnomalyDetection(
            anomaly_flag=False,
            anomaly_score=0.0,
            anomaly_type="normal",
            severity="low",
            event_tag=None,
            related_events=[],
            statistical_significance=0.0,
            historical_comparison=0.0,
            deviation_magnitude=0.0,
            recommended_action="monitor",
            confidence=0.5,
            detected_at=datetime.now()
        )
    
    def _create_default_refined_features(self, start_time: float) -> RefinedFeatureSet:
        """기본 정제 피처 세트 생성"""
        return RefinedFeatureSet(
            fusion_score=0.0,
            ml_refined_prediction=self._create_default_prediction(),
            event_impact=self._create_default_event_impact(),
            strategy_performance=self._create_default_strategy_performance(),
            anomaly_detection=self._create_default_anomaly_detection(),
            advanced_features=AdvancedFeatures(
                multimodal_sentiment={},
                temporal_features={},
                network_features={},
                risk_features={},
                emotional_ai={},
                prediction_meta={}
            ),
            timeseries_features={},
            network_graph_features={},
            meta_learning_features={},
            feature_quality_score=0.0,
            completeness_ratio=0.0,
            timestamp=datetime.now()
        )
    
    # 추가 메서드들 (간략화)
    def _analyze_market_spillover(self, event_type: str, impact_score: float) -> float:
        return min(impact_score * np.random.uniform(0.5, 1.5), 1.0)
    
    def _analyze_correlation_impact(self, event_type: str) -> float:
        correlation_impacts = {'regulatory': 0.8, 'earnings': 0.3, 'technical': 0.1, 'news': 0.5, 'social': 0.2}
        return correlation_impacts.get(event_type, 0.3)
    
    def _estimate_liquidity_impact(self, event_type: str, impact_score: float) -> float:
        return impact_score * np.random.uniform(0.3, 0.8)
    
    def _estimate_prediction_accuracy(self, event_type: str) -> float:
        accuracies = {'technical': 0.7, 'social': 0.6, 'news': 0.75, 'regulatory': 0.8, 'earnings': 0.85}
        return accuracies.get(event_type, 0.7)
    
    def _find_historical_precedence(self, text: str, event_type: str) -> float:
        return np.random.uniform(0.3, 0.9)  # 모의값
    
    def _detect_sentiment_anomaly(self, multimodal_result: MultiModalSentiment) -> Dict:
        extreme_threshold = 0.8
        is_extreme = abs(multimodal_result.text_sentiment) > extreme_threshold
        return {
            'is_anomaly': is_extreme,
            'score': abs(multimodal_result.text_sentiment) if is_extreme else 0.0
        }
    
    def _detect_price_anomaly(self, price_data: Dict) -> Dict:
        if 'prices' not in price_data or len(price_data['prices']) < 20:
            return {'is_anomaly': False, 'score': 0.0}
        
        prices = price_data['prices'][-20:]
        returns = np.diff(prices) / prices[:-1]
        z_scores = np.abs((returns - np.mean(returns)) / (np.std(returns) + 1e-8))
        max_z = np.max(z_scores)
        
        return {
            'is_anomaly': max_z > self.anomaly_thresholds['price_anomaly'],
            'score': min(max_z / 5.0, 1.0)
        }
    
    def _detect_volume_anomaly(self, volume_data: Dict) -> Dict:
        if 'volumes' not in volume_data or len(volume_data['volumes']) < 20:
            return {'is_anomaly': False, 'score': 0.0}
        
        volumes = volume_data['volumes'][-20:]
        volume_changes = np.diff(volumes) / (np.array(volumes[:-1]) + 1e-8)
        z_scores = np.abs((volume_changes - np.mean(volume_changes)) / (np.std(volume_changes) + 1e-8))
        max_z = np.max(z_scores)
        
        return {
            'is_anomaly': max_z > self.anomaly_thresholds['volume_anomaly'],
            'score': min(max_z / 3.0, 1.0)
        }
    
    def _detect_correlation_anomaly(self, multimodal_result: MultiModalSentiment) -> Dict:
        # 감정과 다른 지표들 간의 상관관계 이상
        sentiment_network_corr = abs(multimodal_result.text_sentiment - multimodal_result.network_effect)
        is_anomaly = sentiment_network_corr > 0.7
        return {
            'is_anomaly': is_anomaly,
            'score': sentiment_network_corr if is_anomaly else 0.0
        }
    
    def _determine_anomaly_severity(self, score: float) -> str:
        if score > 0.8:
            return "critical"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _tag_anomaly_event(self, anomalies: List, multimodal_result: MultiModalSentiment) -> str:
        if multimodal_result.black_swan_probability > 0.05:
            return "black_swan"
        elif multimodal_result.panic_indicator > 0.6:
            return "panic_selling"
        elif multimodal_result.viral_score > 0.8:
            return "viral_event"
        else:
            return "market_anomaly"
    
    def _find_related_events(self, anomalies: List) -> List[str]:
        return ["event_1", "event_2"]  # 모의값
    
    def _compare_with_history(self, anomalies: List) -> float:
        return np.random.uniform(0.3, 0.9)  # 모의값
    
    def _recommend_action(self, severity: str, anomaly_type: str) -> str:
        if severity == "critical":
            return "alert"
        elif severity == "high":
            return "investigate"
        elif severity == "medium":
            return "monitor"
        else:
            return "log"
    
    def _calculate_feature_quality(self, advanced_features: AdvancedFeatures, 
                                 ml_prediction: MarketPrediction, 
                                 event_impact: EventImpactAnalysis) -> Tuple[float, float]:
        """피처 품질 및 완성도 계산"""
        quality_scores = []
        completeness_scores = []
        
        # 멀티모달 감정 품질
        modal_count = len([v for v in advanced_features.multimodal_sentiment.values() if v != 0.0])
        modal_quality = modal_count / 4.0  # 4개 모달리티
        quality_scores.append(modal_quality)
        completeness_scores.append(modal_count / 4.0)
        
        # 예측 신뢰도
        pred_quality = 1.0 - ml_prediction.prediction_uncertainty
        quality_scores.append(pred_quality)
        completeness_scores.append(1.0 if ml_prediction.model_ensemble_size > 0 else 0.0)
        
        # 이벤트 영향도 신뢰도
        event_quality = event_impact.prediction_accuracy
        quality_scores.append(event_quality)
        completeness_scores.append(1.0 if event_impact.impact_score > 0 else 0.5)
        
        # 피처 다양성
        feature_diversity = len([f for f in [
            advanced_features.temporal_features,
            advanced_features.network_features,
            advanced_features.risk_features,
            advanced_features.emotional_ai,
            advanced_features.prediction_meta
        ] if f])
        diversity_score = feature_diversity / 5.0
        quality_scores.append(diversity_score)
        completeness_scores.append(diversity_score)
        
        overall_quality = statistics.mean(quality_scores)
        overall_completeness = statistics.mean(completeness_scores)
        
        return overall_quality, overall_completeness
    
    def _update_analysis_history(self, refined_features: RefinedFeatureSet):
        """분석 히스토리 업데이트"""
        self.prediction_history.append(refined_features)
        
        # 이벤트 히스토리 업데이트
        if refined_features.event_impact.impact_score > 0.3:
            self.event_history.append(refined_features.event_impact)
        
        # 이상 탐지 히스토리
        if refined_features.anomaly_detection.anomaly_flag:
            self.anomaly_cache.append(refined_features.anomaly_detection)
    
    def _update_advanced_stats(self, processing_time: float):
        """고급 통계 업데이트"""
        self.advanced_stats["feature_extractions"] += 1
        
        # 처리 시간 평균 업데이트
        current_avg = self.advanced_stats.get("avg_processing_time", processing_time)
        count = self.advanced_stats["feature_extractions"]
        self.advanced_stats["avg_processing_time"] = (
            (current_avg * (count - 1) + processing_time) / count
        )
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """고급 통계 반환"""
        base_stats = self.get_fusion_stats()
        
        advanced_stats = {
            **base_stats,
            **self.advanced_stats,
            "prediction_history_size": len(self.prediction_history),
            "event_history_size": len(self.event_history),
            "anomaly_cache_size": len(self.anomaly_cache),
            "ml_models_available": len([m for m in self.ml_models.values() if m is not None]),
            "anomaly_detectors_available": len([d for d in self.anomaly_detectors.values() if d is not None])
        }
        
        return advanced_stats


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_advanced_fusion():
        """고급 융합 매니저 테스트"""
        
        # 고급 스코어러 초기화
        advanced_scorer = AdvancedKeywordScorer()
        
        # 고급 융합 매니저 초기화
        fusion_manager = AdvancedFusionManager(
            advanced_scorer=advanced_scorer,
            ml_refinement_enabled=True
        )
        
        print("=== 고급 융합 분석 테스트 ===\n")
        
        # 테스트 데이터
        test_text = "Breaking: Bitcoin ETF approval sparks massive institutional buying frenzy as price surges above $50,000"
        test_market_data = {
            "price_data": {
                "prices": [45000 + i * 50 + np.random.normal(0, 100) for i in range(200)]
            },
            "volume_data": {
                "volumes": [1000000 + i * 5000 + np.random.normal(0, 100000) for i in range(200)]
            }
        }
        test_social_data = {
            "twitter": {"likes": 5000, "retweets": 1200, "replies": 800},
            "reddit": {"upvotes": 850, "comments": 150},
            "retweet_velocity": 120.5,
            "cross_platform_spread": 0.85,
            "influencer_amplification": 2.3
        }
        
        # 고급 융합 분석 실행
        result = await fusion_manager.advanced_fusion_analysis(
            content_hash="test_hash_123",
            text=test_text,
            market_data=test_market_data,
            social_data=test_social_data
        )
        
        print(f"텍스트: {test_text}\n")
        
        print("=== 융합 점수 + ML 리파인 예측 ===")
        print(f"📊 융합 점수: {result.fusion_score:.3f}")
        print(f"🔮 ML 예측 방향: {result.ml_refined_prediction.direction}")
        print(f"🎯 예측 확률: {result.ml_refined_prediction.probability:.3f}")
        print(f"📈 변동성 예측: {result.ml_refined_prediction.volatility_forecast:.3f}")
        print(f"💪 신뢰도 레벨: {result.ml_refined_prediction.confidence_level.value}")
        
        print("\n=== 이벤트 영향도 + 시장 반응 ===")
        print(f"🏷️ 이벤트 타입: {result.event_impact.event_type}")
        print(f"💥 영향도 스코어: {result.event_impact.impact_score:.3f}")
        print(f"⏱️ 반응 지연 (분): {result.event_impact.lag_estimate:.1f}")
        print(f"⏳ 지속 시간 (시간): {result.event_impact.duration_estimate:.1f}")
        print(f"🌊 시장 파급효과: {result.event_impact.market_spillover:.3f}")
        
        print("\n=== 전략 실시간 성과 지표 ===")
        print(f"💰 ROI: {result.strategy_performance.roi:.3f}")
        print(f"📊 샤프 비율: {result.strategy_performance.sharpe_ratio:.3f}")
        print(f"📉 최대 낙폭: {result.strategy_performance.max_drawdown:.3f}")
        print(f"🎯 승률: {result.strategy_performance.win_rate:.3f}")
        print(f"⚠️ VaR 95%: {result.strategy_performance.var_95:.3f}")
        
        print("\n=== 이상 탐지 + 이벤트 태깅 ===")
        print(f"🚨 이상 플래그: {result.anomaly_detection.anomaly_flag}")
        print(f"📊 이상 점수: {result.anomaly_detection.anomaly_score:.3f}")
        print(f"🏷️ 이상 타입: {result.anomaly_detection.anomaly_type}")
        print(f"⚡ 심각도: {result.anomaly_detection.severity}")
        print(f"🏷️ 이벤트 태그: {result.anomaly_detection.event_tag}")
        print(f"💡 권고 행동: {result.anomaly_detection.recommended_action}")
        
        print("\n=== 고급 피처 품질 ===")
        print(f"✨ 피처 품질 점수: {result.feature_quality_score:.3f}")
        print(f"📋 완성도 비율: {result.completeness_ratio:.3f}")
        print(f"🔧 피처 버전: {result.feature_version}")
        
        print("\n=== 네트워크 그래프 피처 (샘플) ===")
        for key, value in list(result.network_graph_features.items())[:5]:
            print(f"  {key}: {value:.3f}")
        
        print("\n=== 메타 학습 피처 (샘플) ===")
        for key, value in list(result.meta_learning_features.items())[:5]:
            print(f"  {key}: {value:.3f}")
        
        print("\n=== 시계열 피처 (크기) ===")
        for key, value in result.timeseries_features.items():
            print(f"  {key}: {len(value)} 포인트")
        
        # 통계
        print("\n=== 고급 융합 매니저 통계 ===")
        stats = fusion_manager.get_advanced_statistics()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
    
    # 테스트 실행
    asyncio.run(test_advanced_fusion())