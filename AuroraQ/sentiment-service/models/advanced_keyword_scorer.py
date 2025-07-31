#!/usr/bin/env python3
"""
Advanced Multi-Modal Keyword Scorer
멀티모달 감정 분석 및 고급 피처 엔지니어링
"""

import re
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import statistics

from .keyword_scorer import KeywordScorer, KeywordScore, SentimentDirection

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """시장 국면"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class EmotionalState(Enum):
    """감정 상태"""
    FEAR = "fear"
    GREED = "greed"
    EUPHORIA = "euphoria"
    PANIC = "panic"
    OPTIMISM = "optimism"
    PESSIMISM = "pessimism"
    UNCERTAINTY = "uncertainty"
    CONFIDENCE = "confidence"

@dataclass
class MultiModalSentiment:
    """멀티모달 감정 분석 결과"""
    # 기본 감정 스코어
    text_sentiment: float
    price_action_sentiment: float
    volume_sentiment: float
    social_engagement: float
    
    # 고급 피처
    emotional_state: EmotionalState
    market_regime: MarketRegime
    viral_score: float
    network_effect: float
    
    # 시계열 피처
    momentum_1h: float
    momentum_4h: float
    momentum_24h: float
    volatility_regime: str
    
    # 위험 지표
    black_swan_probability: float
    tail_risk: float
    herding_behavior: float
    panic_indicator: float
    
    # 메타 피처
    confidence_calibration: float
    prediction_stability: float
    ensemble_agreement: float
    
    # 기본 정보
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdvancedFeatures:
    """고급 피처 세트"""
    # 멀티모달 감정
    multimodal_sentiment: Dict[str, float]
    
    # 시계열 피처
    temporal_features: Dict[str, float]
    
    # 네트워크 효과
    network_features: Dict[str, float]
    
    # 위험 지표
    risk_features: Dict[str, float]
    
    # 감정 지능
    emotional_ai: Dict[str, float]
    
    # 예측 메타 피처
    prediction_meta: Dict[str, float]

class AdvancedKeywordScorer(KeywordScorer):
    """고급 멀티모달 키워드 스코어러"""
    
    def __init__(self):
        """초기화"""
        super().__init__()
        self._initialize_advanced_dictionaries()
        self._initialize_models()
        self.analysis_history = []
        self.regime_detector = MarketRegimeDetector()
        self.emotional_analyzer = EmotionalIntelligence()
        logger.info("AdvancedKeywordScorer initialized successfully")
    
    def _initialize_advanced_dictionaries(self):
        """고급 키워드 사전 초기화"""
        
        # 감정 상태 키워드
        self.emotional_keywords = {
            # 공포 관련
            "fear": -0.8, "scared": -0.7, "terrified": -0.9, "worried": -0.5,
            "anxiety": -0.6, "panic": -0.9, "nervous": -0.5, "concerned": -0.4,
            
            # 탐욕 관련
            "greed": 0.7, "greedy": 0.6, "fomo": 0.8, "euphoric": 0.9,
            "excited": 0.6, "pumped": 0.8, "moon": 0.9, "lambo": 0.8,
            
            # 불확실성
            "uncertain": -0.4, "confused": -0.3, "mixed": -0.2, "unclear": -0.3,
            "volatile": -0.4, "unstable": -0.5, "chaotic": -0.7,
            
            # 신뢰
            "confident": 0.7, "certain": 0.6, "sure": 0.5, "convinced": 0.8,
            "bullish": 0.7, "bearish": -0.7, "optimistic": 0.6, "pessimistic": -0.6
        }
        
        # 네트워크 효과 키워드
        self.network_keywords = {
            # 바이럴 확산
            "viral": 0.8, "trending": 0.7, "popular": 0.6, "buzz": 0.7,
            "hype": 0.6, "attention": 0.5, "spotlight": 0.6, "mainstream": 0.7,
            
            # 영향력
            "influencer": 0.6, "celebrity": 0.7, "endorsement": 0.8, "backing": 0.7,
            "support": 0.5, "adoption": 0.8, "partnership": 0.7, "collaboration": 0.6,
            
            # 커뮤니티
            "community": 0.5, "followers": 0.4, "subscribers": 0.4, "members": 0.3,
            "engagement": 0.6, "discussion": 0.4, "debate": 0.3, "conversation": 0.4
        }
        
        # 시장 미시구조 키워드
        self.microstructure_keywords = {
            # 유동성
            "liquidity": 0.4, "liquid": 0.3, "illiquid": -0.6, "thin": -0.5,
            "depth": 0.4, "spread": -0.3, "slippage": -0.5, "impact": -0.3,
            
            # 주문 플로우
            "buying": 0.5, "selling": -0.5, "accumulation": 0.6, "distribution": -0.6,
            "volume": 0.3, "participation": 0.4, "activity": 0.3, "flow": 0.2,
            
            # 시장 구조
            "maker": 0.2, "taker": -0.1, "arbitrage": 0.3, "efficiency": 0.4,
            "friction": -0.3, "costs": -0.2, "fees": -0.2, "latency": -0.3
        }
    
    def _initialize_models(self):
        """고급 모델 초기화"""
        # 시계열 분석 설정
        self.lookback_periods = {
            "short": 60,    # 1시간
            "medium": 240,  # 4시간
            "long": 1440    # 24시간
        }
        
        # 위험 임계값
        self.risk_thresholds = {
            "black_swan": 0.01,
            "tail_risk": 0.05,
            "volatility": 0.03,
            "correlation": 0.8
        }
        
        # 감정 캘리브레이션 파라미터
        self.emotion_params = {
            "fear_threshold": -0.6,
            "greed_threshold": 0.7,
            "uncertainty_threshold": 0.4,
            "confidence_threshold": 0.8
        }
    
    def analyze_advanced(self, 
                        text: str, 
                        price_data: Optional[Dict] = None,
                        volume_data: Optional[Dict] = None,
                        social_data: Optional[Dict] = None) -> MultiModalSentiment:
        """고급 멀티모달 감정 분석"""
        start_time = time.time()
        
        try:
            # 1. 기본 텍스트 감정 분석
            basic_result = self.analyze(text)
            text_sentiment = basic_result.score
            
            # 2. 가격 행동 감정
            price_sentiment = self._analyze_price_action(price_data) if price_data else 0.0
            
            # 3. 거래량 감정
            volume_sentiment = self._analyze_volume_sentiment(volume_data) if volume_data else 0.0
            
            # 4. 소셜 참여도
            social_engagement = self._analyze_social_engagement(social_data) if social_data else 0.0
            
            # 5. 감정 상태 분류
            emotional_state = self._classify_emotional_state(text, text_sentiment)
            
            # 6. 시장 국면 감지
            market_regime = self._detect_market_regime(price_data, volume_data)
            
            # 7. 바이럴 점수
            viral_score = self._calculate_viral_score(text, social_data)
            
            # 8. 네트워크 효과
            network_effect = self._calculate_network_effect(text, social_data)
            
            # 9. 시계열 모멘텀
            momentum_1h = self._calculate_momentum(price_data, "1h") if price_data else 0.0
            momentum_4h = self._calculate_momentum(price_data, "4h") if price_data else 0.0
            momentum_24h = self._calculate_momentum(price_data, "24h") if price_data else 0.0
            
            # 10. 변동성 국면
            volatility_regime = self._detect_volatility_regime(price_data) if price_data else "normal"
            
            # 11. 위험 지표들
            black_swan_prob = self._calculate_black_swan_probability(price_data, text_sentiment)
            tail_risk = self._calculate_tail_risk(price_data)
            herding_behavior = self._detect_herding_behavior(text, social_data)
            panic_indicator = self._calculate_panic_indicator(text_sentiment, volume_sentiment)
            
            # 12. 메타 피처들
            confidence_calibration = self._calibrate_confidence(basic_result.confidence, text_sentiment)
            prediction_stability = self._calculate_prediction_stability()
            ensemble_agreement = self._calculate_ensemble_agreement(text_sentiment, price_sentiment, volume_sentiment)
            
            # 결과 생성
            result = MultiModalSentiment(
                text_sentiment=text_sentiment,
                price_action_sentiment=price_sentiment,
                volume_sentiment=volume_sentiment,
                social_engagement=social_engagement,
                emotional_state=emotional_state,
                market_regime=market_regime,
                viral_score=viral_score,
                network_effect=network_effect,
                momentum_1h=momentum_1h,
                momentum_4h=momentum_4h,
                momentum_24h=momentum_24h,
                volatility_regime=volatility_regime,
                black_swan_probability=black_swan_prob,
                tail_risk=tail_risk,
                herding_behavior=herding_behavior,
                panic_indicator=panic_indicator,
                confidence_calibration=confidence_calibration,
                prediction_stability=prediction_stability,
                ensemble_agreement=ensemble_agreement,
                processing_time=time.time() - start_time,
                timestamp=datetime.now(),
                metadata={
                    "basic_result": basic_result.__dict__,
                    "input_sources": {
                        "has_price": price_data is not None,
                        "has_volume": volume_data is not None,
                        "has_social": social_data is not None
                    }
                }
            )
            
            # 히스토리에 추가
            self.analysis_history.append(result)
            if len(self.analysis_history) > 1000:  # 최대 1000개 유지
                self.analysis_history.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}", exc_info=True)
            # 기본값 반환
            return self._create_default_result(start_time)
    
    def _analyze_price_action(self, price_data: Dict) -> float:
        """가격 행동 감정 분석"""
        try:
            if not price_data or 'prices' not in price_data:
                return 0.0
            
            prices = price_data['prices'][-50:]  # 최근 50개 데이터포인트
            if len(prices) < 2:
                return 0.0
            
            # 가격 변화율
            returns = np.diff(prices) / prices[:-1]
            
            # 트렌드 강도
            trend = np.mean(returns)
            
            # 변동성
            volatility = np.std(returns)
            
            # 모멘텀 (최근 vs 이전)
            recent_momentum = np.mean(returns[-10:]) if len(returns) >= 10 else trend
            
            # 감정 점수 계산 (-1 ~ 1)
            sentiment = np.tanh(trend * 100) * (1 - min(volatility * 10, 0.5))
            sentiment += np.tanh(recent_momentum * 50) * 0.3
            
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            logger.warning(f"Price action analysis failed: {e}")
            return 0.0
    
    def _analyze_volume_sentiment(self, volume_data: Dict) -> float:
        """거래량 감정 분석"""
        try:
            if not volume_data or 'volumes' not in volume_data:
                return 0.0
            
            volumes = volume_data['volumes'][-50:]
            if len(volumes) < 2:
                return 0.0
            
            # 거래량 변화율
            volume_changes = np.diff(volumes) / (volumes[:-1] + 1e-8)
            
            # 평균 거래량 대비 현재 거래량
            avg_volume = np.mean(volumes[:-1])
            current_volume = volumes[-1]
            volume_ratio = current_volume / (avg_volume + 1e-8)
            
            # 거래량 모멘텀
            volume_momentum = np.mean(volume_changes[-10:]) if len(volume_changes) >= 10 else 0
            
            # 감정 점수 계산
            sentiment = 0.0
            
            # 높은 거래량 = 관심 증가 = 긍정적
            if volume_ratio > 1.5:
                sentiment += 0.3
            elif volume_ratio > 1.2:
                sentiment += 0.1
            elif volume_ratio < 0.7:
                sentiment -= 0.2
            
            # 거래량 증가 트렌드 = 긍정적
            if volume_momentum > 0.1:
                sentiment += 0.4
            elif volume_momentum < -0.1:
                sentiment -= 0.3
            
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            logger.warning(f"Volume sentiment analysis failed: {e}")
            return 0.0
    
    def _analyze_social_engagement(self, social_data: Dict) -> float:
        """소셜 참여도 분석"""
        try:
            if not social_data:
                return 0.0
            
            engagement = 0.0
            total_weight = 0.0
            
            # 트위터 데이터
            if 'twitter' in social_data:
                twitter = social_data['twitter']
                likes = twitter.get('likes', 0)
                retweets = twitter.get('retweets', 0)
                replies = twitter.get('replies', 0)
                
                # 정규화된 참여도 (로그 스케일)
                twitter_engagement = np.log1p(likes + retweets * 2 + replies * 1.5) / 10
                engagement += twitter_engagement * 0.4
                total_weight += 0.4
            
            # 레딧 데이터
            if 'reddit' in social_data:
                reddit = social_data['reddit']
                upvotes = reddit.get('upvotes', 0)
                comments = reddit.get('comments', 0)
                
                reddit_engagement = np.log1p(upvotes + comments * 2) / 8
                engagement += reddit_engagement * 0.3
                total_weight += 0.3
            
            # 기타 플랫폼
            if 'other_platforms' in social_data:
                other = social_data['other_platforms']
                total_mentions = other.get('mentions', 0)
                
                other_engagement = np.log1p(total_mentions) / 5
                engagement += other_engagement * 0.3
                total_weight += 0.3
            
            if total_weight > 0:
                engagement = engagement / total_weight
            
            return max(0.0, min(1.0, engagement))
            
        except Exception as e:
            logger.warning(f"Social engagement analysis failed: {e}")
            return 0.0
    
    def _classify_emotional_state(self, text: str, sentiment: float) -> EmotionalState:
        """감정 상태 분류"""
        try:
            processed_text = self._preprocess_text(text).lower()
            
            # 키워드 기반 감정 감지
            fear_words = ["fear", "scared", "panic", "worried", "anxiety", "crash", "dump"]
            greed_words = ["moon", "pump", "lambo", "fomo", "greed", "euphoric"]
            uncertainty_words = ["uncertain", "confused", "mixed", "volatile", "unclear"]
            confidence_words = ["confident", "sure", "bullish", "optimistic", "strong"]
            
            fear_count = sum(1 for word in fear_words if word in processed_text)
            greed_count = sum(1 for word in greed_words if word in processed_text)
            uncertainty_count = sum(1 for word in uncertainty_words if word in processed_text)
            confidence_count = sum(1 for word in confidence_words if word in processed_text)
            
            # 감정 점수와 키워드를 결합하여 분류
            if sentiment < -0.6 or fear_count >= 2:
                return EmotionalState.FEAR if sentiment < -0.8 else EmotionalState.PESSIMISM
            elif sentiment > 0.7 or greed_count >= 2:
                return EmotionalState.GREED if sentiment > 0.8 else EmotionalState.OPTIMISM
            elif uncertainty_count >= 2 or abs(sentiment) < 0.2:
                return EmotionalState.UNCERTAINTY
            elif confidence_count >= 2 or sentiment > 0.5:
                return EmotionalState.CONFIDENCE
            else:
                return EmotionalState.UNCERTAINTY
                
        except Exception as e:
            logger.warning(f"Emotional state classification failed: {e}")
            return EmotionalState.UNCERTAINTY
    
    def _detect_market_regime(self, price_data: Optional[Dict], volume_data: Optional[Dict]) -> MarketRegime:
        """시장 국면 감지"""
        try:
            if not price_data or 'prices' not in price_data:
                return MarketRegime.SIDEWAYS
            
            prices = price_data['prices'][-100:]  # 최근 100개 데이터포인트
            if len(prices) < 20:
                return MarketRegime.SIDEWAYS
            
            # 가격 변화율
            returns = np.diff(prices) / prices[:-1]
            
            # 변동성 계산
            volatility = np.std(returns) * np.sqrt(len(returns))
            
            # 트렌드 강도
            trend = np.mean(returns)
            
            # 국면 분류
            if volatility > 0.05:  # 5% 이상 변동성
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.01:  # 1% 미만 변동성
                return MarketRegime.LOW_VOLATILITY
            elif trend > 0.02:  # 2% 이상 상승 트렌드
                return MarketRegime.BULL_MARKET
            elif trend < -0.02:  # 2% 이상 하락 트렌드
                return MarketRegime.BEAR_MARKET
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            logger.warning(f"Market regime detection failed: {e}")
            return MarketRegime.SIDEWAYS
    
    def _calculate_viral_score(self, text: str, social_data: Optional[Dict]) -> float:
        """바이럴 점수 계산"""
        try:
            viral_score = 0.0
            
            # 텍스트 기반 바이럴 키워드
            processed_text = self._preprocess_text(text).lower()
            viral_keywords = ["viral", "trending", "buzz", "hype", "attention", "mainstream"]
            viral_count = sum(1 for word in viral_keywords if word in processed_text)
            viral_score += min(viral_count * 0.2, 0.5)
            
            # 소셜 데이터 기반
            if social_data:
                # 확산 속도 (retweet velocity)
                if 'retweet_velocity' in social_data:
                    velocity = social_data['retweet_velocity']
                    viral_score += min(np.log1p(velocity) / 10, 0.3)
                
                # 플랫폼 간 확산
                if 'cross_platform_spread' in social_data:
                    spread = social_data['cross_platform_spread']
                    viral_score += spread * 0.2
            
            return max(0.0, min(1.0, viral_score))
            
        except Exception as e:
            logger.warning(f"Viral score calculation failed: {e}")
            return 0.0
    
    def _calculate_network_effect(self, text: str, social_data: Optional[Dict]) -> float:
        """네트워크 효과 계산"""
        try:
            network_effect = 0.0
            
            # 네트워크 키워드 매칭
            processed_text = self._preprocess_text(text).lower()
            words = processed_text.split()
            
            for word in words:
                if word in self.network_keywords:
                    network_effect += self.network_keywords[word] * 0.1
            
            # 소셜 네트워크 지표
            if social_data:
                # 인플루언서 증폭도
                if 'influencer_amplification' in social_data:
                    amplification = social_data['influencer_amplification']
                    network_effect += min(amplification * 0.3, 0.4)
                
                # 에코챔버 효과
                if 'echo_chamber_score' in social_data:
                    echo = social_data['echo_chamber_score']
                    network_effect += echo * 0.2
            
            return max(0.0, min(1.0, network_effect))
            
        except Exception as e:
            logger.warning(f"Network effect calculation failed: {e}")
            return 0.0
    
    def _calculate_momentum(self, price_data: Dict, period: str) -> float:
        """모멘텀 계산"""
        try:
            prices = price_data['prices']
            
            # 기간별 데이터포인트 수
            periods = {"1h": -12, "4h": -48, "24h": -144}  # 5분 간격 기준
            lookback = periods.get(period, -24)
            
            if len(prices) < abs(lookback):
                return 0.0
            
            recent_prices = prices[lookback:]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            return max(-1.0, min(1.0, momentum * 10))  # 스케일링
            
        except Exception as e:
            logger.warning(f"Momentum calculation failed: {e}")
            return 0.0
    
    def _detect_volatility_regime(self, price_data: Dict) -> str:
        """변동성 국면 감지"""
        try:
            prices = price_data['prices'][-50:]
            if len(prices) < 10:
                return "normal"
            
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            if volatility > 0.05:
                return "high"
            elif volatility < 0.01:
                return "low"
            else:
                return "normal"
                
        except Exception as e:
            logger.warning(f"Volatility regime detection failed: {e}")
            return "normal"
    
    def _calculate_black_swan_probability(self, price_data: Optional[Dict], sentiment: float) -> float:
        """블랙 스완 확률 계산"""
        try:
            base_probability = 0.01  # 1% 기본 확률
            
            # 극단적 감정은 블랙 스완 확률 증가
            if abs(sentiment) > 0.8:
                base_probability *= 2
            
            # 가격 데이터가 있는 경우 추가 분석
            if price_data and 'prices' in price_data:
                prices = price_data['prices'][-50:]
                if len(prices) > 10:
                    returns = np.diff(prices) / prices[:-1]
                    # 극단적 수익률 빈도
                    extreme_returns = np.sum(np.abs(returns) > 3 * np.std(returns))
                    if extreme_returns > 0:
                        base_probability *= (1 + extreme_returns * 0.5)
            
            return min(base_probability, 0.1)  # 최대 10%
            
        except Exception as e:
            logger.warning(f"Black swan probability calculation failed: {e}")
            return 0.01
    
    def _calculate_tail_risk(self, price_data: Optional[Dict]) -> float:
        """테일 리스크 계산"""
        try:
            if not price_data or 'prices' not in price_data:
                return 0.05  # 기본값
            
            prices = price_data['prices'][-100:]
            if len(prices) < 20:
                return 0.05
            
            returns = np.diff(prices) / prices[:-1]
            
            # VaR (Value at Risk) 95% 신뢰구간
            var_95 = np.percentile(returns, 5)
            
            # 테일 리스크는 VaR의 절댓값을 정규화
            tail_risk = abs(var_95) * 10  # 스케일링
            
            return max(0.0, min(1.0, tail_risk))
            
        except Exception as e:
            logger.warning(f"Tail risk calculation failed: {e}")
            return 0.05
    
    def _detect_herding_behavior(self, text: str, social_data: Optional[Dict]) -> float:
        """군집행동 감지"""
        try:
            herding_score = 0.0
            
            # 텍스트 기반 군집행동 키워드
            processed_text = self._preprocess_text(text).lower()
            herding_words = ["everyone", "all", "consensus", "majority", "crowd", "follow"]
            herding_count = sum(1 for word in herding_words if word in processed_text)
            herding_score += min(herding_count * 0.2, 0.4)
            
            # 소셜 데이터 기반
            if social_data:
                # 모든 플랫폼에서 같은 방향 감정
                if 'sentiment_agreement' in social_data:
                    agreement = social_data['sentiment_agreement']
                    herding_score += agreement * 0.6
            
            return max(0.0, min(1.0, herding_score))
            
        except Exception as e:
            logger.warning(f"Herding behavior detection failed: {e}")
            return 0.0
    
    def _calculate_panic_indicator(self, text_sentiment: float, volume_sentiment: float) -> float:
        """패닉 지표 계산"""
        try:
            # 부정적 감정 + 높은 거래량 = 패닉
            if text_sentiment < -0.6 and volume_sentiment > 0.3:
                panic = abs(text_sentiment) * volume_sentiment
                return max(0.0, min(1.0, panic))
            return 0.0
            
        except Exception as e:
            logger.warning(f"Panic indicator calculation failed: {e}")
            return 0.0
    
    def _calibrate_confidence(self, base_confidence: float, sentiment: float) -> float:
        """신뢰도 캘리브레이션"""
        try:
            # 극단적 감정일수록 신뢰도 높임
            extreme_factor = abs(sentiment) * 0.3
            calibrated = base_confidence + extreme_factor
            return max(0.0, min(1.0, calibrated))
            
        except Exception as e:
            logger.warning(f"Confidence calibration failed: {e}")
            return base_confidence
    
    def _calculate_prediction_stability(self) -> float:
        """예측 안정성 계산"""
        try:
            if len(self.analysis_history) < 5:
                return 0.5  # 기본값
            
            # 최근 5개 분석의 감정 점수 분산
            recent_sentiments = [h.text_sentiment for h in self.analysis_history[-5:]]
            stability = 1.0 - min(np.std(recent_sentiments), 1.0)
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.warning(f"Prediction stability calculation failed: {e}")
            return 0.5
    
    def _calculate_ensemble_agreement(self, text_sentiment: float, price_sentiment: float, volume_sentiment: float) -> float:
        """앙상블 합의도 계산"""
        try:
            sentiments = [text_sentiment, price_sentiment, volume_sentiment]
            sentiments = [s for s in sentiments if s != 0.0]  # 0이 아닌 값만
            
            if len(sentiments) < 2:
                return 0.5
            
            # 모든 감정 점수가 같은 방향인지 확인
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            neutral_count = len(sentiments) - positive_count - negative_count
            
            max_agreement = max(positive_count, negative_count, neutral_count)
            agreement = max_agreement / len(sentiments)
            
            return max(0.0, min(1.0, agreement))
            
        except Exception as e:
            logger.warning(f"Ensemble agreement calculation failed: {e}")
            return 0.5
    
    def _create_default_result(self, start_time: float) -> MultiModalSentiment:
        """기본 결과 생성"""
        return MultiModalSentiment(
            text_sentiment=0.0,
            price_action_sentiment=0.0,
            volume_sentiment=0.0,
            social_engagement=0.0,
            emotional_state=EmotionalState.UNCERTAINTY,
            market_regime=MarketRegime.SIDEWAYS,
            viral_score=0.0,
            network_effect=0.0,
            momentum_1h=0.0,
            momentum_4h=0.0,
            momentum_24h=0.0,
            volatility_regime="normal",
            black_swan_probability=0.01,
            tail_risk=0.05,
            herding_behavior=0.0,
            panic_indicator=0.0,
            confidence_calibration=0.5,
            prediction_stability=0.5,
            ensemble_agreement=0.5,
            processing_time=time.time() - start_time,
            timestamp=datetime.now(),
            metadata={"error": "Analysis failed, using defaults"}
        )
    
    def extract_advanced_features(self, result: MultiModalSentiment) -> AdvancedFeatures:
        """고급 피처 추출"""
        return AdvancedFeatures(
            multimodal_sentiment={
                "text_sentiment": result.text_sentiment,
                "price_action_sentiment": result.price_action_sentiment,
                "volume_sentiment": result.volume_sentiment,
                "social_engagement": result.social_engagement
            },
            temporal_features={
                "momentum_1h": result.momentum_1h,
                "momentum_4h": result.momentum_4h,
                "momentum_24h": result.momentum_24h,
                "volatility_regime_score": {"high": 1.0, "normal": 0.5, "low": 0.0}[result.volatility_regime]
            },
            network_features={
                "viral_score": result.viral_score,
                "network_effect": result.network_effect,
                "herding_behavior": result.herding_behavior
            },
            risk_features={
                "black_swan_probability": result.black_swan_probability,
                "tail_risk": result.tail_risk,
                "panic_indicator": result.panic_indicator
            },
            emotional_ai={
                "emotional_state_score": {
                    EmotionalState.FEAR: -0.8,
                    EmotionalState.PANIC: -0.9,
                    EmotionalState.PESSIMISM: -0.6,
                    EmotionalState.UNCERTAINTY: 0.0,
                    EmotionalState.CONFIDENCE: 0.6,
                    EmotionalState.OPTIMISM: 0.7,
                    EmotionalState.GREED: 0.8,
                    EmotionalState.EUPHORIA: 0.9
                }[result.emotional_state]
            },
            prediction_meta={
                "confidence_calibration": result.confidence_calibration,
                "prediction_stability": result.prediction_stability,
                "ensemble_agreement": result.ensemble_agreement
            }
        )

class MarketRegimeDetector:
    """시장 국면 감지기"""
    
    def __init__(self):
        self.regime_history = []
    
    def detect_regime(self, data: Dict) -> MarketRegime:
        # 구현 예정
        return MarketRegime.SIDEWAYS

class EmotionalIntelligence:
    """감정 지능 분석기"""
    
    def __init__(self):
        self.emotion_patterns = {}
    
    def analyze_emotion_evolution(self, history: List) -> Dict:
        # 구현 예정
        return {}


# 테스트 코드
if __name__ == "__main__":
    import json
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 고급 스코어러 초기화
    scorer = AdvancedKeywordScorer()
    
    # 테스트 데이터
    test_text = "Bitcoin surges amid institutional adoption and viral social media buzz"
    test_price_data = {
        "prices": [40000 + i * 100 + np.random.normal(0, 50) for i in range(100)]
    }
    test_volume_data = {
        "volumes": [1000000 + i * 10000 + np.random.normal(0, 50000) for i in range(100)]
    }
    test_social_data = {
        "twitter": {"likes": 1500, "retweets": 300, "replies": 150},
        "reddit": {"upvotes": 250, "comments": 45},
        "retweet_velocity": 50.5,
        "cross_platform_spread": 0.7
    }
    
    print("=== 고급 멀티모달 감정 분석 테스트 ===\n")
    
    # 고급 분석 실행
    result = scorer.analyze_advanced(
        text=test_text,
        price_data=test_price_data,
        volume_data=test_volume_data,
        social_data=test_social_data
    )
    
    print(f"텍스트: {test_text}")
    print(f"처리시간: {result.processing_time*1000:.1f}ms\n")
    
    print("=== 멀티모달 감정 스코어 ===")
    print(f"📝 텍스트 감정: {result.text_sentiment:.3f}")
    print(f"📈 가격 행동 감정: {result.price_action_sentiment:.3f}")
    print(f"📊 거래량 감정: {result.volume_sentiment:.3f}")
    print(f"💬 소셜 참여도: {result.social_engagement:.3f}")
    
    print("\n=== 고급 지표 ===")
    print(f"😨 감정 상태: {result.emotional_state.value}")
    print(f"🎯 시장 국면: {result.market_regime.value}")
    print(f"🔥 바이럴 점수: {result.viral_score:.3f}")
    print(f"🌐 네트워크 효과: {result.network_effect:.3f}")
    
    print("\n=== 시계열 피처 ===")
    print(f"⚡ 1시간 모멘텀: {result.momentum_1h:.3f}")
    print(f"⚡ 4시간 모멘텀: {result.momentum_4h:.3f}")
    print(f"⚡ 24시간 모멘텀: {result.momentum_24h:.3f}")
    print(f"📊 변동성 국면: {result.volatility_regime}")
    
    print("\n=== 위험 지표 ===")
    print(f"🦢 블랙 스완 확률: {result.black_swan_probability:.3f}")
    print(f"📉 테일 리스크: {result.tail_risk:.3f}")
    print(f"🐑 군집행동: {result.herding_behavior:.3f}")
    print(f"😱 패닉 지표: {result.panic_indicator:.3f}")
    
    print("\n=== 메타 피처 ===")
    print(f"🎯 신뢰도 캘리브레이션: {result.confidence_calibration:.3f}")
    print(f"📏 예측 안정성: {result.prediction_stability:.3f}")
    print(f"🤝 앙상블 합의도: {result.ensemble_agreement:.3f}")
    
    # 고급 피처 추출
    advanced_features = scorer.extract_advanced_features(result)
    print("\n=== 추출된 고급 피처 세트 ===")
    for category, features in advanced_features.__dict__.items():
        print(f"\n{category.upper()}:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")