"""
시그널 처리 계층 (Signal Processing Layer)
- AdaptiveEntrySystem: 변동성/레짐/감정 기반 동적 임계값
- ProbabilisticEntry: 신뢰도 기반 포지션 크기 결정
- TimeBasedFilter: 세션별 노이즈 필터링
- MultiTimeframeConfirmation: 상위 타임프레임 신호 가중치
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, time
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """시장 레짐 정의"""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    CRISIS = "crisis"


class TradingSession(Enum):
    """거래 세션 정의"""
    ASIA = "asia"          # 00:00 - 08:00 UTC
    EUROPE = "europe"      # 08:00 - 16:00 UTC
    US = "us"              # 16:00 - 24:00 UTC


@dataclass
class SignalResult:
    """시그널 결과"""
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 ~ 1.0
    position_size: float  # 0.0 ~ 1.0
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = None


class AdaptiveEntrySystem:
    """적응형 진입 시스템"""
    
    def __init__(self):
        """초기화"""
        # 레짐별 임계값 조정 계수
        self.regime_thresholds = {
            MarketRegime.BULL: {"buy": 0.9, "sell": 1.1},
            MarketRegime.BEAR: {"buy": 1.1, "sell": 0.9},
            MarketRegime.NEUTRAL: {"buy": 1.0, "sell": 1.0},
            MarketRegime.VOLATILE: {"buy": 1.2, "sell": 1.2},
            MarketRegime.CRISIS: {"buy": 1.5, "sell": 1.5}
        }
        
        # 감정 점수별 조정 계수
        self.sentiment_adjustments = {
            "extreme_fear": {"buy": 0.8, "sell": 1.2},      # < 0.2
            "fear": {"buy": 0.9, "sell": 1.1},              # 0.2 ~ 0.4
            "neutral": {"buy": 1.0, "sell": 1.0},           # 0.4 ~ 0.6
            "greed": {"buy": 1.1, "sell": 0.9},             # 0.6 ~ 0.8
            "extreme_greed": {"buy": 1.2, "sell": 0.8}      # > 0.8
        }
        
        # 변동성별 포지션 크기 조정
        self.volatility_position_scale = {
            "low": 1.2,      # < 1% daily
            "normal": 1.0,   # 1% ~ 3% daily
            "high": 0.7,     # 3% ~ 5% daily
            "extreme": 0.3   # > 5% daily
        }
    
    def detect_regime(self, price_data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Tuple[MarketRegime, float]:
        """시장 레짐 감지"""
        if len(price_data) < 50:
            return MarketRegime.NEUTRAL, 0.5
        
        try:
            # 입력 검증 - indicators가 None이거나 빈 딕셔너리일 경우 빈 딕셔너리로 초기화
            if not indicators:
                indicators = {}
            
            # 이동평균 기반 추세
            sma_20 = indicators.get("sma_20", price_data['close'].rolling(20).mean())
            sma_50 = indicators.get("sma_50", price_data['close'].rolling(50).mean())
            
            current_price = price_data['close'].iloc[-1]
            
            # 변동성
            volatility = price_data['close'].pct_change().rolling(20).std().iloc[-1]
            
            # ADX (추세 강도)
            adx = indicators.get("adx", pd.Series([25]))  # 기본값
            adx_value = adx.iloc[-1] if not adx.empty else 25
            
            # 레짐 판단
            confidence = 0.5
            
            if volatility > 0.05:  # 5% 이상 일일 변동성
                return MarketRegime.CRISIS, 0.9
            
            if volatility > 0.03:  # 3% 이상 일일 변동성
                return MarketRegime.VOLATILE, 0.7
            
            # 추세 판단
            if current_price > sma_20.iloc[-1] * 1.02 and sma_20.iloc[-1] > sma_50.iloc[-1]:
                confidence = min(0.9, 0.5 + adx_value / 50)
                return MarketRegime.BULL, confidence
            
            if current_price < sma_20.iloc[-1] * 0.98 and sma_20.iloc[-1] < sma_50.iloc[-1]:
                confidence = min(0.9, 0.5 + adx_value / 50)
                return MarketRegime.BEAR, confidence
            
            return MarketRegime.NEUTRAL, 0.6
            
        except Exception as e:
            logger.error(f"레짐 감지 오류: {e}")
            return MarketRegime.NEUTRAL, 0.5
    
    def get_sentiment_category(self, sentiment_score: float) -> str:
        """감정 점수 카테고리화"""
        if sentiment_score < 0.2:
            return "extreme_fear"
        elif sentiment_score < 0.4:
            return "fear"
        elif sentiment_score < 0.6:
            return "neutral"
        elif sentiment_score < 0.8:
            return "greed"
        else:
            return "extreme_greed"
    
    def get_volatility_category(self, volatility: float) -> str:
        """변동성 카테고리화"""
        if volatility < 0.01:
            return "low"
        elif volatility < 0.03:
            return "normal"
        elif volatility < 0.05:
            return "high"
        else:
            return "extreme"
    
    def adjust_thresholds(self, 
                         base_threshold: float,
                         signal_type: str,
                         regime: MarketRegime,
                         sentiment_score: float,
                         volatility: float) -> float:
        """임계값 동적 조정"""
        # 레짐 조정
        regime_adj = self.regime_thresholds[regime][signal_type.lower()]
        
        # 감정 조정
        sentiment_cat = self.get_sentiment_category(sentiment_score)
        sentiment_adj = self.sentiment_adjustments[sentiment_cat][signal_type.lower()]
        
        # 최종 임계값
        adjusted = base_threshold * regime_adj * sentiment_adj
        
        # 변동성이 높으면 추가 조정
        vol_cat = self.get_volatility_category(volatility)
        if vol_cat in ["high", "extreme"]:
            adjusted *= 1.2  # 더 보수적으로
        
        return adjusted


class ProbabilisticEntry:
    """확률적 진입 시스템"""
    
    def __init__(self):
        """초기화"""
        self.min_confidence = 0.6
        self.max_position_size = 1.0
    
    def calculate_position_size(self,
                              signal_strength: float,
                              confidence: float,
                              volatility: float,
                              sentiment_score: float) -> float:
        """신뢰도 기반 포지션 크기 계산"""
        # 기본 포지션 크기 (신호 강도와 신뢰도의 곱)
        base_size = signal_strength * confidence
        
        # 변동성 조정 (변동성이 높을수록 포지션 감소)
        vol_multiplier = 1.0
        if volatility > 0.03:
            vol_multiplier = 0.7
        elif volatility > 0.05:
            vol_multiplier = 0.3
        
        # 감정 극단값 조정
        sentiment_multiplier = 1.0
        if sentiment_score < 0.2 or sentiment_score > 0.8:
            sentiment_multiplier = 0.8
        
        # 최종 포지션 크기
        position_size = base_size * vol_multiplier * sentiment_multiplier
        
        # 범위 제한
        position_size = max(0.1, min(self.max_position_size, position_size))
        
        return round(position_size, 2)
    
    def calculate_confidence(self,
                           indicators: Dict[str, pd.Series],
                           regime_confidence: float,
                           signal_alignment: float) -> float:
        """종합 신뢰도 계산"""
        confidences = []
        
        # 입력 검증 - indicators가 None이거나 빈 딕셔너리일 경우 빈 딕셔너리로 초기화
        if not indicators:
            indicators = {}
        
        # 레짐 신뢰도
        confidences.append(regime_confidence)
        
        # 지표 정렬도
        confidences.append(signal_alignment)
        
        # RSI 신뢰도
        if "rsi" in indicators and indicators["rsi"] is not None and not indicators["rsi"].empty:
            try:
                rsi = indicators["rsi"].iloc[-1]
                if 30 < rsi < 70:
                    rsi_confidence = 0.5
                else:
                    rsi_confidence = 0.8
                confidences.append(rsi_confidence)
            except (IndexError, ValueError):
                # RSI 계산 실패 시 기본값 사용
                pass
        
        # 평균 신뢰도
        return np.mean(confidences)


class TimeBasedFilter:
    """시간대별 필터링"""
    
    def __init__(self):
        """초기화"""
        # 세션별 노이즈 레벨 (0.0 ~ 1.0)
        self.session_noise_levels = {
            TradingSession.ASIA: 0.3,
            TradingSession.EUROPE: 0.5,
            TradingSession.US: 0.7
        }
        
        # 세션별 활성 시간 (UTC)
        self.session_hours = {
            TradingSession.ASIA: (0, 8),
            TradingSession.EUROPE: (8, 16),
            TradingSession.US: (16, 24)
        }
    
    def get_current_session(self, timestamp: pd.Timestamp) -> TradingSession:
        """현재 거래 세션 확인"""
        hour = timestamp.hour
        
        for session, (start, end) in self.session_hours.items():
            if start <= hour < end:
                return session
        
        return TradingSession.ASIA  # 기본값
    
    def should_filter_signal(self, 
                           timestamp: pd.Timestamp,
                           signal_strength: float,
                           volatility: float) -> bool:
        """신호 필터링 여부 결정"""
        session = self.get_current_session(timestamp)
        noise_level = self.session_noise_levels[session]
        
        # 주말 필터
        if timestamp.weekday() >= 5:  # 토요일, 일요일
            return True
        
        # 세션별 노이즈 기준
        noise_threshold = noise_level * (1 + volatility)
        
        # 신호가 노이즈 수준보다 약하면 필터
        return signal_strength < noise_threshold
    
    def adjust_for_session(self, 
                         signal: SignalResult,
                         timestamp: pd.Timestamp) -> SignalResult:
        """세션별 신호 조정"""
        session = self.get_current_session(timestamp)
        
        # US 세션은 변동성이 높으므로 포지션 축소
        if session == TradingSession.US:
            signal.position_size *= 0.8
        
        # 아시아 세션은 안정적이므로 약간 증가
        elif session == TradingSession.ASIA:
            signal.position_size *= 1.1
        
        return signal


class MultiTimeframeConfirmation:
    """다중 타임프레임 확인"""
    
    def __init__(self):
        """초기화"""
        # 타임프레임별 가중치
        self.timeframe_weights = {
            "5T": 0.5,
            "15T": 0.3,
            "1H": 0.2
        }
    
    def calculate_alignment(self, 
                          timeframe_signals: Dict[str, str],
                          timeframe_strengths: Dict[str, float]) -> Tuple[float, str]:
        """타임프레임 신호 정렬도 계산"""
        if not timeframe_signals:
            return 0.0, "HOLD"
        
        # 신호별 점수 계산
        signal_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        
        for tf, signal in timeframe_signals.items():
            weight = self.timeframe_weights.get(tf, 0.0)
            strength = timeframe_strengths.get(tf, 0.5)
            
            signal_scores[signal] += weight * strength
        
        # 최고 점수 신호 선택
        best_signal = max(signal_scores, key=signal_scores.get)
        alignment_score = signal_scores[best_signal]
        
        # 정렬도가 낮으면 HOLD
        if alignment_score < 0.4:
            return alignment_score, "HOLD"
        
        return alignment_score, best_signal
    
    def get_higher_timeframe_trend(self, 
                                 mtf_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """상위 타임프레임 추세 확인"""
        trends = {}
        
        for tf, data in mtf_data.items():
            if len(data) < 20:
                trends[tf] = "neutral"
                continue
            
            # 단순 이동평균 기반 추세
            sma = data['close'].rolling(20).mean()
            current = data['close'].iloc[-1]
            
            if current > sma.iloc[-1] * 1.01:
                trends[tf] = "up"
            elif current < sma.iloc[-1] * 0.99:
                trends[tf] = "down"
            else:
                trends[tf] = "neutral"
        
        return trends


class SignalProcessor:
    """
    시그널 처리 메인 클래스
    모든 시그널 처리 컴포넌트 통합
    """
    
    def __init__(self):
        """초기화"""
        self.adaptive_entry = AdaptiveEntrySystem()
        self.probabilistic_entry = ProbabilisticEntry()
        self.time_filter = TimeBasedFilter()
        self.mtf_confirmation = MultiTimeframeConfirmation()
        
        self.min_data_points = 50
    
    def process_signal(self, 
                      strategy_signal: Dict[str, Any],
                      market_data: Dict[str, Any],
                      indicators: Dict[str, pd.Series],
                      sentiment_score: float) -> SignalResult:
        """
        전략 신호를 처리하여 최종 거래 신호 생성
        
        Args:
            strategy_signal: 전략에서 생성된 원시 신호
            market_data: 시장 데이터 (price, timeframes 등)
            indicators: 계산된 지표들
            sentiment_score: 감정 점수
            
        Returns:
            처리된 신호 결과
        """
        try:
            price_data = market_data["price"]
            timestamp = price_data['timestamp'].iloc[-1]
            current_price = price_data['close'].iloc[-1]
            
            # 입력 검증 - indicators가 None이거나 빈 딕셔너리일 경우 빈 딕셔너리로 초기화
            if not indicators:
                indicators = {}
            
            # 데이터 충분성 확인
            if len(price_data) < self.min_data_points:
                return SignalResult(
                    action="HOLD",
                    confidence=0.0,
                    position_size=0.0,
                    entry_price=current_price,
                    reason="Insufficient data"
                )
            
            # 1. 시장 레짐 감지
            regime, regime_confidence = self.adaptive_entry.detect_regime(price_data, indicators)
            
            # 2. 변동성 계산
            volatility = price_data['close'].pct_change().rolling(20).std().iloc[-1]
            
            # 3. 원시 신호 추출
            raw_action = strategy_signal.get("action", "HOLD").upper()  # 대소문자 통일
            raw_strength = strategy_signal.get("strength", 0.5)
            
            # 4. 시간대 필터링
            if self.time_filter.should_filter_signal(timestamp, raw_strength, volatility):
                return SignalResult(
                    action="HOLD",
                    confidence=0.0,
                    position_size=0.0,
                    entry_price=current_price,
                    reason="Filtered by time/noise"
                )
            
            # 5. 다중 타임프레임 확인 (있는 경우)
            alignment_score = raw_strength  # 기본값
            if "timeframes" in market_data:
                # 각 타임프레임에서 신호 생성 (간단한 예시)
                tf_signals = {}
                tf_strengths = {}
                
                for tf, tf_data in market_data["timeframes"].items():
                    if len(tf_data) >= 20:
                        tf_sma = tf_data['close'].rolling(20).mean()
                        tf_current = tf_data['close'].iloc[-1]
                        
                        if tf_current > tf_sma.iloc[-1] * 1.01:
                            tf_signals[tf] = "BUY"
                            tf_strengths[tf] = (tf_current / tf_sma.iloc[-1] - 1) * 10
                        elif tf_current < tf_sma.iloc[-1] * 0.99:
                            tf_signals[tf] = "SELL"
                            tf_strengths[tf] = (1 - tf_current / tf_sma.iloc[-1]) * 10
                        else:
                            tf_signals[tf] = "HOLD"
                            tf_strengths[tf] = 0.5
                
                alignment_score, aligned_action = self.mtf_confirmation.calculate_alignment(
                    tf_signals, tf_strengths
                )
                
                # 원시 신호와 정렬된 신호가 다르면 신뢰도 감소
                if aligned_action != raw_action:
                    alignment_score *= 0.7
            
            # 6. 임계값 조정
            threshold = 0.7  # 기본 임계값
            adjusted_threshold = self.adaptive_entry.adjust_thresholds(
                threshold, raw_action, regime, sentiment_score, volatility
            )
            
            # 7. 신호 강도가 조정된 임계값을 넘지 못하면 HOLD
            if raw_strength < adjusted_threshold:
                return SignalResult(
                    action="HOLD",
                    confidence=raw_strength / adjusted_threshold,
                    position_size=0.0,
                    entry_price=current_price,
                    reason=f"Below threshold: {raw_strength:.2f} < {adjusted_threshold:.2f}"
                )
            
            # 8. 신뢰도 계산
            confidence = self.probabilistic_entry.calculate_confidence(
                indicators, regime_confidence, alignment_score
            )
            
            # 9. 포지션 크기 계산
            position_size = self.probabilistic_entry.calculate_position_size(
                raw_strength, confidence, volatility, sentiment_score
            )
            
            # 10. Stop Loss / Take Profit 계산
            atr = indicators.get("atr", pd.Series([current_price * 0.02]))
            try:
                atr_value = atr.iloc[-1] if atr is not None and not atr.empty else current_price * 0.02
            except (IndexError, ValueError):
                atr_value = current_price * 0.02
            
            if raw_action == "BUY":
                stop_loss = current_price - 2 * atr_value
                take_profit = current_price + 3 * atr_value
            elif raw_action == "SELL":
                stop_loss = current_price + 2 * atr_value
                take_profit = current_price - 3 * atr_value
            else:
                stop_loss = None
                take_profit = None
            
            # 11. 세션별 조정
            signal_result = SignalResult(
                action=raw_action,
                confidence=confidence,
                position_size=position_size,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Regime: {regime.value}, Sentiment: {sentiment_score:.2f}",
                metadata={
                    "regime": regime.value,
                    "volatility": volatility,
                    "sentiment_score": sentiment_score,
                    "alignment_score": alignment_score,
                    "adjusted_threshold": adjusted_threshold
                }
            )
            
            # 세션 조정 적용
            signal_result = self.time_filter.adjust_for_session(signal_result, timestamp)
            
            return signal_result
            
        except Exception as e:
            logger.error(f"신호 처리 오류: {e}")
            return SignalResult(
                action="HOLD",
                confidence=0.0,
                position_size=0.0,
                entry_price=market_data["price"]['close'].iloc[-1],
                reason=f"Error: {str(e)}"
            )