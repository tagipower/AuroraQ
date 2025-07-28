"""
Enhanced RuleStrategyD - 변동성 + 감정 + 시장 체제 with Caching
향상된 캐싱과 필터링이 적용된 전략 D
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from strategy.base_rule_strategy import BaseRuleStrategy
from utils.logger import get_logger
from config.rule_param_loader import get_rule_params

logger = get_logger("RuleStrategyD")


class RuleStrategyD(BaseRuleStrategy):
    """
    향상된 RuleStrategyD - 변동성 + 감정 + 시장 체제 기반 전략
    
    개선사항:
    - 향상된 지표 캐싱 시스템 활용
    - 강화된 진입 조건 필터
    - 표준화된 메트릭 사용
    - 중복 계산 제거
    
    진입 조건:
    1. 변동성 비율 적정 범위 (더 엄격한 기준)
    2. 감정 점수 긍정적 임계값 (더 보수적)
    3. 시장 체제 점수 우호적 (더 신중한 판단)
    4. 추세 조건 + 기술적 지표 확인
    5. 향상된 필터 시스템 통과
    
    청산 조건:
    1. 동적 TP/SL 도달
    2. 체제/감정 변화 감지
    3. 변동성 과열 보호
    4. 신뢰도 기반 조기 청산
    """

    def __init__(self):
        super().__init__(name="RuleStrategyD")
        
        # 파라미터 로드
        params = get_rule_params("RuleD")
        
        # 변동성 설정 (더 엄격하게)
        self.volatility_ratio_min = params.get("volatility_ratio_min", 1.2)    # 최소 변동성 요구
        self.volatility_ratio_max = params.get("volatility_ratio_max", 2.8)    # 최대 변동성 허용
        self.volatility_ratio_exit = params.get("volatility_ratio_exit", 3.5)  # 1.8 → 3.5 (더 관대)
        
        # 감정 및 시장 체제 설정 (더 엄격하게)
        self.sentiment_entry_threshold = params.get("sentiment_entry_threshold", 0.25)  # 0.1 → 0.25
        self.regime_entry_threshold = params.get("regime_entry_threshold", 0.3)        # 0.15 → 0.3
        self.exit_sentiment_drop = params.get("exit_sentiment_drop", 0.2)              # 0.15 → 0.2
        
        # 추세 조건 (더 엄격)
        self.trend_condition = params.get("trend_condition", "long")  # "any" → "long" (복원)
        self.trend_strength_min = params.get("trend_strength_min", 0.1)  # 최소 추세 강도
        
        # 리스크 관리 (보수적으로 조정)
        self.stop_loss_pct = params.get("stop_loss_pct", 0.022)     # 2.5% → 2.2%
        self.take_profit_pct = params.get("take_profit_pct", 0.04)  # 3.5% → 4%
        self.max_hold_bars = params.get("max_hold_bars", 20)        # 40 → 20 (더 빠른 회전)
        
        # 기술적 지표 설정 (더 엄격)
        self.volatility_window = params.get("volatility_window", 20)  # 15 → 20 (복원)
        self.trend_window = params.get("trend_window", 50)            # 30 → 50 (복원)
        self.rsi_period = params.get("rsi_period", 14)
        self.volume_threshold = params.get("volume_threshold", 1.3)   # 1.0 → 1.3
        
        # RSI 범위 (더 보수적)
        self.rsi_neutral_min = params.get("rsi_neutral_min", 35)     # 25 → 35
        self.rsi_neutral_max = params.get("rsi_neutral_max", 65)     # 75 → 65
        
        # 동적 조정 파라미터
        self.vol_multiplier_tp = params.get("vol_multiplier_tp", 1.4)
        self.vol_multiplier_sl = params.get("vol_multiplier_sl", 1.2)
        
        # 추가 필터 설정
        self.momentum_window = params.get("momentum_window", 5)
        self.momentum_threshold = params.get("momentum_threshold", 0.003)  # 더 강한 모멘텀 요구
        self.min_price_change = params.get("min_price_change", 0.002)     # 최소 가격 변화
        
        # 상태 추적
        self.entry_metrics = {}
        self._last_volatility_ratio = 1.0
        self._last_trend_score = 0.0
        
        logger.info(
            f"향상된 RuleStrategyD 초기화: "
            f"변동성({self.volatility_ratio_min}-{self.volatility_ratio_max}), "
            f"감정({self.sentiment_entry_threshold}), "
            f"체제({self.regime_entry_threshold})"
        )

    def calculate_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 지표 계산 (캐시 활용)"""
        try:
            if len(price_data) < max(self.volatility_window * 2, self.trend_window, self.rsi_period + 1):
                return {}
            
            indicators = {}
            
            # 1. 변동성 비율 계산 (캐시 활용 - 간단한 방법)
            returns = price_data["close"].pct_change().dropna()
            if len(returns) >= self.volatility_window * 2:
                recent_vol = returns.tail(self.volatility_window).std()
                past_vol = returns.tail(self.volatility_window * 2).head(self.volatility_window).std()
                
                if past_vol > 0:
                    volatility_ratio = recent_vol / past_vol
                    indicators["volatility_ratio"] = float(volatility_ratio)
                    self._last_volatility_ratio = volatility_ratio
                else:
                    indicators["volatility_ratio"] = self._last_volatility_ratio
                
                # 변동성 범위 조건
                indicators["volatility_ok"] = self.volatility_ratio_min <= indicators["volatility_ratio"] <= self.volatility_ratio_max
            
            # 2. 추세 점수 계산 (캐시 활용)
            sma_short = self.get_cached_indicator(
                "sma", price_data, period=10
            )
            sma_long = self.get_cached_indicator(
                "sma", price_data, period=self.trend_window
            )
            
            if sma_short is not None and sma_long is not None:
                sma_short_val = float(sma_short.iloc[-1])
                sma_long_val = float(sma_long.iloc[-1])
                
                if sma_long_val > 0:
                    trend_strength = (sma_short_val - sma_long_val) / sma_long_val
                    trend_score = np.tanh(trend_strength * 10)  # -1 ~ 1로 정규화
                    
                    indicators["trend_score"] = float(trend_score)
                    indicators["trend_strength"] = float(abs(trend_strength))
                    self._last_trend_score = trend_score
                    
                    # 추세 조건 확인
                    if self.trend_condition == "long":
                        indicators["trend_condition_ok"] = trend_score > self.trend_strength_min
                    elif self.trend_condition == "short":
                        indicators["trend_condition_ok"] = trend_score < -self.trend_strength_min
                    else:
                        indicators["trend_condition_ok"] = abs(trend_score) > self.trend_strength_min
            
            # 3. RSI 계산 (캐시 활용)
            rsi_series = self.get_cached_indicator(
                "rsi", price_data, period=self.rsi_period
            )
            
            if rsi_series is not None:
                rsi = float(rsi_series.iloc[-1])
                indicators["rsi"] = rsi
                indicators["rsi_neutral"] = self.rsi_neutral_min <= rsi <= self.rsi_neutral_max
            
            # 4. 거래량 비율 계산
            if "volume" in price_data.columns and len(price_data) >= 20:
                avg_volume = price_data["volume"].tail(20).mean()
                current_volume = price_data["volume"].iloc[-1]
                volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                
                indicators["volume_ratio"] = volume_ratio
                indicators["volume_sufficient"] = volume_ratio >= self.volume_threshold
            
            # 5. 가격 모멘텀 계산
            if len(price_data) >= self.momentum_window:
                momentum = (price_data["close"].iloc[-1] - price_data["close"].iloc[-self.momentum_window]) / price_data["close"].iloc[-self.momentum_window]
                indicators["price_momentum"] = float(momentum)
                indicators["momentum_positive"] = momentum > self.momentum_threshold
                indicators["momentum_strong"] = abs(momentum) > self.momentum_threshold
            
            # 6. 추가 지표
            if len(price_data) >= 2:
                price_change = (price_data["close"].iloc[-1] - price_data["close"].iloc[-2]) / price_data["close"].iloc[-2]
                indicators["price_change"] = float(price_change)
                indicators["price_change_sufficient"] = abs(price_change) >= self.min_price_change
            
            return indicators
            
        except Exception as e:
            logger.error(f"지표 계산 오류: {e}")
            return {}

    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """향상된 진입 조건 확인"""
        try:
            indicators = self.calculate_indicators(price_data)
            
            # 필수 지표 확인
            required_indicators = ["volatility_ratio", "trend_score", "rsi", "volume_ratio"]
            if not all(ind in indicators for ind in required_indicators):
                logger.debug("필수 지표 부족")
                return None
            
            # 현재 값들
            current_price = self.safe_last(price_data, "close")
            current_time = pd.Timestamp.now()
            
            # 감정 및 체제 점수 추출
            sentiment = float(getattr(price_data, 'sentiment', 0.0))
            regime = float(getattr(price_data, 'regime', 0.0))
            
            # 동적 임계값 계산
            tp, sl = self._calculate_dynamic_thresholds(indicators.get("volatility_ratio", 1.0))
            
            # 강화된 진입 조건
            conditions = {
                # 1. 변동성 적정 범위 (핵심 조건)
                "volatility_ok": indicators.get("volatility_ok", False),
                
                # 2. 감정 점수 긍정적 (더 엄격)
                "sentiment_ok": sentiment >= self.sentiment_entry_threshold,
                
                # 3. 시장 체제 우호적 (더 엄격)
                "regime_ok": regime >= self.regime_entry_threshold,
                
                # 4. 추세 조건 만족 (핵심 조건)
                "trend_ok": indicators.get("trend_condition_ok", False),
                
                # 5. RSI 중립 구간 (보수적)
                "rsi_neutral": indicators.get("rsi_neutral", False),
                
                # 6. 거래량 충분 (더 엄격)
                "volume_sufficient": indicators.get("volume_sufficient", False),
                
                # 7. 모멘텀 양호 (추가 조건)
                "momentum_positive": indicators.get("momentum_positive", False),
                
                # 8. 가격 변화 충분 (추가 조건)
                "price_change_sufficient": indicators.get("price_change_sufficient", False),
            }
            
            # 핵심 조건 (모두 만족 필요)
            core_conditions = [
                conditions["volatility_ok"],
                conditions["sentiment_ok"],
                conditions["regime_ok"],
                conditions["trend_ok"]
            ]
            
            # 보조 조건 (최소 3개 만족)
            aux_conditions = [
                conditions["rsi_neutral"],
                conditions["volume_sufficient"],
                conditions["momentum_positive"],
                conditions["price_change_sufficient"]
            ]
            
            # 모든 핵심 조건 + 보조 조건 3개 이상 (더 엄격)
            if all(core_conditions) and sum(aux_conditions) >= 3:
                # 신뢰도 계산
                confidence = self._calculate_enhanced_confidence(indicators, conditions, sentiment, regime)
                
                # 최소 신뢰도 확인
                if confidence < 0.7:  # 더 엄격한 기준
                    logger.debug(f"신뢰도 부족: {confidence:.3f}")
                    return None
                
                # 리스크/리워드 비율 확인
                risk_reward_ratio = tp / sl
                if risk_reward_ratio < 1.8:  # 더 엄격한 기준
                    logger.debug(f"리스크/리워드 비율 부족: {risk_reward_ratio:.2f}")
                    return None
                
                # 진입 메트릭 저장
                self.entry_metrics = {
                    "volatility_ratio": indicators["volatility_ratio"],
                    "trend_score": indicators["trend_score"],
                    "sentiment": sentiment,
                    "regime": regime,
                    "rsi": indicators["rsi"],
                    "momentum": indicators.get("price_momentum", 0)
                }
                
                # 진입 정보 생성
                entry_info = {
                    "side": "LONG",
                    "confidence": confidence,
                    "reason": self._generate_detailed_entry_reason(conditions, indicators, sentiment, regime),
                    "stop_loss": current_price * (1 - sl),
                    "take_profit": current_price * (1 + tp),
                    "indicators": indicators,
                    "conditions": conditions,
                    "risk_reward_ratio": risk_reward_ratio
                }
                
                logger.info(
                    f"[{self.name}] 향상된 진입 신호 | "
                    f"가격: {current_price:.2f} | "
                    f"신뢰도: {confidence:.3f} | "
                    f"변동성: {indicators['volatility_ratio']:.2f} | "
                    f"추세: {indicators['trend_score']:.2f} | "
                    f"감정: {sentiment:.2f} | "
                    f"체제: {regime:.2f}"
                )
                
                return entry_info
            
            return None
            
        except Exception as e:
            logger.error(f"진입 조건 확인 오류: {e}")
            return None

    def should_exit(self, position, price_data: pd.DataFrame) -> Optional[str]:
        """향상된 청산 조건 확인"""
        try:
            if not position or not hasattr(position, 'entry_price'):
                return None
            
            current_price = self.safe_last(price_data, "close")
            entry_price = position.entry_price
            
            # PnL 계산
            pnl_ratio = (current_price - entry_price) / entry_price
            
            # 보유 시간
            holding_time_seconds = position.holding_time.total_seconds()
            holding_bars = holding_time_seconds / 300  # 5분봉 기준
            
            # 현재 감정/체제 점수
            sentiment = float(getattr(price_data, 'sentiment', 0.0))
            regime = float(getattr(price_data, 'regime', 0.0))
            
            # 동적 TP/SL 가져오기
            indicators = self.calculate_indicators(price_data)
            if indicators:
                tp, sl = self._calculate_dynamic_thresholds(indicators.get("volatility_ratio", 1.0))
            else:
                tp, sl = self.take_profit_pct, self.stop_loss_pct
            
            # 청산 조건들 (우선순위 순)
            exit_conditions = [
                # 1. 손절/익절
                (pnl_ratio <= -sl, f"손절 ({pnl_ratio:.1%})"),
                (pnl_ratio >= tp, f"익절 ({pnl_ratio:.1%})"),
                
                # 2. 시간 초과
                (holding_bars >= self.max_hold_bars, f"시간초과 ({holding_bars:.0f}봉)"),
                
                # 3. 신뢰도 기반 조기 청산
                (hasattr(position, 'confidence') and 
                 position.confidence < 0.4 and pnl_ratio < -0.01,
                 f"낮은신뢰도청산 (신뢰도:{getattr(position, 'confidence', 0):.2f})"),
            ]
            
            # 4. 감정/체제 악화
            entry_sentiment = self.entry_metrics.get("sentiment", 0.5)
            entry_regime = self.entry_metrics.get("regime", 0.5)
            
            sentiment_drop = entry_sentiment - sentiment
            regime_drop = entry_regime - regime
            
            if sentiment_drop >= self.exit_sentiment_drop:
                exit_conditions.append((True, f"감정악화 ({sentiment_drop:.2f}하락)"))
            
            if regime_drop >= 0.2:  # 체제 점수 20% 이상 하락
                exit_conditions.append((True, f"체제악화 ({regime_drop:.2f}하락)"))
            
            # 5. 변동성 과열
            if indicators and "volatility_ratio" in indicators:
                current_vol = indicators["volatility_ratio"]
                if current_vol >= self.volatility_ratio_exit:
                    exit_conditions.append((True, f"변동성과열 (Vol:{current_vol:.2f})"))
            
            # 6. 추세 반전
            if indicators and "trend_score" in indicators:
                current_trend = indicators["trend_score"]
                entry_trend = self.entry_metrics.get("trend_score", 0)
                
                # LONG 포지션에서 추세가 음수로 전환되거나 크게 약화
                if entry_trend > 0.1 and current_trend < -0.05:
                    exit_conditions.append((True, f"추세반전 ({current_trend:.2f})"))
                elif entry_trend > 0.1 and current_trend < entry_trend * 0.3:  # 70% 이상 약화
                    exit_conditions.append((True, f"추세약화 ({current_trend:.2f})"))
            
            # 조건 확인
            for condition, reason in exit_conditions:
                if condition:
                    logger.info(
                        f"[{self.name}] 향상된 청산 신호 | "
                        f"이유: {reason} | "
                        f"PnL: {pnl_ratio:.2%} | "
                        f"보유: {holding_bars:.1f}봉"
                    )
                    return reason
            
            return None
            
        except Exception as e:
            logger.error(f"청산 조건 확인 오류: {e}")
            return None

    def _calculate_dynamic_thresholds(self, volatility_ratio: float) -> Tuple[float, float]:
        """변동성 기반 동적 임계값 계산"""
        # 변동성 비율을 기반으로 조정
        vol_clamped = max(1.0, min(volatility_ratio, 3.0))
        vol_adjustment = (vol_clamped - 1.0) / 2.0  # 0~1 범위로 정규화
        
        # 기본값에서 변동성에 따라 조정
        tp = self.take_profit_pct * (1 + vol_adjustment * self.vol_multiplier_tp)
        sl = self.stop_loss_pct * (1 + vol_adjustment * self.vol_multiplier_sl)
        
        # 합리적 범위 제한
        tp = max(0.025, min(tp, 0.12))   # 2.5% ~ 12%
        sl = max(0.015, min(sl, 0.06))   # 1.5% ~ 6%
        
        return tp, sl

    def _calculate_enhanced_confidence(self, 
                                     indicators: Dict[str, float], 
                                     conditions: Dict[str, bool],
                                     sentiment: float,
                                     regime: float) -> float:
        """향상된 신뢰도 계산"""
        base_confidence = 0.5
        
        # 변동성 적정성 보너스
        volatility_ratio = indicators.get("volatility_ratio", 1.0)
        if 1.5 <= volatility_ratio <= 2.0:  # 이상적 범위
            base_confidence += 0.15
        elif self.volatility_ratio_min <= volatility_ratio <= self.volatility_ratio_max:
            base_confidence += 0.1
        
        # 추세 강도 보너스
        trend_strength = indicators.get("trend_strength", 0)
        base_confidence += min(0.15, trend_strength * 1.5)
        
        # 감정 점수 보너스 (높을수록 좋음)
        sentiment_bonus = min(0.12, max(0, (sentiment - self.sentiment_entry_threshold) * 0.6))
        base_confidence += sentiment_bonus
        
        # 체제 점수 보너스 (높을수록 좋음)
        regime_bonus = min(0.12, max(0, (regime - self.regime_entry_threshold) * 0.4))
        base_confidence += regime_bonus
        
        # RSI 중립 보너스
        if conditions.get("rsi_neutral", False):
            rsi = indicators.get("rsi", 50)
            # 50에 가까울수록 높은 점수
            rsi_distance = abs(rsi - 50) / 50
            base_confidence += 0.08 * (1 - rsi_distance)
        
        # 거래량 보너스
        if conditions.get("volume_sufficient", False):
            volume_ratio = indicators.get("volume_ratio", 1.0)
            volume_bonus = min(0.08, (volume_ratio - self.volume_threshold) * 0.1)
            base_confidence += volume_bonus
        
        # 모멘텀 보너스
        if conditions.get("momentum_positive", False):
            momentum = indicators.get("price_momentum", 0)
            base_confidence += min(0.08, momentum * 20)
        
        return max(0.0, min(1.0, base_confidence))

    def _generate_detailed_entry_reason(self, 
                                       conditions: Dict[str, bool], 
                                       indicators: Dict[str, float],
                                       sentiment: float,
                                       regime: float) -> str:
        """상세한 진입 이유 생성"""
        reasons = []
        
        if conditions.get("volatility_ok"):
            vol_ratio = indicators.get("volatility_ratio", 1.0)
            reasons.append(f"적정변동성({vol_ratio:.2f})")
        
        if conditions.get("trend_ok"):
            trend_score = indicators.get("trend_score", 0)
            reasons.append(f"상승추세({trend_score:.2f})")
        
        if conditions.get("sentiment_ok"):
            reasons.append(f"긍정감정({sentiment:.2f})")
        
        if conditions.get("regime_ok"):
            reasons.append(f"우호체제({regime:.2f})")
        
        if conditions.get("rsi_neutral"):
            rsi = indicators.get("rsi", 50)
            reasons.append(f"RSI중립({rsi:.0f})")
        
        if conditions.get("volume_sufficient"):
            volume_ratio = indicators.get("volume_ratio", 1.0)
            reasons.append(f"충분거래량({volume_ratio:.1f}x)")
        
        if conditions.get("momentum_positive"):
            momentum = indicators.get("price_momentum", 0)
            reasons.append(f"양모멘텀({momentum:.3%})")
        
        if conditions.get("price_change_sufficient"):
            price_change = indicators.get("price_change", 0)
            reasons.append(f"가격변화({price_change:.3%})")
        
        return ", ".join(reasons) if reasons else "향상된조건충족"

    def get_regime_preferences(self) -> Dict[str, float]:
        """시장 체제별 선호도 (전략 D 맞춤)"""
        return {
            "bull": 0.95,        # 상승장 최고 선호
            "bear": 0.1,         # 하락장 회피
            "sideways": 0.4,     # 횡보장 부적합
            "high_volatility": 0.8,  # 고변동성 활용 (적정 범위 내)
            "crisis": 0.05       # 위기 상황 강력 회피
        }