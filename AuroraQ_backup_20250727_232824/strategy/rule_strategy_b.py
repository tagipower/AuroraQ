"""
Enhanced RuleStrategyB - 거래량 스파이크 + RSI with Caching
향상된 캐싱과 필터링이 적용된 전략 B
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from strategy.base_rule_strategy import BaseRuleStrategy
from utils.logger import get_logger
from config.rule_param_loader import get_rule_params

logger = get_logger("RuleStrategyB")


class RuleStrategyB(BaseRuleStrategy):
    """
    향상된 RuleStrategyB - 거래량 스파이크 + RSI + 이동평균 전략
    
    개선사항:
    - 향상된 지표 캐싱 시스템 활용
    - 강화된 진입 조건 필터
    - 표준화된 메트릭 사용
    - 중복 계산 제거
    
    진입 조건:
    1. 이동평균 상승 추세 (강화된 조건)
    2. 거래량 스파이크 (더 엄격한 기준)
    3. RSI 적정 범위 (더 신중한 범위)
    4. 향상된 필터 시스템 통과
    
    청산 조건:
    1. 동적 TP/SL 도달
    2. 필터 조건 악화
    3. RSI 극값 도달
    4. 신뢰도 기반 조기 청산
    """

    def __init__(self):
        super().__init__(name="RuleStrategyB")
        
        # 파라미터 로드
        params = get_rule_params("RuleB")
        
        # 이동평균 설정 (더 엄격하게 조정)
        self.ma_short_len = params.get("ma_short_len", 8)        # 5 → 8 (더 안정적)
        self.ma_long_len = params.get("ma_long_len", 35)         # 30 → 35 (트렌드 확인)
        self.ema60_len = params.get("ema60_len", 60)
        self.ema120_len = params.get("ema120_len", 120)
        
        # 거래량 및 RSI 설정 (더 엄격하게)
        self.volume_spike_base = params.get("volume_spike_ratio", 1.5)  # 1.1 → 1.5 (강화)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold_low = params.get("rsi_threshold_low", 30)    # 25 → 30 (더 보수적)
        self.rsi_threshold_high = params.get("rsi_threshold_high", 70)  # 75 → 70 (더 보수적)
        
        # 감정 분석 설정 (더 엄격)
        self.sentiment_threshold = params.get("sentiment_threshold", 0.2)  # 0.1 → 0.2
        
        # 리스크 관리 (보수적으로 조정)
        self.take_profit_pct = params.get("take_profit_pct", 0.028)     # 2.5% → 2.8%
        self.stop_loss_pct = params.get("stop_loss_pct", 0.015)        # 1.3% → 1.5%
        self.max_hold_bars = params.get("max_hold_bars", 8)             # 더 빠른 회전
        
        # 추가 조건 (강화)
        self.momentum_threshold = params.get("momentum_threshold", 0.002)  # 0.001 → 0.002
        self.min_volume_ratio = params.get("min_volume_ratio", 1.2)     # 1.0 → 1.2
        self.trend_strength_threshold = params.get("trend_strength_threshold", 0.002)  # 0.001 → 0.002
        
        # RSI 극값 설정 (더 보수적)
        self.rsi_oversold = params.get("rsi_oversold", 25)      # 30 → 25
        self.rsi_overbought = params.get("rsi_overbought", 75)  # 70 → 75
        
        # 동적 조정 파라미터
        self.vol_multiplier_tp = params.get("vol_multiplier_tp", 1.2)   # 보수적
        self.vol_multiplier_sl = params.get("vol_multiplier_sl", 1.1)   
        self.volatility_window = params.get("volatility_window", 20)
        
        # 상태 추적
        self.entry_metrics = {}
        self._last_rsi = 50.0
        self._last_volume_ratio = 1.0
        
        logger.info(
            f"향상된 RuleStrategyB 초기화: "
            f"MA({self.ma_short_len}/{self.ma_long_len}), "
            f"EMA({self.ema60_len}/{self.ema120_len}), "
            f"Volume({self.volume_spike_base}), RSI({self.rsi_period})"
        )

    def calculate_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 지표 계산 (캐시 활용)"""
        try:
            if len(price_data) < max(self.ma_long_len, self.ema120_len, self.rsi_period + 1):
                return {}
            
            indicators = {}
            
            # 1. 이동평균 계산 (캐시 활용)
            ma_short = self.get_cached_indicator(
                "sma", price_data, period=self.ma_short_len
            )
            ma_long = self.get_cached_indicator(
                "sma", price_data, period=self.ma_long_len
            )
            ema60 = self.get_cached_indicator(
                "ema", price_data, period=self.ema60_len
            )
            ema120 = self.get_cached_indicator(
                "ema", price_data, period=self.ema120_len
            )
            
            if ma_short is not None and ma_long is not None:
                indicators["ma_short"] = float(ma_short.iloc[-1])
                indicators["ma_long"] = float(ma_long.iloc[-1])
                indicators["ma_trend_ok"] = indicators["ma_short"] > indicators["ma_long"]
                indicators["ma_diff_pct"] = (indicators["ma_short"] - indicators["ma_long"]) / indicators["ma_long"] if indicators["ma_long"] > 0 else 0.0
            
            if ema60 is not None and ema120 is not None:
                indicators["ema60"] = float(ema60.iloc[-1])
                indicators["ema120"] = float(ema120.iloc[-1])
                indicators["ema_trend_ok"] = indicators["ema60"] > indicators["ema120"]
                
                # 추세 강도 계산
                if indicators["ema120"] > 0:
                    trend_strength = (indicators["ema60"] - indicators["ema120"]) / indicators["ema120"]
                    indicators["trend_strength"] = float(trend_strength)
            
            # 2. RSI 계산 (캐시 활용)
            rsi_series = self.get_cached_indicator(
                "rsi", price_data, period=self.rsi_period
            )
            
            if rsi_series is not None:
                rsi = float(rsi_series.iloc[-1])
                indicators["rsi"] = rsi
                indicators["rsi_neutral"] = self.rsi_threshold_low <= rsi <= self.rsi_threshold_high
                indicators["rsi_oversold"] = rsi <= self.rsi_oversold
                indicators["rsi_overbought"] = rsi >= self.rsi_overbought
                self._last_rsi = rsi
            
            # 3. 거래량 분석
            if "volume" in price_data.columns and len(price_data) >= 20:
                avg_volume = price_data["volume"].tail(20).mean()
                current_volume = price_data["volume"].iloc[-1]
                volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                
                indicators["volume_ratio"] = volume_ratio
                indicators["volume_spike"] = volume_ratio >= self.volume_spike_base
                indicators["volume_sufficient"] = volume_ratio >= self.min_volume_ratio
                self._last_volume_ratio = volume_ratio
            
            # 4. 변동성 계산
            returns = price_data["close"].pct_change().dropna()
            if len(returns) >= self.volatility_window:
                volatility = float(returns.tail(self.volatility_window).std())
                indicators["volatility"] = volatility
            
            # 5. 가격 모멘텀
            if len(price_data) >= 5:
                momentum = (price_data["close"].iloc[-1] - price_data["close"].iloc[-5]) / price_data["close"].iloc[-5]
                indicators["price_momentum"] = float(momentum)
                indicators["momentum_positive"] = momentum >= self.momentum_threshold
            
            # 6. 추가 지표
            if len(price_data) >= 2:
                price_change = (price_data["close"].iloc[-1] - price_data["close"].iloc[-2]) / price_data["close"].iloc[-2]
                indicators["price_change"] = float(price_change)
            
            return indicators
            
        except Exception as e:
            logger.error(f"지표 계산 오류: {e}")
            return {}

    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """향상된 진입 조건 확인"""
        try:
            indicators = self.calculate_indicators(price_data)
            
            # 필수 지표 확인
            required_indicators = ["ma_short", "ma_long", "ema60", "ema120", "rsi", "volume_ratio"]
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
            tp, sl = self._calculate_dynamic_thresholds(indicators.get("volatility", 0.02))
            
            # 강화된 진입 조건
            conditions = {
                # 1. 이동평균 추세 확인 (강화된 조건)
                "ma_trend": indicators.get("ma_trend_ok", False),
                "ema_trend": indicators.get("ema_trend_ok", False),
                "ma_diff_sufficient": abs(indicators.get("ma_diff_pct", 0)) >= self.trend_strength_threshold,
                
                # 2. 거래량 조건 (더 엄격)
                "volume_spike": indicators.get("volume_spike", False),
                "volume_sufficient": indicators.get("volume_sufficient", False),
                
                # 3. RSI 조건 (더 보수적)
                "rsi_neutral": indicators.get("rsi_neutral", False),
                
                # 4. 감정 및 모멘텀
                "sentiment_ok": sentiment >= self.sentiment_threshold,
                "momentum_positive": indicators.get("momentum_positive", False),
                
                # 5. 변동성 적정
                "volatility_ok": 0.005 <= indicators.get("volatility", 0.02) <= 0.06,
                
                # 6. 추세 강도
                "trend_strength_ok": abs(indicators.get("trend_strength", 0)) >= self.trend_strength_threshold
            }
            
            # 핵심 조건 (모두 만족 필요)
            core_conditions = [
                conditions["ma_trend"],
                conditions["ema_trend"],
                conditions["volume_spike"],
                conditions["rsi_neutral"]
            ]
            
            # 보조 조건 (최소 4개 만족)
            aux_conditions = [
                conditions["ma_diff_sufficient"],
                conditions["volume_sufficient"],
                conditions["sentiment_ok"],
                conditions["momentum_positive"],
                conditions["volatility_ok"],
                conditions["trend_strength_ok"]
            ]
            
            # 모든 핵심 조건 + 보조 조건 4개 이상 (더 엄격)
            if all(core_conditions) and sum(aux_conditions) >= 4:
                # 신뢰도 계산
                confidence = self._calculate_enhanced_confidence(indicators, conditions, sentiment)
                
                # 최소 신뢰도 확인
                if confidence < 0.65:  # 더 엄격한 기준
                    logger.debug(f"신뢰도 부족: {confidence:.3f}")
                    return None
                
                # 리스크/리워드 비율 확인
                risk_reward_ratio = tp / sl
                if risk_reward_ratio < 1.6:  # 더 엄격한 기준
                    logger.debug(f"리스크/리워드 비율 부족: {risk_reward_ratio:.2f}")
                    return None
                
                # 진입 메트릭 저장
                self.entry_metrics = {
                    "ma_diff_pct": indicators.get("ma_diff_pct", 0),
                    "volume_ratio": indicators.get("volume_ratio", 1.0),
                    "rsi": indicators.get("rsi", 50),
                    "sentiment": sentiment,
                    "trend_strength": indicators.get("trend_strength", 0)
                }
                
                # 진입 정보 생성
                entry_info = {
                    "side": "LONG",
                    "confidence": confidence,
                    "reason": self._generate_detailed_entry_reason(conditions, indicators, sentiment),
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
                    f"Volume: {indicators['volume_ratio']:.1f}x | "
                    f"RSI: {indicators['rsi']:.0f} | "
                    f"감정: {sentiment:.2f}"
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
            
            # 감정 점수
            sentiment = float(getattr(price_data, 'sentiment', 0.0))
            
            # 동적 TP/SL 가져오기
            indicators = self.calculate_indicators(price_data)
            if indicators:
                tp, sl = self._calculate_dynamic_thresholds(indicators.get("volatility", 0.02))
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
                 position.confidence < 0.4 and pnl_ratio < -0.005,
                 f"낮은신뢰도청산 (신뢰도:{getattr(position, 'confidence', 0):.2f})"),
                
                # 4. 감정 점수 하락
                (sentiment < self.sentiment_threshold - 0.1, f"감정악화 ({sentiment:.2f})"),
            ]
            
            # RSI 극값 체크
            if indicators and "rsi" in indicators:
                current_rsi = indicators["rsi"]
                entry_rsi = self.entry_metrics.get("rsi", 50)
                
                # RSI 과매수 영역 진입
                if current_rsi >= self.rsi_overbought:
                    exit_conditions.append((True, f"RSI과매수 ({current_rsi:.0f})"))
                
                # RSI 급격한 하락
                elif entry_rsi > 50 and current_rsi < entry_rsi - 15:
                    exit_conditions.append((True, f"RSI급락 ({current_rsi:.0f})"))
            
            # 거래량 급감 체크
            if indicators and "volume_ratio" in indicators:
                current_volume = indicators["volume_ratio"]
                entry_volume = self.entry_metrics.get("volume_ratio", 1.0)
                
                # 거래량이 진입 시점 대비 50% 이하로 감소
                if current_volume < entry_volume * 0.5:
                    exit_conditions.append((True, f"거래량급감 ({current_volume:.1f}x)"))
            
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

    def _calculate_dynamic_thresholds(self, volatility: float) -> Tuple[float, float]:
        """변동성 기반 동적 임계값 계산"""
        # 변동성 정규화
        vol_clamped = max(0.005, min(volatility, 0.08))
        vol_adjustment = (vol_clamped - 0.005) / 0.075  # 0~1로 정규화
        
        # 동적 조정
        tp = self.take_profit_pct * (1 + vol_adjustment * self.vol_multiplier_tp)
        sl = self.stop_loss_pct * (1 + vol_adjustment * self.vol_multiplier_sl)
        
        # 범위 제한
        tp = max(0.02, min(tp, 0.08))   # 2% ~ 8%
        sl = max(0.01, min(sl, 0.04))   # 1% ~ 4%
        
        return tp, sl

    def _calculate_enhanced_confidence(self, 
                                     indicators: Dict[str, float], 
                                     conditions: Dict[str, bool],
                                     sentiment: float) -> float:
        """향상된 신뢰도 계산"""
        base_confidence = 0.5
        
        # 이동평균 추세 강도 보너스
        ma_diff_pct = abs(indicators.get("ma_diff_pct", 0))
        base_confidence += min(0.15, ma_diff_pct * 30)
        
        # 거래량 스파이크 보너스
        volume_ratio = indicators.get("volume_ratio", 1.0)
        volume_bonus = min(0.2, (volume_ratio - self.volume_spike_base) * 0.1)
        base_confidence += volume_bonus
        
        # RSI 중립성 보너스
        rsi = indicators.get("rsi", 50)
        rsi_distance = abs(rsi - 50) / 50
        base_confidence += 0.1 * (1 - rsi_distance)
        
        # 감정 점수 보너스
        sentiment_bonus = min(0.12, max(0, (sentiment - self.sentiment_threshold) * 0.4))
        base_confidence += sentiment_bonus
        
        # 추세 강도 보너스
        trend_strength = abs(indicators.get("trend_strength", 0))
        base_confidence += min(0.1, trend_strength * 20)
        
        # 모멘텀 보너스
        if conditions.get("momentum_positive", False):
            momentum = indicators.get("price_momentum", 0)
            base_confidence += min(0.08, momentum * 15)
        
        return max(0.0, min(1.0, base_confidence))

    def _generate_detailed_entry_reason(self, 
                                       conditions: Dict[str, bool], 
                                       indicators: Dict[str, float],
                                       sentiment: float) -> str:
        """상세한 진입 이유 생성"""
        reasons = []
        
        if conditions.get("ma_trend") and conditions.get("ema_trend"):
            ma_diff = indicators.get("ma_diff_pct", 0)
            reasons.append(f"상승추세({ma_diff:.2%})")
        
        if conditions.get("volume_spike"):
            volume_ratio = indicators.get("volume_ratio", 1.0)
            reasons.append(f"거래량급증({volume_ratio:.1f}x)")
        
        if conditions.get("rsi_neutral"):
            rsi = indicators.get("rsi", 50)
            reasons.append(f"RSI중립({rsi:.0f})")
        
        if conditions.get("sentiment_ok"):
            reasons.append(f"긍정감정({sentiment:.2f})")
        
        if conditions.get("momentum_positive"):
            momentum = indicators.get("price_momentum", 0)
            reasons.append(f"모멘텀({momentum:.3%})")
        
        if conditions.get("trend_strength_ok"):
            strength = indicators.get("trend_strength", 0)
            reasons.append(f"강한추세({strength:.3%})")
        
        return ", ".join(reasons) if reasons else "향상된조건충족"

    def get_regime_preferences(self) -> Dict[str, float]:
        """시장 체제별 선호도 (전략 B 맞춤)"""
        return {
            "bull": 0.85,       # 상승장 선호
            "bear": 0.3,        # 하락장 제한적
            "sideways": 0.6,    # 횡보장 보통
            "high_volatility": 0.7,  # 변동성 활용
            "crisis": 0.15      # 위기 상황 회피
        }