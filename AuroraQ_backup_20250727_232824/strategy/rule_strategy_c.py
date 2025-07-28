"""
Enhanced RuleStrategyC - 볼린저 밴드 + 감정 기반 역추세 with Caching
향상된 캐싱과 필터링이 적용된 전략 C
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from strategy.base_rule_strategy import BaseRuleStrategy
from utils.logger import get_logger
from config.rule_param_loader import get_rule_params

logger = get_logger("RuleStrategyC")


class RuleStrategyC(BaseRuleStrategy):
    """
    향상된 RuleStrategyC - 볼린저 밴드 + 감정 기반 역추세 전략
    
    개선사항:
    - 향상된 지표 캐싱 시스템 활용
    - 강화된 진입 조건 필터
    - 표준화된 메트릭 사용
    - 중복 계산 제거
    
    진입 조건:
    1. 볼린저 밴드 상단/하단 극값 (더 엄격한 기준)
    2. RSI 과매수/과매도 확인 (더 보수적 범위)
    3. 감정 점수 극값 감지 (더 신중한 기준)
    4. 변동성 스파이크 + 거래량 확인
    5. 향상된 필터 시스템 통과
    
    청산 조건:
    1. 동적 TP/SL 도달
    2. 반전 신호 감지
    3. 신뢰도 기반 조기 청산
    4. 시간 기반 보호
    """

    def __init__(self):
        super().__init__(name="RuleStrategyC")
        
        # 파라미터 로드
        params = get_rule_params("RuleC")
        
        # 볼린저 밴드 설정 (더 엄격하게)
        self.bb_window = params.get("bb_window", 20)
        self.bb_std = params.get("bb_std", 2.2)  # 2.0 → 2.2 (더 엄격한 기준)
        self.bb_std_entry = self.bb_std
        
        # 감정 분석 설정 (더 엄격하게)
        self.sentiment_threshold = params.get("sentiment_threshold", 0.3)      # 0.2 → 0.3
        self.sentiment_prev_threshold = params.get("sentiment_prev_threshold", 0.4)  # 0.3 → 0.4
        self.sentiment_drop_min = params.get("sentiment_drop_min", 0.12)       # 0.08 → 0.12
        
        # 변동성 및 리바운드 설정 (더 엄격하게)
        self.volatility_spike_ratio = params.get("volatility_spike_ratio", 1.8)     # 1.5 → 1.8
        self.rebound_sentiment_jump = params.get("rebound_sentiment_jump", 0.2)     # 0.15 → 0.2
        
        # RSI 설정 (더 엄격)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 25)      # 30 → 25 (더 극값)
        self.rsi_overbought = params.get("rsi_overbought", 75)  # 70 → 75 (더 극값)
        
        # 리스크 관리 (보수적으로)
        self.take_profit_pct = params.get("take_profit_pct", 0.03)      # 2.5% → 3%
        self.stop_loss_pct = params.get("stop_loss_pct", 0.018)        # 1.5% → 1.8%
        self.max_hold_bars = params.get("max_hold_bars", 6)             # 8 → 6 (더 빠른 회전)
        
        # 추가 필터 (강화)
        self.min_volume_ratio = params.get("min_volume_ratio", 1.1)     # 1.0 → 1.1
        self.bb_position_threshold = params.get("bb_position_threshold", 0.95)  # 0.9 → 0.95 (더 극값)
        self.reversal_strength_threshold = params.get("reversal_strength_threshold", 0.003)  # 0.002 → 0.003
        
        # 동적 조정 파라미터
        self.vol_multiplier_tp = params.get("vol_multiplier_tp", 1.3)
        self.vol_multiplier_sl = params.get("vol_multiplier_sl", 1.1)
        self.volatility_window = params.get("volatility_window", 15)    # 20 → 15 (더 빠른 반응)
        
        # 상태 추적
        self.entry_metrics = {}
        self._last_bb_position = 0.5
        self._last_sentiment = 0.0
        self._sentiment_history = []
        
        logger.info(
            f"향상된 RuleStrategyC 초기화: "
            f"BB({self.bb_window}, {self.bb_std}), "
            f"RSI({self.rsi_period}), "
            f"감정임계값({self.sentiment_threshold})"
        )

    def calculate_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 지표 계산 (캐시 활용)"""
        try:
            if len(price_data) < max(self.bb_window + 1, self.rsi_period + 1, self.volatility_window):
                return {}
            
            indicators = {}
            
            # 1. 볼린저 밴드 계산 (캐시 활용)
            bb_result = self.get_cached_indicator(
                "bollinger", price_data, period=self.bb_window, std=self.bb_std
            )
            
            if bb_result is not None and isinstance(bb_result, dict):
                current_price = self.safe_last(price_data, "close")
                
                indicators["bb_upper"] = float(bb_result["upper"].iloc[-1])
                indicators["bb_middle"] = float(bb_result["middle"].iloc[-1])
                indicators["bb_lower"] = float(bb_result["lower"].iloc[-1])
                
                # BB 포지션 계산 (0: 하단, 0.5: 중간, 1: 상단)
                bb_range = indicators["bb_upper"] - indicators["bb_lower"]
                if bb_range > 0:
                    bb_position = (current_price - indicators["bb_lower"]) / bb_range
                    indicators["bb_position"] = float(bb_position)
                    self._last_bb_position = bb_position
                    
                    # 극값 조건
                    indicators["bb_near_upper"] = bb_position >= self.bb_position_threshold
                    indicators["bb_near_lower"] = bb_position <= (1 - self.bb_position_threshold)
                
                # BB 폭 분석
                bb_width = bb_range / indicators["bb_middle"] if indicators["bb_middle"] > 0 else 0
                indicators["bb_width"] = float(bb_width)
                indicators["bb_squeeze"] = bb_width < 0.02  # 밴드 수축
                indicators["bb_expansion"] = bb_width > 0.08  # 밴드 확장
            
            # 2. RSI 계산 (캐시 활용)
            rsi_series = self.get_cached_indicator(
                "rsi", price_data, period=self.rsi_period
            )
            
            if rsi_series is not None:
                rsi = float(rsi_series.iloc[-1])
                indicators["rsi"] = rsi
                indicators["rsi_oversold"] = rsi <= self.rsi_oversold
                indicators["rsi_overbought"] = rsi >= self.rsi_overbought
                
                # RSI 다이버전스 체크
                if len(rsi_series) >= 5:
                    rsi_trend = float(rsi_series.iloc[-1] - rsi_series.iloc[-5])
                    price_trend = float(price_data["close"].iloc[-1] - price_data["close"].iloc[-5])
                    
                    # 다이버전스: 가격과 RSI가 반대 방향
                    if price_trend > 0 and rsi_trend < 0:
                        indicators["rsi_bearish_divergence"] = True
                    elif price_trend < 0 and rsi_trend > 0:
                        indicators["rsi_bullish_divergence"] = True
            
            # 3. 변동성 계산
            returns = price_data["close"].pct_change().dropna()
            if len(returns) >= self.volatility_window:
                current_vol = float(returns.tail(self.volatility_window).std())
                past_vol = float(returns.tail(self.volatility_window * 2).head(self.volatility_window).std())
                
                indicators["volatility"] = current_vol
                indicators["volatility_ratio"] = current_vol / past_vol if past_vol > 0 else 1.0
                indicators["volatility_spike"] = indicators["volatility_ratio"] >= self.volatility_spike_ratio
            
            # 4. 거래량 분석
            if "volume" in price_data.columns and len(price_data) >= 20:
                avg_volume = price_data["volume"].tail(20).mean()
                current_volume = price_data["volume"].iloc[-1]
                volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                
                indicators["volume_ratio"] = volume_ratio
                indicators["volume_sufficient"] = volume_ratio >= self.min_volume_ratio
            
            # 5. 가격 모멘텀 및 반전 신호
            if len(price_data) >= 3:
                # 단기 모멘텀
                momentum_1 = (price_data["close"].iloc[-1] - price_data["close"].iloc[-2]) / price_data["close"].iloc[-2]
                momentum_3 = (price_data["close"].iloc[-1] - price_data["close"].iloc[-3]) / price_data["close"].iloc[-3]
                
                indicators["momentum_1bar"] = float(momentum_1)
                indicators["momentum_3bar"] = float(momentum_3)
                
                # 반전 신호 감지
                indicators["reversal_signal"] = abs(momentum_1) >= self.reversal_strength_threshold and momentum_1 * momentum_3 < 0
            
            return indicators
            
        except Exception as e:
            logger.error(f"지표 계산 오류: {e}")
            return {}

    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """향상된 진입 조건 확인"""
        try:
            indicators = self.calculate_indicators(price_data)
            
            # 필수 지표 확인
            required_indicators = ["bb_upper", "bb_lower", "bb_position", "rsi", "volatility_ratio"]
            if not all(ind in indicators for ind in required_indicators):
                logger.debug("필수 지표 부족")
                return None
            
            # 현재 값들
            current_price = self.safe_last(price_data, "close")
            current_time = pd.Timestamp.now()
            
            # 감정 점수 분석
            sentiment = float(getattr(price_data, 'sentiment', 0.0))
            
            # 감정 이력 관리
            self._sentiment_history.append(sentiment)
            if len(self._sentiment_history) > 5:
                self._sentiment_history.pop(0)
            
            # 감정 변화 분석
            sentiment_change = 0.0
            if len(self._sentiment_history) >= 2:
                sentiment_change = sentiment - self._sentiment_history[-2]
            
            # 동적 임계값 계산
            tp, sl = self._calculate_dynamic_thresholds(indicators.get("volatility", 0.02))
            
            # 강화된 진입 조건 (역추세 전략)
            conditions = {
                # 1. 볼린저 밴드 극값 조건 (더 엄격)
                "bb_extreme": indicators.get("bb_near_upper", False) or indicators.get("bb_near_lower", False),
                "bb_upper_sell": indicators.get("bb_near_upper", False) and indicators.get("rsi_overbought", False),
                "bb_lower_buy": indicators.get("bb_near_lower", False) and indicators.get("rsi_oversold", False),
                
                # 2. 감정 극값 조건 (더 엄격)
                "sentiment_extreme": sentiment <= self.sentiment_threshold or sentiment >= (1 - self.sentiment_threshold),
                "sentiment_drop": sentiment_change <= -self.sentiment_drop_min,
                "sentiment_spike": sentiment_change >= self.rebound_sentiment_jump,
                
                # 3. 변동성 및 거래량 (더 엄격)
                "volatility_spike": indicators.get("volatility_spike", False),
                "volume_sufficient": indicators.get("volume_sufficient", False),
                
                # 4. RSI 확인 (더 보수적)
                "rsi_extreme": indicators.get("rsi_oversold", False) or indicators.get("rsi_overbought", False),
                
                # 5. 반전 신호
                "reversal_signal": indicators.get("reversal_signal", False),
                "rsi_divergence": indicators.get("rsi_bullish_divergence", False) or indicators.get("rsi_bearish_divergence", False),
                
                # 6. BB 상태
                "bb_not_squeeze": not indicators.get("bb_squeeze", False)  # 밴드 수축 시 진입 금지
            }
            
            # 진입 시나리오 판단
            entry_scenario = None
            
            # 시나리오 1: 하단 반등 (LONG)
            long_conditions = [
                conditions["bb_lower_buy"],
                conditions["sentiment_extreme"] and sentiment <= self.sentiment_threshold,
                conditions["volatility_spike"],
                conditions["volume_sufficient"],
                conditions["bb_not_squeeze"]
            ]
            
            # 시나리오 2: 감정 급락 후 반등 (LONG)
            rebound_conditions = [
                conditions["sentiment_drop"] or conditions["sentiment_spike"],
                indicators.get("bb_position", 0.5) < 0.3,  # 하단 근처
                indicators.get("rsi", 50) < 40,  # RSI 낮음
                conditions["volume_sufficient"],
                conditions["reversal_signal"]
            ]
            
            # 진입 조건 검증 (LONG만, 역추세 전략)
            if sum(long_conditions) >= 4:  # 5개 중 4개 이상
                entry_scenario = "bb_lower_reversal"
                side = "LONG"
            elif sum(rebound_conditions) >= 4:  # 5개 중 4개 이상
                entry_scenario = "sentiment_rebound"
                side = "LONG"
            else:
                return None
            
            # 신뢰도 계산
            confidence = self._calculate_enhanced_confidence(indicators, conditions, sentiment, sentiment_change)
            
            # 최소 신뢰도 확인 (더 엄격)
            if confidence < 0.7:
                logger.debug(f"신뢰도 부족: {confidence:.3f}")
                return None
            
            # 리스크/리워드 비율 확인
            risk_reward_ratio = tp / sl
            if risk_reward_ratio < 1.5:  # 더 엄격한 기준
                logger.debug(f"리스크/리워드 비율 부족: {risk_reward_ratio:.2f}")
                return None
            
            # 진입 메트릭 저장
            self.entry_metrics = {
                "bb_position": indicators.get("bb_position", 0.5),
                "rsi": indicators.get("rsi", 50),
                "sentiment": sentiment,
                "sentiment_change": sentiment_change,
                "volatility_ratio": indicators.get("volatility_ratio", 1.0),
                "entry_scenario": entry_scenario
            }
            
            # 진입 정보 생성
            entry_info = {
                "side": side,
                "confidence": confidence,
                "reason": self._generate_detailed_entry_reason(conditions, indicators, sentiment, entry_scenario),
                "stop_loss": current_price * (1 - sl),
                "take_profit": current_price * (1 + tp),
                "indicators": indicators,
                "conditions": conditions,
                "risk_reward_ratio": risk_reward_ratio
            }
            
            logger.info(
                f"[{self.name}] 향상된 진입 신호 | "
                f"가격: {current_price:.2f} | "
                f"시나리오: {entry_scenario} | "
                f"신뢰도: {confidence:.3f} | "
                f"BB위치: {indicators['bb_position']:.2f} | "
                f"RSI: {indicators['rsi']:.0f} | "
                f"감정: {sentiment:.2f}"
            )
            
            return entry_info
            
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
                 position.confidence < 0.4 and pnl_ratio < -0.008,
                 f"낮은신뢰도청산 (신뢰도:{getattr(position, 'confidence', 0):.2f})"),
            ]
            
            # 4. 반전 완료 신호 체크
            if indicators:
                entry_bb_position = self.entry_metrics.get("bb_position", 0.5)
                current_bb_position = indicators.get("bb_position", 0.5)
                entry_scenario = self.entry_metrics.get("entry_scenario", "")
                
                # BB 중간선 복귀
                if entry_bb_position < 0.3 and current_bb_position > 0.4:  # 하단에서 중간으로
                    exit_conditions.append((True, f"BB중간복귀 ({current_bb_position:.2f})"))
                elif entry_bb_position > 0.7 and current_bb_position < 0.6:  # 상단에서 중간으로
                    exit_conditions.append((True, f"BB중간복귀 ({current_bb_position:.2f})"))
                
                # RSI 중립 복귀
                current_rsi = indicators.get("rsi", 50)
                entry_rsi = self.entry_metrics.get("rsi", 50)
                
                if entry_rsi <= self.rsi_oversold and current_rsi >= 45:  # 과매도에서 중립으로
                    exit_conditions.append((True, f"RSI중립복귀 ({current_rsi:.0f})"))
                elif entry_rsi >= self.rsi_overbought and current_rsi <= 55:  # 과매수에서 중립으로
                    exit_conditions.append((True, f"RSI중립복귀 ({current_rsi:.0f})"))
            
            # 5. 감정 변화 체크
            entry_sentiment = self.entry_metrics.get("sentiment", 0.5)
            sentiment_recovery = abs(sentiment - entry_sentiment)
            
            if sentiment_recovery >= self.rebound_sentiment_jump:
                exit_conditions.append((True, f"감정회복 ({sentiment:.2f})"))
            
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
        vol_clamped = max(0.005, min(volatility, 0.1))
        vol_adjustment = (vol_clamped - 0.005) / 0.095  # 0~1로 정규화
        
        # 역추세 전략에 맞춘 조정
        tp = self.take_profit_pct * (1 + vol_adjustment * self.vol_multiplier_tp)
        sl = self.stop_loss_pct * (1 + vol_adjustment * self.vol_multiplier_sl)
        
        # 범위 제한
        tp = max(0.02, min(tp, 0.1))   # 2% ~ 10%
        sl = max(0.01, min(sl, 0.05))  # 1% ~ 5%
        
        return tp, sl

    def _calculate_enhanced_confidence(self, 
                                     indicators: Dict[str, float], 
                                     conditions: Dict[str, bool],
                                     sentiment: float,
                                     sentiment_change: float) -> float:
        """향상된 신뢰도 계산"""
        base_confidence = 0.5
        
        # BB 극값 위치 보너스
        bb_position = indicators.get("bb_position", 0.5)
        if bb_position <= 0.1 or bb_position >= 0.9:  # 극값
            base_confidence += 0.2
        elif bb_position <= 0.2 or bb_position >= 0.8:
            base_confidence += 0.15
        
        # RSI 극값 보너스
        rsi = indicators.get("rsi", 50)
        if rsi <= 20 or rsi >= 80:  # 극값
            base_confidence += 0.15
        elif rsi <= 30 or rsi >= 70:
            base_confidence += 0.1
        
        # 변동성 스파이크 보너스
        if conditions.get("volatility_spike", False):
            vol_ratio = indicators.get("volatility_ratio", 1.0)
            base_confidence += min(0.15, (vol_ratio - self.volatility_spike_ratio) * 0.1)
        
        # 감정 극값 보너스
        sentiment_distance = min(sentiment, 1 - sentiment)  # 0에 가까울수록 극값
        base_confidence += 0.1 * (1 - sentiment_distance * 2)  # 극값일수록 높은 점수
        
        # 감정 변화 보너스
        if abs(sentiment_change) >= self.sentiment_drop_min:
            base_confidence += min(0.1, abs(sentiment_change) * 0.5)
        
        # 반전 신호 보너스
        if conditions.get("reversal_signal", False):
            base_confidence += 0.1
        
        # 다이버전스 보너스
        if conditions.get("rsi_divergence", False):
            base_confidence += 0.08
        
        # 거래량 보너스
        if conditions.get("volume_sufficient", False):
            volume_ratio = indicators.get("volume_ratio", 1.0)
            base_confidence += min(0.08, (volume_ratio - 1.0) * 0.1)
        
        return max(0.0, min(1.0, base_confidence))

    def _generate_detailed_entry_reason(self, 
                                       conditions: Dict[str, bool], 
                                       indicators: Dict[str, float],
                                       sentiment: float,
                                       entry_scenario: str) -> str:
        """상세한 진입 이유 생성"""
        reasons = []
        
        # 시나리오별 기본 설명
        if entry_scenario == "bb_lower_reversal":
            reasons.append("BB하단반등")
        elif entry_scenario == "sentiment_rebound":
            reasons.append("감정반등")
        
        # 추가 조건들
        bb_position = indicators.get("bb_position", 0.5)
        reasons.append(f"BB위치({bb_position:.2f})")
        
        if conditions.get("rsi_extreme"):
            rsi = indicators.get("rsi", 50)
            reasons.append(f"RSI극값({rsi:.0f})")
        
        if conditions.get("volatility_spike"):
            vol_ratio = indicators.get("volatility_ratio", 1.0)
            reasons.append(f"변동성급증({vol_ratio:.1f}x)")
        
        if conditions.get("sentiment_extreme"):
            reasons.append(f"감정극값({sentiment:.2f})")
        
        if conditions.get("reversal_signal"):
            reasons.append("반전신호")
        
        if conditions.get("rsi_divergence"):
            reasons.append("RSI다이버전스")
        
        return ", ".join(reasons) if reasons else "향상된역추세조건충족"

    def get_regime_preferences(self) -> Dict[str, float]:
        """시장 체제별 선호도 (전략 C 맞춤)"""
        return {
            "bull": 0.4,        # 상승장에서 역추세
            "bear": 0.6,        # 하락장에서 반등 노리기
            "sideways": 0.8,    # 횡보장 최적
            "high_volatility": 0.9,  # 고변동성 선호
            "crisis": 0.7       # 위기 시 반등 기회
        }