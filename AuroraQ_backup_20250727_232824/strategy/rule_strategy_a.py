"""
Enhanced RuleStrategyA - EMA 크로스오버 + ADX with Caching
향상된 캐싱과 필터링이 적용된 전략 A
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from strategy.base_rule_strategy import BaseRuleStrategy
from utils.logger import get_logger
from config.rule_param_loader import get_rule_params

logger = get_logger("RuleStrategyA")


class RuleStrategyA(BaseRuleStrategy):
    """
    향상된 RuleStrategyA - EMA 크로스오버 + ADX 트렌드 강도 전략
    
    개선사항:
    - 향상된 지표 캐싱 시스템 활용
    - 강화된 진입 조건 필터
    - 표준화된 메트릭 사용
    - 중복 계산 제거
    
    진입 조건:
    1. 단기 EMA > 장기 EMA (강화된 조건)
    2. ADX > 동적 임계값 (더 엄격한 기준)
    3. 향상된 필터 시스템 통과
    
    청산 조건:
    1. 동적 TP/SL 도달
    2. 필터 조건 악화
    3. 신뢰도 기반 조기 청산
    """

    def __init__(self):
        super().__init__(name="RuleStrategyA")
        
        # 파라미터 로드
        params = get_rule_params("RuleA")
        
        # 지표 설정 (더 엄격하게 조정)
        self.ema_short_len = params.get("ema_short_len", 8)      # 5 → 8 (더 안정적)
        self.ema_long_len = params.get("ema_long_len", 21)       # 13 → 21 (트렌드 확인)
        self.adx_window = params.get("adx_window", 14)
        self.adx_base_threshold = params.get("adx_threshold", 25) # 15 → 25 (더 강한 트렌드 요구)
        
        # 리스크 관리 (보수적으로 조정)
        self.tp_base = params.get("take_profit_pct", 0.025)      # 2.0% → 2.5%
        self.sl_base = params.get("stop_loss_pct", 0.015)       # 1.2% → 1.5%
        self.volatility_window = params.get("volatility_window", 20)
        self.max_hold_bars = params.get("max_hold_bars", 10)     # 12 → 10 (더 빠른 회전)
        
        # 동적 조정 파라미터 (보수적으로)
        self.vol_multiplier_tp = params.get("vol_multiplier_tp", 1.5)   # 2 → 1.5
        self.vol_multiplier_sl = params.get("vol_multiplier_sl", 1.2)   # 1.5 → 1.2
        self.vol_multiplier_adx = params.get("vol_multiplier_adx", 10)  # 8 → 10
        
        # 필터 설정 (강화)
        self.min_volume_ratio = params.get("min_volume_ratio", 1.0)     # 0.8 → 1.0
        self.min_ema_diff_pct = params.get("min_ema_diff_pct", 0.003)   # 0.002 → 0.003
        self.enable_ema_exit = params.get("enable_ema_exit", True)      # False → True
        
        # 캐시 키 정의
        self.cache_keys = {
            "ema_short": f"ema_short_{self.ema_short_len}",
            "ema_long": f"ema_long_{self.ema_long_len}",
            "adx": f"adx_{self.adx_window}",
            "volatility": f"volatility_{self.volatility_window}"
        }
        
        logger.info(
            f"향상된 RuleStrategyA 초기화: "
            f"EMA({self.ema_short_len}/{self.ema_long_len}), "
            f"ADX({self.adx_window}≥{self.adx_base_threshold}), "
            f"TP/SL({self.tp_base:.1%}/{self.sl_base:.1%})"
        )

    def calculate_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 지표 계산 (캐시 활용)"""
        try:
            if len(price_data) < max(self.ema_long_len, self.volatility_window, self.adx_window + 1):
                return {}
            
            indicators = {}
            
            # 1. EMA 계산 (캐시 활용)
            ema_short = self.get_cached_indicator(
                "ema", price_data, period=self.ema_short_len
            )
            ema_long = self.get_cached_indicator(
                "ema", price_data, period=self.ema_long_len
            )
            
            if ema_short is not None and ema_long is not None:
                indicators["ema_short"] = float(ema_short.iloc[-1])
                indicators["ema_long"] = float(ema_long.iloc[-1])
                indicators["ema_diff"] = indicators["ema_short"] - indicators["ema_long"]
                indicators["ema_ratio"] = indicators["ema_short"] / indicators["ema_long"] if indicators["ema_long"] > 0 else 1.0
                indicators["ema_diff_pct"] = indicators["ema_diff"] / indicators["ema_long"] if indicators["ema_long"] > 0 else 0.0
            
            # 2. ADX 계산 (캐시 활용)
            adx_series = self.get_cached_indicator(
                "adx", price_data, period=self.adx_window
            )
            
            if adx_series is not None:
                indicators["adx"] = float(adx_series.iloc[-1])
            
            # 3. 변동성 계산 (캐시 활용 - 간단한 방법)
            returns = price_data["close"].pct_change().dropna()
            if len(returns) >= self.volatility_window:
                volatility = float(returns.tail(self.volatility_window).std())
                indicators["volatility"] = volatility
            
            # 4. 추가 지표 (캐시 활용)
            if "volume" in price_data.columns and len(price_data) >= 20:
                avg_volume = price_data["volume"].tail(20).mean()
                current_volume = price_data["volume"].iloc[-1]
                indicators["volume_ratio"] = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
            
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
            required_indicators = ["ema_short", "ema_long", "adx", "volatility"]
            if not all(ind in indicators for ind in required_indicators):
                logger.debug("필수 지표 부족")
                return None
            
            # 현재 값들
            current_price = self.safe_last(price_data, "close")
            current_time = pd.Timestamp.now()
            
            # 동적 임계값 계산
            tp, sl, adx_threshold = self._calculate_dynamic_thresholds(indicators["volatility"])
            
            # 강화된 진입 조건
            conditions = {
                # 1. EMA 크로스오버 (강화된 조건)
                "ema_bullish": indicators["ema_short"] > indicators["ema_long"],
                "ema_diff_sufficient": abs(indicators["ema_diff_pct"]) >= self.min_ema_diff_pct,
                
                # 2. ADX 트렌드 강도 (더 엄격)
                "adx_strong": indicators["adx"] >= adx_threshold,
                
                # 3. 볼륨 확인 (더 엄격)
                "volume_sufficient": indicators.get("volume_ratio", 1.0) >= self.min_volume_ratio,
                
                # 4. 변동성 적정 범위
                "volatility_ok": 0.005 <= indicators["volatility"] <= 0.08,
                
                # 5. 가격 모멘텀 (추가 조건)
                "momentum_positive": indicators.get("price_change", 0) >= -0.001  # 큰 하락 중이 아님
            }
            
            # 모든 조건 검증
            if not all(conditions.values()):
                failed_conditions = [k for k, v in conditions.items() if not v]
                logger.debug(f"진입 조건 실패: {failed_conditions}")
                return None
            
            # 리스크/리워드 비율 확인 (더 엄격)
            risk_reward_ratio = tp / sl
            if risk_reward_ratio < 1.5:  # 1.2 → 1.5
                logger.debug(f"리스크/리워드 비율 부족: {risk_reward_ratio:.2f}")
                return None
            
            # 신뢰도 계산 (다중 요소 고려)
            confidence = self._calculate_enhanced_confidence(indicators, conditions)
            
            # 진입 정보 생성
            entry_info = {
                "side": "LONG",
                "confidence": confidence,
                "reason": self._generate_detailed_entry_reason(conditions, indicators),
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
                f"EMA_diff: {indicators['ema_diff_pct']:.3%} | "
                f"ADX: {indicators['adx']:.1f} | "
                f"Vol: {indicators['volatility']:.3%}"
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
            
            # 동적 TP/SL 가져오기
            indicators = self.calculate_indicators(price_data)
            if indicators:
                tp, sl, _ = self._calculate_dynamic_thresholds(indicators.get("volatility", 0.02))
            else:
                tp, sl = self.tp_base, self.sl_base
            
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
            ]
            
            # EMA 역전 체크 (활성화된 경우)
            if self.enable_ema_exit and indicators and "ema_short" in indicators:
                ema_short = indicators["ema_short"]
                ema_long = indicators["ema_long"]
                ema_reversal = ema_short < ema_long * 0.998  # 더 엄격한 기준
                
                exit_conditions.append(
                    (ema_reversal, f"EMA역전 (Short:{ema_short:.2f} < Long:{ema_long:.2f})")
                )
            
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

    def _calculate_dynamic_thresholds(self, volatility: float) -> Tuple[float, float, float]:
        """변동성 기반 동적 임계값 계산 (보수적 조정)"""
        # 변동성 정규화 (극단값 제한)
        vol_clamped = max(0.005, min(volatility, 0.1))
        
        # 보수적 조정
        tp = self.tp_base * (1 + vol_clamped * self.vol_multiplier_tp)
        sl = self.sl_base * (1 + vol_clamped * self.vol_multiplier_sl)
        adx_threshold = self.adx_base_threshold * (1 + vol_clamped * self.vol_multiplier_adx)
        
        # 더 엄격한 범위 제한
        tp = max(0.015, min(tp, 0.08))   # 1.5% ~ 8%
        sl = max(0.008, min(sl, 0.04))   # 0.8% ~ 4%
        adx_threshold = max(20, min(adx_threshold, 40))  # 20~40
        
        return tp, sl, adx_threshold

    def _calculate_enhanced_confidence(self, 
                                     indicators: Dict[str, float], 
                                     conditions: Dict[str, bool]) -> float:
        """향상된 신뢰도 계산"""
        base_confidence = 0.5
        
        # 조건별 가중치
        if conditions.get("ema_bullish"):
            ema_strength = min(0.2, abs(indicators.get("ema_diff_pct", 0)) * 50)
            base_confidence += ema_strength
        
        if conditions.get("adx_strong"):
            adx_bonus = min(0.2, (indicators.get("adx", 0) - self.adx_base_threshold) / 50)
            base_confidence += adx_bonus
        
        if conditions.get("volume_sufficient"):
            volume_ratio = indicators.get("volume_ratio", 1.0)
            volume_bonus = min(0.15, max(0, (volume_ratio - 1.0) * 0.1))
            base_confidence += volume_bonus
        
        # 변동성 보너스 (적정 범위)
        volatility = indicators.get("volatility", 0.02)
        if 0.01 <= volatility <= 0.03:  # 이상적인 변동성 범위
            base_confidence += 0.1
        
        # 모멘텀 보너스
        price_change = indicators.get("price_change", 0)
        if price_change > 0.001:  # 상승 모멘텀
            base_confidence += min(0.1, price_change * 10)
        
        return max(0.0, min(1.0, base_confidence))

    def _generate_detailed_entry_reason(self, 
                                       conditions: Dict[str, bool], 
                                       indicators: Dict[str, float]) -> str:
        """상세한 진입 이유 생성"""
        reasons = []
        
        if conditions.get("ema_bullish"):
            ema_diff_pct = indicators.get("ema_diff_pct", 0)
            reasons.append(f"EMA상향크로스({ema_diff_pct:.2%})")
        
        if conditions.get("adx_strong"):
            adx = indicators.get("adx", 0)
            reasons.append(f"강한트렌드(ADX:{adx:.1f})")
        
        if indicators.get("volume_ratio", 1.0) > 1.5:
            reasons.append("거래량급증")
        
        if indicators.get("volatility", 0.02) > 0.02:
            reasons.append("활발한변동성")
        
        if indicators.get("price_change", 0) > 0.002:
            reasons.append("상승모멘텀")
        
        return ", ".join(reasons) if reasons else "향상된조건충족"

    def get_regime_preferences(self) -> Dict[str, float]:
        """시장 체제별 선호도 (전략 A 맞춤)"""
        return {
            "bull": 0.9,        # 상승장 선호
            "bear": 0.2,        # 하락장 회피
            "sideways": 0.7,    # 횡보장 보통
            "high_volatility": 0.3,  # 고변동성 회피
            "crisis": 0.1       # 위기 상황 회피
        }