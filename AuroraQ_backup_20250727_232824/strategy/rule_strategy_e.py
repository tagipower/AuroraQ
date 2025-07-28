"""
Enhanced RuleStrategyE - 브레이크아웃 + 레인지 확장 with Caching
향상된 캐싱과 필터링이 적용된 전략 E
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from strategy.base_rule_strategy import BaseRuleStrategy
from utils.logger import get_logger
from config.rule_param_loader import get_rule_params

logger = get_logger("RuleStrategyE")


class RuleStrategyE(BaseRuleStrategy):
    """
    향상된 RuleStrategyE - 브레이크아웃 + 레인지 확장 전략
    
    개선사항:
    - 향상된 지표 캐싱 시스템 활용
    - 강화된 진입 조건 필터
    - 표준화된 메트릭 사용
    - 중복 계산 제거
    
    진입 조건:
    1. 다중 기간 브레이크아웃 (더 엄격한 확인)
    2. 레인지 수축 후 확장 (더 신중한 판단)
    3. 거래량 스파이크 + 추세 확인 (더 강한 신호 요구)
    4. 모멘텀 컨펌 (복수 지표 일치)
    5. 향상된 필터 시스템 통과
    
    청산 조건:
    1. 동적 TP/SL + 트레일링 스톱
    2. 모멘텀 소진 감지
    3. 브레이크아웃 실패 보호
    4. 신뢰도 기반 조기 청산
    """

    def __init__(self):
        super().__init__(name="RuleStrategyE")
        
        # 파라미터 로드
        params = get_rule_params("RuleE")
        
        # 브레이크아웃 설정 (더 엄격하게)
        self.breakout_window_short = params.get("breakout_window_short", 8)    # 5 → 8 (더 안정적)
        self.breakout_window_medium = params.get("breakout_window_medium", 15) # 10 → 15
        self.breakout_window_long = params.get("breakout_window_long", 30)     # 20 → 30
        self.breakout_buffer = params.get("breakout_buffer", 0.0015)  # 0.0001 → 0.0015 (더 확실한 돌파)
        
        # 레인지/변동성 설정 (더 엄격하게)
        self.range_window = params.get("range_window", 15)                    # 10 → 15 (복원)
        self.range_std_threshold = params.get("range_std_threshold", 0.002)   # 0.0015 → 0.002 (복원)
        self.range_atr_threshold = params.get("range_atr_threshold", 0.004)   # 0.003 → 0.004 (복원)
        self.squeeze_window = params.get("squeeze_window", 25)                # 20 → 25 (복원)
        self.expansion_ratio = params.get("expansion_ratio", 1.8)             # 1.3 → 1.8 (복원, 더 강한 확장 요구)
        
        # 거래량 설정 (더 엄격하게)
        self.volume_window = params.get("volume_window", 8)                   # 5 → 8 (복원)
        self.volume_spike_ratio = params.get("volume_spike_ratio", 1.5)       # 1.1 → 1.5 (더 강한 스파이크 요구)
        self.volume_ma_window = params.get("volume_ma_window", 15)            # 10 → 15 (복원)
        self.obv_window = params.get("obv_window", 12)                        # 10 → 12 (복원)
        
        # 모멘텀 설정 (더 보수적으로)
        self.rsi_period = params.get("rsi_window", 14)
        self.rsi_breakout_threshold = params.get("rsi_breakout_threshold", 55) # 50 → 55 (더 확실한 상승세)
        self.macd_fast = params.get("macd_fast", 12)
        self.macd_slow = params.get("macd_slow", 26)
        self.macd_signal = params.get("macd_signal", 9)
        
        # 시장 구조 (더 엄격)
        self.support_resistance_window = params.get("support_resistance_window", 100)
        self.sr_touch_threshold = params.get("sr_touch_threshold", 0.0008)  # 0.001 → 0.0008 (더 정밀)
        self.trend_filter_window = params.get("trend_filter_window", 50)
        
        # 리스크 관리 (보수적으로 조정)
        self.tp_base = params.get("take_profit_pct", 0.03)                    # 2.5% → 3%
        self.sl_base = params.get("stop_loss_pct", 0.018)                     # 1.5% → 1.8%
        self.max_hold_bars = params.get("max_hold_bars", 6)                   # 8 → 6 (더 빠른 회전)
        self.trailing_stop_activation = params.get("trailing_stop_activation", 0.015)  # 1.2% → 1.5%
        self.trailing_stop_distance = params.get("trailing_stop_distance", 0.008)     # 0.6% → 0.8%
        
        # 동적 조정 (보수적으로)
        self.vol_multiplier_tp = params.get("vol_multiplier_tp", 2.5)         # 3 → 2.5
        self.vol_multiplier_sl = params.get("vol_multiplier_sl", 1.8)         # 2 → 1.8
        self.breakout_strength_multiplier = params.get("breakout_strength_multiplier", 1.5)  # 2 → 1.5
        
        # 재진입 관리 (더 엄격)
        self.enable_reentry = params.get("enable_reentry", False)             # True → False (재진입 비활성화)
        self.reentry_cooldown = params.get("reentry_cooldown", 600)           # 300 → 600초
        self.max_reentries = params.get("max_reentries", 1)                   # 2 → 1
        
        # 추가 필터 설정
        self.momentum_window = params.get("momentum_window", 5)
        self.momentum_threshold = params.get("momentum_threshold", 0.005)     # 더 강한 모멘텀 요구
        self.min_volume_ratio = params.get("min_volume_ratio", 1.2)           # 최소 거래량 요구
        self.min_confidence = params.get("min_confidence", 0.75)              # 최소 신뢰도
        
        # 상태 추적
        self.entry_metrics = {}
        self.breakout_levels = {}
        self.highest_price = 0.0
        self.reentry_count = 0
        self.last_exit_time = None
        self._last_atr = 0.0
        self._support_resistance_levels = []
        
        logger.info(
            f"향상된 RuleStrategyE 초기화: "
            f"Breakout({self.breakout_window_short}/{self.breakout_window_medium}/{self.breakout_window_long}), "
            f"Range({self.range_std_threshold}), "
            f"TP/SL({self.tp_base:.1%}/{self.sl_base:.1%})"
        )

    def calculate_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 지표 계산 (캐시 활용)"""
        try:
            if len(price_data) < max(self.breakout_window_long, self.squeeze_window, self.support_resistance_window // 2):
                return {}
            
            indicators = {}
            
            # 1. 브레이크아웃 메트릭 계산 (캐시 활용)
            current_price = price_data["close"].iloc[-1]
            
            # 다중 기간 고점 (캐시 활용)
            high_short_series = price_data["high"].tail(self.breakout_window_short + 1)
            high_medium_series = price_data["high"].tail(self.breakout_window_medium + 1)
            high_long_series = price_data["high"].tail(self.breakout_window_long + 1)
            
            if len(high_short_series) >= self.breakout_window_short:
                high_short = high_short_series.iloc[:-1].max()  # 현재봉 제외
                high_medium = high_medium_series.iloc[:-1].max() if len(high_medium_series) >= self.breakout_window_medium else high_short
                high_long = high_long_series.iloc[:-1].max() if len(high_long_series) >= self.breakout_window_long else high_medium
                
                # 브레이크아웃 여부
                indicators["breakout_short"] = current_price > high_short * (1 + self.breakout_buffer)
                indicators["breakout_medium"] = current_price > high_medium * (1 + self.breakout_buffer)
                indicators["breakout_long"] = current_price > high_long * (1 + self.breakout_buffer)
                
                # 브레이크아웃 강도
                breakout_count = sum([indicators["breakout_short"], indicators["breakout_medium"], indicators["breakout_long"]])
                indicators["breakout_strength"] = breakout_count / 3.0
                
                # 브레이크아웃 레벨 저장
                self.breakout_levels = {
                    "short": high_short,
                    "medium": high_medium,
                    "long": high_long
                }
            else:
                # 데이터가 부족한 경우 기본값 설정
                indicators["breakout_short"] = False
                indicators["breakout_medium"] = False
                indicators["breakout_long"] = False
                indicators["breakout_strength"] = 0.0
                self.breakout_levels = {
                    "short": current_price,
                    "medium": current_price,
                    "long": current_price
                }
            
            # 2. 레인지/변동성 메트릭 (캐시 활용)
            returns = price_data["close"].pct_change().dropna()
            if len(returns) >= self.squeeze_window:
                # 표준편차 기반 변동성
                recent_std = returns.tail(self.range_window).std()
                historical_std = returns.tail(self.squeeze_window).std()
                
                indicators["range_std"] = float(recent_std)
                indicators["is_range"] = recent_std < self.range_std_threshold
                indicators["volatility_ratio"] = float(recent_std / (historical_std + 1e-6))
                indicators["range_expansion"] = indicators["volatility_ratio"] > self.expansion_ratio
            else:
                # 데이터가 부족한 경우 기본값 설정
                indicators["range_std"] = 0.02  # 적당한 변동성
                indicators["is_range"] = True   # 레인지 상태로 간주
                indicators["volatility_ratio"] = 1.0  # 중립 비율
                indicators["range_expansion"] = False
            
            # 3. ATR 계산 (캐시 활용)
            atr_series = self.get_cached_indicator(
                "atr", price_data, period=14
            )
            
            if atr_series is not None:
                try:
                    # 안전한 ATR 값 추출 (StrategyAdapter의 _safe_float 사용)
                    if hasattr(self, '_safe_float'):
                        current_atr = self._safe_float(atr_series, 0.0)
                    else:
                        # 폴백 로직
                        if hasattr(atr_series, 'iloc') and len(atr_series) > 0:
                            current_atr = float(atr_series.iloc[-1])
                        elif hasattr(atr_series, '__iter__') and not isinstance(atr_series, str):
                            current_atr = float(list(atr_series)[-1]) if atr_series else 0.0
                        else:
                            current_atr = float(atr_series)
                    
                    atr_pct = current_atr / current_price if current_price > 0 else 0.0
                    
                    indicators["atr"] = current_atr
                    indicators["atr_pct"] = atr_pct
                    indicators["is_low_atr"] = atr_pct < self.range_atr_threshold
                    self._last_atr = current_atr
                except Exception as e:
                    logger.warning(f"ATR 계산 오류: {e}")
                    indicators["atr"] = 0.0
                    indicators["atr_pct"] = 0.0
                    indicators["is_low_atr"] = False
            
            # 4. 볼린저 밴드 스퀴즈 (캐시 활용)
            bb_upper = self.get_cached_indicator(
                "bb_upper", price_data, period=20, std=2
            )
            bb_lower = self.get_cached_indicator(
                "bb_lower", price_data, period=20, std=2
            )
            
            if bb_upper is not None and bb_lower is not None:
                try:
                    # 안전한 값 추출 (StrategyAdapter의 _safe_float 사용)
                    if hasattr(self, '_safe_float'):
                        upper_val = self._safe_float(bb_upper, 0.0)
                        lower_val = self._safe_float(bb_lower, 0.0)
                    else:
                        # 폴백 로직
                        if hasattr(bb_upper, 'iloc') and len(bb_upper) > 0:
                            upper_val = float(bb_upper.iloc[-1])
                        elif hasattr(bb_upper, '__iter__') and not isinstance(bb_upper, str):
                            upper_val = float(list(bb_upper)[-1]) if bb_upper else 0.0
                        else:
                            upper_val = float(bb_upper)
                        
                        if hasattr(bb_lower, 'iloc') and len(bb_lower) > 0:
                            lower_val = float(bb_lower.iloc[-1])
                        elif hasattr(bb_lower, '__iter__') and not isinstance(bb_lower, str):
                            lower_val = float(list(bb_lower)[-1]) if bb_lower else 0.0
                        else:
                            lower_val = float(bb_lower)
                    
                    current_price = float(price_data["close"].iloc[-1])
                    bb_width = (upper_val - lower_val) / current_price
                    
                    indicators["bb_squeeze"] = bb_width < self.range_atr_threshold * 0.8
                    indicators["bb_width_ratio"] = float(bb_width)
                except Exception as e:
                    logger.warning(f"볼린저 밴드 계산 오류: {e}")
                    indicators["bb_squeeze"] = False
                    indicators["bb_width_ratio"] = 0.0
            
            # 5. 거래량 메트릭
            if "volume" in price_data.columns and len(price_data) >= self.volume_ma_window:
                current_volume = price_data["volume"].iloc[-1]
                
                # 거래량 스파이크
                short_ma = price_data["volume"].tail(self.volume_window).mean()
                long_ma = price_data["volume"].tail(self.volume_ma_window).mean()
                
                indicators["volume_spike"] = current_volume > short_ma * self.volume_spike_ratio
                indicators["volume_ratio_short"] = float(current_volume / (short_ma + 1e-6))
                indicators["volume_ratio_long"] = float(current_volume / (long_ma + 1e-6))
                indicators["volume_sufficient"] = indicators["volume_ratio_short"] >= self.min_volume_ratio
                
                # 거래량 추세
                if len(price_data) >= 10:
                    volume_trend = np.polyfit(range(10), price_data["volume"].tail(10).values, 1)[0]
                    indicators["volume_trend"] = volume_trend > 0
                    indicators["volume_acceleration"] = float(volume_trend)
            else:
                # volume 데이터가 없거나 부족한 경우 기본값 설정
                indicators["volume_spike"] = False
                indicators["volume_ratio_short"] = 1.0  # 기본값
                indicators["volume_ratio_long"] = 1.0
                indicators["volume_sufficient"] = True  # 거래량 조건을 우회
                indicators["volume_trend"] = False
                indicators["volume_acceleration"] = 0.0
            
            # 6. RSI 계산 (캐시 활용)
            rsi_series = self.get_cached_indicator(
                "rsi", price_data, period=self.rsi_period
            )
            
            if rsi_series is not None:
                try:
                    # 안전한 RSI 값 추출 (StrategyAdapter의 _safe_float 사용)
                    if hasattr(self, '_safe_float'):
                        rsi = self._safe_float(rsi_series, 50.0)
                    else:
                        # 폴백 로직
                        if hasattr(rsi_series, 'iloc') and len(rsi_series) > 0:
                            rsi = float(rsi_series.iloc[-1])
                        elif hasattr(rsi_series, '__iter__') and not isinstance(rsi_series, str):
                            rsi = float(list(rsi_series)[-1]) if rsi_series else 50.0
                        else:
                            rsi = float(rsi_series)
                    
                    indicators["rsi"] = rsi
                    indicators["rsi_bullish"] = rsi > self.rsi_breakout_threshold
                except Exception as e:
                    logger.warning(f"RSI 계산 오류: {e}")
                    indicators["rsi"] = 50.0
                    indicators["rsi_bullish"] = False
            else:
                # RSI 데이터가 없는 경우 기본값 설정
                indicators["rsi"] = 50.0
                indicators["rsi_bullish"] = False
            
            # 7. MACD 계산 (캐시 활용) - 개별 지표 요청으로 변경
            macd_line = self.get_cached_indicator(
                "macd_line", price_data, 
                fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal
            )
            macd_signal = self.get_cached_indicator(
                "macd_signal", price_data, 
                fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal
            )
            
            if macd_line is not None and macd_signal is not None:
                try:
                    # 안전한 MACD 값 추출 (StrategyAdapter의 _safe_float 사용)
                    if hasattr(self, '_safe_float'):
                        macd_val = self._safe_float(macd_line, 0.0)
                        signal_val = self._safe_float(macd_signal, 0.0)
                    else:
                        # 폴백 로직
                        if hasattr(macd_line, 'iloc') and len(macd_line) > 0:
                            macd_val = float(macd_line.iloc[-1])
                        elif hasattr(macd_line, '__len__') and len(macd_line) > 0:
                            macd_val = float(macd_line.iloc[-1] if hasattr(macd_line, 'iloc') else list(macd_line)[-1])
                        else:
                            macd_val = 0.0
                        
                        if hasattr(macd_signal, 'iloc') and len(macd_signal) > 0:
                            signal_val = float(macd_signal.iloc[-1])
                        elif hasattr(macd_signal, '__len__') and len(macd_signal) > 0:
                            signal_val = float(macd_signal.iloc[-1] if hasattr(macd_signal, 'iloc') else list(macd_signal)[-1])
                        else:
                            signal_val = 0.0
                    
                    macd_histogram = macd_val - signal_val
                    
                    indicators["macd_bullish"] = macd_val > signal_val
                    indicators["macd_histogram"] = macd_histogram
                    indicators["macd_positive"] = macd_histogram > 0
                except Exception as e:
                    logger.warning(f"MACD 계산 오류: {e}")
                    indicators["macd_bullish"] = False
                    indicators["macd_histogram"] = 0.0
                    indicators["macd_positive"] = False
            
            # 8. 가격 모멘텀
            if len(price_data) >= 21:  # 20일 momentum을 위해 21개 필요
                try:
                    current_close = float(price_data["close"].iloc[-1])
                    close_5 = float(price_data["close"].iloc[-6])
                    close_10 = float(price_data["close"].iloc[-11])
                    close_20 = float(price_data["close"].iloc[-21])
                    
                    momentum_5 = (current_close - close_5) / close_5 if close_5 > 0 else 0.0
                    momentum_10 = (current_close - close_10) / close_10 if close_10 > 0 else 0.0
                    momentum_20 = (current_close - close_20) / close_20 if close_20 > 0 else 0.0
                    
                    indicators["momentum_5"] = float(momentum_5)
                    indicators["momentum_10"] = float(momentum_10)
                    indicators["momentum_20"] = float(momentum_20)
                    indicators["momentum_increasing"] = momentum_5 > momentum_10 > momentum_20
                    indicators["momentum_strong"] = momentum_5 > self.momentum_threshold
                except Exception as e:
                    logger.warning(f"모멘텀 계산 오류: {e}")
                    indicators["momentum_5"] = 0.0
                    indicators["momentum_10"] = 0.0
                    indicators["momentum_20"] = 0.0
                    indicators["momentum_increasing"] = False
                    indicators["momentum_strong"] = False
            
            # 9. 지지/저항 레벨 (간단화)
            if len(price_data) >= 50:  # 전체 구간이 아닌 최근 50봉만 사용
                sr_levels = self._identify_support_resistance(price_data.tail(50))
                indicators["support_resistance_levels"] = sr_levels
                indicators["near_resistance"] = self._check_near_levels(current_price, sr_levels)
            
            indicators["current_price"] = float(current_price)
            
            return indicators
            
        except Exception as e:
            logger.error(f"지표 계산 오류: {e}")
            return {}

    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """향상된 진입 조건 확인"""
        try:
            indicators = self.calculate_indicators(price_data)
            
            # 필수 지표 확인
            required_indicators = ["breakout_strength", "volatility_ratio", "volume_ratio_short", "rsi"]
            missing_indicators = [ind for ind in required_indicators if ind not in indicators]
            if missing_indicators:
                logger.warning(f"RuleStrategyE 필수 지표 부족: {missing_indicators}, 현재 지표: {list(indicators.keys())}")
                return None
            
            # 디버그: 지표 값 확인
            logger.warning(f"RuleStrategyE 지표 값: breakout_strength={indicators['breakout_strength']}, "
                          f"volatility_ratio={indicators['volatility_ratio']}, "
                          f"volume_ratio_short={indicators['volume_ratio_short']}, "
                          f"rsi={indicators['rsi']}")
            
            # 현재 값들
            current_price = self.safe_last(price_data, "close")
            current_time = pd.Timestamp.now()
            
            # 재진입 쿨다운 체크
            if self._check_reentry_cooldown(current_time):
                logger.debug("재진입 쿨다운 중")
                return None
            
            # 동적 임계값 계산
            tp, sl = self._calculate_dynamic_thresholds(indicators)
            
            # 강화된 진입 조건
            conditions = {
                # 1. 브레이크아웃 조건 (핵심)
                "breakout_any": indicators.get("breakout_short", False) or 
                               indicators.get("breakout_medium", False) or 
                               indicators.get("breakout_long", False),
                
                "breakout_strong": indicators.get("breakout_strength", 0) >= 0.67,  # 3개 중 2개 이상
                
                # 2. 레인지/변동성 조건 (둘 다 만족 필요)
                "range_condition": indicators.get("is_range", False) or indicators.get("is_low_atr", False),
                "range_expansion": indicators.get("range_expansion", False),
                
                # 3. 거래량 조건 (강화)
                "volume_spike": indicators.get("volume_spike", False),
                "volume_sufficient": indicators.get("volume_sufficient", False),
                "volume_trend": indicators.get("volume_trend", False),
                
                # 4. 모멘텀 조건 (강화)
                "rsi_bullish": indicators.get("rsi_bullish", False),
                "macd_bullish": indicators.get("macd_bullish", False),
                "momentum_strong": indicators.get("momentum_strong", False),
                "momentum_increasing": indicators.get("momentum_increasing", False),
                
                # 5. 추가 확인 조건
                "volatility_adequate": 0.01 <= indicators.get("volatility_ratio", 1.0) <= 3.0,
                "not_overbought": indicators.get("rsi", 50) < 75,  # RSI 과매수 회피
            }
            
            # 핵심 조건 (모두 만족 필요)
            core_conditions = [
                conditions["breakout_any"],
                conditions["range_condition"] and conditions["range_expansion"],  # AND 조건
                conditions["volume_spike"] and conditions["volume_sufficient"],    # AND 조건
            ]
            
            # 모멘텀 조건 (최소 3개 중 2개 만족)
            momentum_conditions = [
                conditions["rsi_bullish"],
                conditions["macd_bullish"],
                conditions["momentum_strong"],
                conditions["momentum_increasing"]
            ]
            
            # 추가 조건 (모두 만족)
            additional_conditions = [
                conditions["breakout_strong"],
                conditions["volume_trend"],
                conditions["volatility_adequate"],
                conditions["not_overbought"]
            ]
            
            # 디버그: 조건 체크 결과
            logger.debug(f"조건 체크 - 핵심: {core_conditions} (all={all(core_conditions)})")
            logger.debug(f"조건 체크 - 모멘텀: {momentum_conditions} (sum={sum(momentum_conditions)}/4 >= 2)")
            logger.debug(f"조건 체크 - 추가: {additional_conditions} (all={all(additional_conditions)})")
            
            # 최종 평가: 핵심 + 모멘텀(2/4) + 추가 조건 모두
            if (all(core_conditions) and 
                sum(momentum_conditions) >= 2 and 
                all(additional_conditions)):
                
                # 신뢰도 계산
                confidence = self._calculate_enhanced_confidence(indicators, conditions)
                
                # 최소 신뢰도 확인 (더 엄격)
                if confidence < self.min_confidence:
                    logger.debug(f"신뢰도 부족: {confidence:.3f} < {self.min_confidence}")
                    return None
                
                # 리스크/리워드 비율 확인
                risk_reward_ratio = tp / sl
                if risk_reward_ratio < 1.7:  # 더 엄격한 기준
                    logger.debug(f"리스크/리워드 비율 부족: {risk_reward_ratio:.2f}")
                    return None
                
                # 진입 메트릭 저장
                self.entry_metrics = {
                    "breakout_strength": indicators["breakout_strength"],
                    "breakout_levels": self.breakout_levels.copy(),
                    "volatility_ratio": indicators["volatility_ratio"],
                    "volume_ratio": indicators["volume_ratio_short"],
                    "rsi": indicators["rsi"],
                    "momentum_5": indicators.get("momentum_5", 0),
                    "tp": tp,
                    "sl": sl
                }
                
                # 최고가 추적 초기화
                self.highest_price = current_price
                
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
                    f"브레이크아웃: {indicators['breakout_strength']:.2f} | "
                    f"Volume: {indicators['volume_ratio_short']:.1f}x | "
                    f"TP: {tp:.1%}, SL: {sl:.1%}"
                )
                
                return entry_info
            else:
                # 디버그: 신호가 생성되지 않은 이유
                logger.debug(f"RuleStrategyE 신호 없음 - "
                           f"핵심조건: {all(core_conditions)}, "
                           f"모멘텀조건: {sum(momentum_conditions)}/4>=2 ({sum(momentum_conditions)>=2}), "
                           f"추가조건: {all(additional_conditions)}")
            
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
            
            # 최고가 업데이트 및 드로다운 계산
            self.highest_price = max(self.highest_price, current_price)
            high_water_pnl = (self.highest_price - entry_price) / entry_price
            drawdown = (self.highest_price - current_price) / self.highest_price
            
            # 보유 시간
            holding_time_seconds = position.holding_time.total_seconds()
            holding_bars = holding_time_seconds / 300  # 5분봉 기준
            
            # 동적 TP/SL
            tp = self.entry_metrics.get("tp", self.tp_base)
            sl = self.entry_metrics.get("sl", self.sl_base)
            
            # 지표 재계산
            indicators = self.calculate_indicators(price_data)
            
            # 청산 조건들 (우선순위 순)
            exit_conditions = [
                # 1. 손절/익절
                (pnl_ratio <= -sl, f"손절 ({pnl_ratio:.1%})"),
                (pnl_ratio >= tp, f"익절 ({pnl_ratio:.1%})"),
                
                # 2. 향상된 트레일링 스톱
                (high_water_pnl >= self.trailing_stop_activation and 
                 drawdown >= self.trailing_stop_distance,
                 f"트레일링스톱 (최고:{high_water_pnl:.1%}, 드로다운:{drawdown:.1%})"),
                
                # 3. 시간 초과
                (holding_bars >= self.max_hold_bars, f"시간초과 ({holding_bars:.0f}봉)"),
                
                # 4. 신뢰도 기반 조기 청산
                (hasattr(position, 'confidence') and 
                 position.confidence < 0.5 and pnl_ratio < -0.008,
                 f"낮은신뢰도청산 (신뢰도:{getattr(position, 'confidence', 0):.2f})"),
            ]
            
            # 5. 브레이크아웃 실패 (더 엄격한 기준)
            if self.breakout_levels:
                short_level = self.breakout_levels.get("short", entry_price)
                if current_price < short_level * (1 - self.breakout_buffer):
                    exit_conditions.append((True, "브레이크아웃실패"))
            
            # 6. 모멘텀 소진 (더 정밀한 감지)
            if indicators:
                rsi = indicators.get("rsi", 50)
                momentum_5 = indicators.get("momentum_5", 0)
                entry_momentum = self.entry_metrics.get("momentum_5", 0)
                
                # RSI 급락 또는 모멘텀 반전
                if (rsi < 45 and pnl_ratio < 0.01) or (momentum_5 < entry_momentum * 0.3):
                    exit_conditions.append((True, f"모멘텀소진 (RSI:{rsi:.0f}, 모멘텀:{momentum_5:.3%})"))
            
            # 7. 거래량 급감 (더 엄격한 기준)
            if indicators and "volume_ratio_short" in indicators:
                current_vol = indicators["volume_ratio_short"]
                entry_vol = self.entry_metrics.get("volume_ratio", 1.0)
                
                if current_vol < entry_vol * 0.4 and holding_bars > 2:  # 60% 감소 → 40% 감소
                    exit_conditions.append((True, f"거래량급감 (Vol:{current_vol:.1f}x)"))
            
            # 8. MACD 약세 전환 (손익 고려)
            if indicators and indicators.get("macd_bullish") == False:
                if pnl_ratio > 0.005:  # 수익 중일 때만
                    exit_conditions.append((True, "MACD약세전환"))
            
            # 조건 확인
            for condition, reason in exit_conditions:
                if condition:
                    logger.info(
                        f"[{self.name}] 향상된 청산 신호 | "
                        f"이유: {reason} | "
                        f"PnL: {pnl_ratio:.2%} | "
                        f"최고 PnL: {high_water_pnl:.2%} | "
                        f"보유: {holding_bars:.1f}봉"
                    )
                    
                    # 재진입 관리
                    if reason == "브레이크아웃실패" and self.enable_reentry:
                        self.last_exit_time = pd.Timestamp.now()
                        if self.reentry_count < self.max_reentries:
                            self.reentry_count += 1
                    else:
                        self.reentry_count = 0
                    
                    return reason
            
            return None
            
        except Exception as e:
            logger.error(f"청산 조건 확인 오류: {e}")
            return None

    def _calculate_dynamic_thresholds(self, indicators: Dict[str, float]) -> Tuple[float, float]:
        """변동성 기반 동적 임계값 계산"""
        # 기본값
        tp = self.tp_base
        sl = self.sl_base
        
        # 브레이크아웃 강도 기반 조정
        breakout_strength = indicators.get("breakout_strength", 0)
        if breakout_strength > 0.67:  # 강한 브레이크아웃
            tp *= (1 + breakout_strength * self.breakout_strength_multiplier)
            sl *= 0.85  # 타이트한 손절
        
        # 변동성 기반 조정
        vol_ratio = indicators.get("volatility_ratio", 1.0)
        if vol_ratio > 1.5:  # 변동성 확대
            vol_adjustment = min(vol_ratio - 1.0, 2.0)  # 최대 2배 제한
            tp *= (1 + vol_adjustment * self.vol_multiplier_tp * 0.1)
            sl *= (1 + vol_adjustment * self.vol_multiplier_sl * 0.1)
        
        # ATR 기반 조정
        atr_pct = indicators.get("atr_pct", 0)
        if atr_pct > 0:
            tp = max(tp, atr_pct * 2.5)  # ATR의 2.5배 이상
            sl = max(sl, atr_pct * 1.2)  # ATR의 1.2배 이상
        
        # 범위 제한 (더 보수적)
        tp = max(0.02, min(tp, 0.08))   # 2% ~ 8%
        sl = max(0.01, min(sl, 0.04))   # 1% ~ 4%
        
        return tp, sl

    def _calculate_enhanced_confidence(self, 
                                     indicators: Dict[str, float], 
                                     conditions: Dict[str, bool]) -> float:
        """향상된 신뢰도 계산"""
        base_confidence = 0.5
        
        # 브레이크아웃 강도 (핵심 요소)
        breakout_strength = indicators.get("breakout_strength", 0)
        base_confidence += breakout_strength * 0.2
        
        # 레인지/변동성 조건 (둘 다 만족 시 보너스)
        if conditions.get("range_condition") and conditions.get("range_expansion"):
            base_confidence += 0.15
        elif conditions.get("range_condition") or conditions.get("range_expansion"):
            base_confidence += 0.08
        
        # 거래량 확인 (더 엄격한 평가)
        if conditions.get("volume_spike") and conditions.get("volume_sufficient"):
            volume_ratio = indicators.get("volume_ratio_short", 1.0)
            volume_bonus = min(0.15, (volume_ratio - self.volume_spike_ratio) * 0.08)
            base_confidence += volume_bonus
        
        if conditions.get("volume_trend"):
            base_confidence += 0.08
        
        # 모멘텀 지표들 (개별 평가)
        momentum_score = 0
        if conditions.get("rsi_bullish"):
            rsi = indicators.get("rsi", 50)
            # RSI 55-70 범위에서 최고 점수
            if 55 <= rsi <= 70:
                momentum_score += 0.1
            elif rsi > 70:  # 과매수 패널티
                momentum_score += 0.05
        
        if conditions.get("macd_bullish"):
            momentum_score += 0.08
        
        if conditions.get("momentum_strong"):
            momentum_5 = indicators.get("momentum_5", 0)
            momentum_score += min(0.1, momentum_5 * 15)  # 모멘텀 크기에 비례
        
        if conditions.get("momentum_increasing"):
            momentum_score += 0.07
        
        base_confidence += momentum_score
        
        # 변동성 적정성 보너스
        vol_ratio = indicators.get("volatility_ratio", 1.0)
        if 1.5 <= vol_ratio <= 2.5:  # 이상적 범위
            base_confidence += 0.1
        elif vol_ratio > 3.0:  # 과도한 변동성 패널티
            base_confidence -= 0.05
        
        return max(0.0, min(1.0, base_confidence))

    def _generate_detailed_entry_reason(self, 
                                       conditions: Dict[str, bool], 
                                       indicators: Dict[str, float]) -> str:
        """상세한 진입 이유 생성"""
        reasons = []
        
        # 브레이크아웃 타입
        if indicators.get("breakout_long", False):
            reasons.append("장기고점돌파")
        elif indicators.get("breakout_medium", False):
            reasons.append("중기고점돌파")
        elif indicators.get("breakout_short", False):
            reasons.append("단기고점돌파")
        
        breakout_strength = indicators.get("breakout_strength", 0)
        reasons.append(f"강도({breakout_strength:.2f})")
        
        # 레인지/변동성
        if conditions.get("range_condition"):
            reasons.append("레인지수축")
        if conditions.get("range_expansion"):
            vol_ratio = indicators.get("volatility_ratio", 1.0)
            reasons.append(f"변동성확대({vol_ratio:.1f}x)")
        
        # 거래량
        if conditions.get("volume_spike"):
            volume_ratio = indicators.get("volume_ratio_short", 1.0)
            reasons.append(f"거래량급증({volume_ratio:.1f}x)")
        
        if conditions.get("volume_trend"):
            reasons.append("거래량추세")
        
        # 모멘텀
        momentum_reasons = []
        if conditions.get("rsi_bullish"):
            rsi = indicators.get("rsi", 50)
            momentum_reasons.append(f"RSI({rsi:.0f})")
        
        if conditions.get("macd_bullish"):
            momentum_reasons.append("MACD상승")
        
        if conditions.get("momentum_strong"):
            momentum_5 = indicators.get("momentum_5", 0)
            momentum_reasons.append(f"모멘텀({momentum_5:.2%})")
        
        if momentum_reasons:
            reasons.append("모멘텀:" + "+".join(momentum_reasons))
        
        return ", ".join(reasons) if reasons else "향상된브레이크아웃조건충족"

    def _check_reentry_cooldown(self, current_time) -> bool:
        """재진입 쿨다운 확인"""
        if not self.enable_reentry or self.last_exit_time is None:
            return False
        
        time_diff = (current_time - self.last_exit_time).total_seconds()
        return time_diff < self.reentry_cooldown

    def _identify_support_resistance(self, price_data: pd.DataFrame) -> List[float]:
        """지지/저항 레벨 식별 (간단화)"""
        if len(price_data) < 10:
            return []
        
        levels = []
        high_values = price_data["high"].values
        low_values = price_data["low"].values
        
        # 간단한 피벗 포인트 찾기
        for i in range(2, len(high_values) - 2):
            # 저항선 (고점)
            if (high_values[i] > high_values[i-1] and high_values[i] > high_values[i-2] and 
                high_values[i] > high_values[i+1] and high_values[i] > high_values[i+2]):
                levels.append(float(high_values[i]))
            
            # 지지선 (저점)
            if (low_values[i] < low_values[i-1] and low_values[i] < low_values[i-2] and 
                low_values[i] < low_values[i+1] and low_values[i] < low_values[i+2]):
                levels.append(float(low_values[i]))
        
        return sorted(list(set(levels)))[-5:] if levels else []  # 최근 5개만

    def _check_near_levels(self, current_price: float, levels: List[float]) -> bool:
        """현재가가 지지/저항선 근처인지 확인"""
        if not levels:
            return False
        
        for level in levels:
            if abs(current_price - level) / level < self.sr_touch_threshold:
                return True
        return False

    def get_regime_preferences(self) -> Dict[str, float]:
        """시장 체제별 선호도 (전략 E 맞춤 - 브레이크아웃)"""
        return {
            "bull": 0.9,         # 상승장 최고 선호 (브레이크아웃 기회)
            "bear": 0.3,         # 하락장 부분 활용 (반등 브레이크아웃)
            "sideways": 0.85,    # 횡보장 선호 (레인지 브레이크아웃)
            "high_volatility": 0.95,  # 고변동성 최고 선호 (브레이크아웃 최적)
            "crisis": 0.2        # 위기 상황 회피
        }