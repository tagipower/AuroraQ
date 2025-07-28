#!/usr/bin/env python3
"""
AuroraQ Production Rule Strategies
==================================

통합 룰 전략 모듈 - 모든 룰 기반 전략을 포함합니다.

Strategies included:
- RuleStrategyA: EMA 크로스오버 + ADX 트렌드 강도 전략
- RuleStrategyB: 거래량 스파이크 + RSI + 이동평균 전략  
- RuleStrategyC: 볼린저밴드 + 스토캐스틱 전략
- RuleStrategyD: MACD + 이동평균 수렴/발산 전략
- RuleStrategyE: 다중 지표 종합 분석 전략

독립적 동작:
- 외부 의존성 최소화
- 기본 파라미터 내장
- 표준화된 인터페이스 제공
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import logging
import sys
import os

# 통합 모듈 import (Fallback 포함)
try:
    from AuroraQ_Shared.utils.logger import get_logger
except ImportError:
    def get_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

try:
    from AuroraQ_Shared.utils.config_manager import load_config
except ImportError:
    def load_config(**kwargs):
        return type('Config', (), {
            'trading': type('Trading', (), {}),
            'risk': type('Risk', (), {}),
            'log_level': 'INFO'
        })()

logger = get_logger("RuleStrategies")

class BaseRuleStrategy:
    """룰 전략 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.last_signal_time = None
        self.position_count = 0
        self.success_count = 0
        self.total_pnl = 0.0
        
        # 기본 설정 로드
        self.config = load_config(component_type="production")
        
        # 지표 캐시
        self._indicator_cache = {}
        
        logger.info(f"{self.name} 전략 초기화 완료")
    
    def safe_last(self, data: pd.DataFrame, column: str, default=0.0):
        """안전한 마지막 값 추출"""
        try:
            if column in data.columns and len(data) > 0:
                value = data[column].iloc[-1]
                return float(value) if not pd.isna(value) else default
            return default
        except (IndexError, KeyError, TypeError):
            return default
    
    def get_cached_indicator(self, indicator_name: str, data: pd.DataFrame, **params) -> Optional[pd.Series]:
        """지표 캐시 시스템"""
        cache_key = f"{indicator_name}_{hash(str(params))}"
        
        if cache_key in self._indicator_cache:
            cached_data, cached_length = self._indicator_cache[cache_key]
            if len(data) == cached_length:
                return cached_data
        
        # 지표 계산
        try:
            if indicator_name == "ema":
                period = params.get("period", 20)
                result = data["close"].ewm(span=period).mean()
            elif indicator_name == "sma":
                period = params.get("period", 20)
                result = data["close"].rolling(period).mean()
            elif indicator_name == "rsi":
                period = params.get("period", 14)
                result = self._calculate_rsi(data["close"], period)
            elif indicator_name == "adx":
                period = params.get("period", 14)
                result = self._calculate_adx(data, period)
            elif indicator_name == "bollinger":
                period = params.get("period", 20)
                std_dev = params.get("std_dev", 2)
                result = self._calculate_bollinger(data["close"], period, std_dev)
            elif indicator_name == "macd":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                result = self._calculate_macd(data["close"], fast, slow, signal)
            elif indicator_name == "stochastic":
                k_period = params.get("k_period", 14)
                d_period = params.get("d_period", 3)
                result = self._calculate_stochastic(data, k_period, d_period)
            else:
                return None
            
            # 캐시 저장
            self._indicator_cache[cache_key] = (result, len(data))
            return result
            
        except Exception as e:
            logger.error(f"지표 계산 오류 {indicator_name}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """ADX 계산"""
        high, low, close = data["high"], data["low"], data["close"]
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr1 = pd.DataFrame(high - low).abs()
        tr2 = pd.DataFrame(high - close.shift()).abs()
        tr3 = pd.DataFrame(low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_bollinger(self, prices: pd.Series, period: int, std_dev: float) -> Dict[str, pd.Series]:
        """볼린저밴드 계산"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, pd.Series]:
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int, d_period: int) -> Dict[str, pd.Series]:
        """스토캐스틱 계산"""
        high, low, close = data["high"], data["low"], data["close"]
        
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def calculate_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """지표 계산 - 테스트 및 분석용 Public 메서드"""
        try:
            if len(price_data) < 50:  # 최소 데이터 요구사항
                return {}
            
            indicators = {}
            
            # 기본 이동평균
            indicators['sma_20'] = self.get_cached_indicator("sma", price_data, period=20)
            indicators['ema_12'] = self.get_cached_indicator("ema", price_data, period=12)
            indicators['ema_26'] = self.get_cached_indicator("ema", price_data, period=26)
            
            # 모멘텀 지표
            indicators['rsi_14'] = self.get_cached_indicator("rsi", price_data, period=14)
            indicators['adx_14'] = self.get_cached_indicator("adx", price_data, period=14)
            
            # 볼린저밴드
            bollinger = self._calculate_bollinger(price_data["close"], 20, 2.0)
            indicators['bb_upper'] = bollinger['upper']
            indicators['bb_middle'] = bollinger['middle']
            indicators['bb_lower'] = bollinger['lower']
            
            # MACD
            macd = self._calculate_macd(price_data["close"], 12, 26, 9)
            indicators['macd'] = macd['macd']
            indicators['macd_signal'] = macd['signal']
            indicators['macd_histogram'] = macd['histogram']
            
            # 스토캐스틱
            stochastic = self._calculate_stochastic(price_data, 14, 3)
            indicators['stoch_k'] = stochastic['k']
            indicators['stoch_d'] = stochastic['d']
            
            # 거래량 관련
            if "volume" in price_data.columns:
                indicators['volume_ma'] = price_data["volume"].rolling(20).mean()
                indicators['volume_ratio'] = price_data["volume"] / indicators['volume_ma']
            
            # 변동성 지표
            indicators['atr_14'] = price_data["close"].rolling(14).apply(
                lambda x: (x.max() - x.min()) / x.mean(), raw=False
            )
            
            return indicators
            
        except Exception as e:
            logger.error(f"지표 계산 오류 ({self.name}): {e}")
            return {}
    
    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """진입 조건 확인 - 하위 클래스에서 구현"""
        raise NotImplementedError
    
    def should_exit(self, position, price_data: pd.DataFrame) -> Optional[str]:
        """청산 조건 확인 - 하위 클래스에서 구현"""
        raise NotImplementedError
    
    def reset(self):
        """전략 상태 리셋"""
        self._indicator_cache.clear()
        self.last_signal_time = None

class RuleStrategyA(BaseRuleStrategy):
    """EMA 크로스오버 + ADX 트렌드 강도 전략"""
    
    def __init__(self):
        super().__init__(name="RuleStrategyA")
        
        # 지표 설정
        self.ema_short_len = 8
        self.ema_long_len = 21
        self.adx_window = 14
        self.adx_threshold = 25
        
        # 리스크 관리
        self.take_profit_pct = 0.025
        self.stop_loss_pct = 0.015
        self.max_hold_bars = 10
        
        # 필터
        self.min_volume_ratio = 1.0
        self.min_ema_diff_pct = 0.003
        
        logger.info(f"RuleStrategyA 초기화: EMA({self.ema_short_len}/{self.ema_long_len}), ADX≥{self.adx_threshold}")
    
    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """진입 조건 확인"""
        try:
            if len(price_data) < max(self.ema_long_len, self.adx_window + 1):
                return None
            
            # 지표 계산
            ema_short = self.get_cached_indicator("ema", price_data, period=self.ema_short_len)
            ema_long = self.get_cached_indicator("ema", price_data, period=self.ema_long_len)
            adx = self.get_cached_indicator("adx", price_data, period=self.adx_window)
            
            if any(x is None for x in [ema_short, ema_long, adx]):
                return None
            
            # 현재 값들
            current_price = self.safe_last(price_data, "close")
            ema_short_val = float(ema_short.iloc[-1])
            ema_long_val = float(ema_long.iloc[-1])
            adx_val = float(adx.iloc[-1])
            
            # 진입 조건
            conditions = {
                "ema_bullish": ema_short_val > ema_long_val,
                "ema_diff_sufficient": abs(ema_short_val - ema_long_val) / ema_long_val >= self.min_ema_diff_pct,
                "adx_strong": adx_val >= self.adx_threshold,
            }
            
            # 볼륨 확인
            if "volume" in price_data.columns and len(price_data) >= 20:
                avg_volume = price_data["volume"].tail(20).mean()
                current_volume = price_data["volume"].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                conditions["volume_sufficient"] = volume_ratio >= self.min_volume_ratio
            
            if not all(conditions.values()):
                return None
            
            # 신뢰도 계산
            confidence = 0.5
            confidence += min(0.3, abs(ema_short_val - ema_long_val) / ema_long_val * 100)
            confidence += min(0.2, (adx_val - self.adx_threshold) / 50)
            
            return {
                "side": "LONG",
                "confidence": min(confidence, 1.0),
                "reason": f"EMA크로스({ema_short_val:.2f}>{ema_long_val:.2f}), ADX={adx_val:.1f}",
                "stop_loss": current_price * (1 - self.stop_loss_pct),
                "take_profit": current_price * (1 + self.take_profit_pct)
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyA 진입 조건 확인 오류: {e}")
            return None
    
    def should_exit(self, position, price_data: pd.DataFrame) -> Optional[str]:
        """청산 조건 확인"""
        try:
            if not position or not hasattr(position, 'entry_price'):
                return None
            
            current_price = self.safe_last(price_data, "close")
            pnl_ratio = (current_price - position.entry_price) / position.entry_price
            
            # 손절/익절
            if pnl_ratio <= -self.stop_loss_pct:
                return f"손절 ({pnl_ratio:.1%})"
            if pnl_ratio >= self.take_profit_pct:
                return f"익절 ({pnl_ratio:.1%})"
            
            # 시간 초과
            if hasattr(position, 'holding_time'):
                holding_bars = position.holding_time.total_seconds() / 300
                if holding_bars >= self.max_hold_bars:
                    return f"시간초과 ({holding_bars:.0f}봉)"
            
            return None
            
        except Exception as e:
            logger.error(f"RuleStrategyA 청산 조건 확인 오류: {e}")
            return None

class RuleStrategyB(BaseRuleStrategy):
    """거래량 스파이크 + RSI + 이동평균 전략"""
    
    def __init__(self):
        super().__init__(name="RuleStrategyB")
        
        # 지표 설정
        self.ma_short_len = 8
        self.ma_long_len = 35
        self.rsi_period = 14
        self.volume_spike_ratio = 1.5
        
        # RSI 임계값
        self.rsi_threshold_low = 30
        self.rsi_threshold_high = 70
        
        # 리스크 관리
        self.take_profit_pct = 0.028
        self.stop_loss_pct = 0.015
        self.max_hold_bars = 8
        
        logger.info(f"RuleStrategyB 초기화: MA({self.ma_short_len}/{self.ma_long_len}), RSI({self.rsi_period})")
    
    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """진입 조건 확인"""
        try:
            if len(price_data) < max(self.ma_long_len, self.rsi_period + 1, 20):
                return None
            
            # 지표 계산
            ma_short = self.get_cached_indicator("sma", price_data, period=self.ma_short_len)
            ma_long = self.get_cached_indicator("sma", price_data, period=self.ma_long_len)
            rsi = self.get_cached_indicator("rsi", price_data, period=self.rsi_period)
            
            if any(x is None for x in [ma_short, ma_long, rsi]):
                return None
            
            # 현재 값들
            current_price = self.safe_last(price_data, "close")
            ma_short_val = float(ma_short.iloc[-1])
            ma_long_val = float(ma_long.iloc[-1])
            rsi_val = float(rsi.iloc[-1])
            
            # 거래량 스파이크 확인
            if "volume" in price_data.columns:
                avg_volume = price_data["volume"].tail(20).mean()
                current_volume = price_data["volume"].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # 진입 조건
            conditions = {
                "ma_bullish": ma_short_val > ma_long_val,
                "rsi_range": self.rsi_threshold_low <= rsi_val <= self.rsi_threshold_high,
                "volume_spike": volume_ratio >= self.volume_spike_ratio,
                "price_above_ma": current_price > ma_short_val
            }
            
            if not all(conditions.values()):
                return None
            
            # 신뢰도 계산
            confidence = 0.5
            confidence += min(0.25, (ma_short_val - ma_long_val) / ma_long_val * 50)
            confidence += min(0.15, max(0, volume_ratio - 1.0) * 0.1)
            if 40 <= rsi_val <= 60:  # 중성 RSI
                confidence += 0.1
            
            return {
                "side": "LONG",
                "confidence": min(confidence, 1.0),
                "reason": f"MA크로스+볼륨스파이크({volume_ratio:.1f}x), RSI={rsi_val:.1f}",
                "stop_loss": current_price * (1 - self.stop_loss_pct),
                "take_profit": current_price * (1 + self.take_profit_pct)
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyB 진입 조건 확인 오류: {e}")
            return None
    
    def should_exit(self, position, price_data: pd.DataFrame) -> Optional[str]:
        """청산 조건 확인"""
        try:
            if not position or not hasattr(position, 'entry_price'):
                return None
            
            current_price = self.safe_last(price_data, "close")
            pnl_ratio = (current_price - position.entry_price) / position.entry_price
            
            # 손절/익절
            if pnl_ratio <= -self.stop_loss_pct:
                return f"손절 ({pnl_ratio:.1%})"
            if pnl_ratio >= self.take_profit_pct:
                return f"익절 ({pnl_ratio:.1%})"
            
            # RSI 극값 확인
            rsi = self.get_cached_indicator("rsi", price_data, period=self.rsi_period)
            if rsi is not None:
                rsi_val = float(rsi.iloc[-1])
                if rsi_val >= 75:  # 과매수
                    return f"RSI과매수 ({rsi_val:.1f})"
            
            # 시간 초과
            if hasattr(position, 'holding_time'):
                holding_bars = position.holding_time.total_seconds() / 300
                if holding_bars >= self.max_hold_bars:
                    return f"시간초과 ({holding_bars:.0f}봉)"
            
            return None
            
        except Exception as e:
            logger.error(f"RuleStrategyB 청산 조건 확인 오류: {e}")
            return None

class RuleStrategyC(BaseRuleStrategy):
    """볼린저밴드 + 스토캐스틱 전략"""
    
    def __init__(self):
        super().__init__(name="RuleStrategyC")
        
        # 지표 설정
        self.bb_period = 20
        self.bb_std_dev = 2
        self.stoch_k_period = 14
        self.stoch_d_period = 3
        
        # 임계값
        self.stoch_oversold = 20
        self.stoch_overbought = 80
        
        # 리스크 관리
        self.take_profit_pct = 0.025
        self.stop_loss_pct = 0.018
        self.max_hold_bars = 12
        
        logger.info(f"RuleStrategyC 초기화: BB({self.bb_period},{self.bb_std_dev}), Stoch({self.stoch_k_period},{self.stoch_d_period})")
    
    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """진입 조건 확인"""
        try:
            if len(price_data) < max(self.bb_period, self.stoch_k_period + 1):
                return None
            
            # 지표 계산
            bb = self.get_cached_indicator("bollinger", price_data, period=self.bb_period, std_dev=self.bb_std_dev)
            stoch = self.get_cached_indicator("stochastic", price_data, k_period=self.stoch_k_period, d_period=self.stoch_d_period)
            
            if bb is None or stoch is None:
                return None
            
            # 현재 값들
            current_price = self.safe_last(price_data, "close")
            bb_lower = float(bb['lower'].iloc[-1])
            bb_middle = float(bb['middle'].iloc[-1])
            bb_upper = float(bb['upper'].iloc[-1])
            stoch_k = float(stoch['k'].iloc[-1])
            stoch_d = float(stoch['d'].iloc[-1])
            
            # 진입 조건 (볼린저밴드 하단 근처에서 스토캐스틱 과매도 반등)
            conditions = {
                "price_near_bb_lower": current_price <= bb_lower * 1.02,  # 하단 2% 내
                "stoch_oversold": stoch_k <= self.stoch_oversold,
                "stoch_turning_up": stoch_k > stoch_d,  # K선이 D선 위로
                "not_falling_knife": current_price > bb_lower * 0.98  # 급락 중이 아님
            }
            
            if not all(conditions.values()):
                return None
            
            # 신뢰도 계산
            confidence = 0.5
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            confidence += min(0.3, (1 - bb_position) * 0.6)  # 하단에 가까울수록 높은 점수
            confidence += min(0.2, max(0, self.stoch_oversold - stoch_k) / self.stoch_oversold)
            
            return {
                "side": "LONG",
                "confidence": min(confidence, 1.0),
                "reason": f"BB반등({bb_position:.1%}), Stoch과매도반등({stoch_k:.1f}>{stoch_d:.1f})",
                "stop_loss": current_price * (1 - self.stop_loss_pct),
                "take_profit": min(current_price * (1 + self.take_profit_pct), bb_middle)  # BB 중간선을 목표로
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyC 진입 조건 확인 오류: {e}")
            return None
    
    def should_exit(self, position, price_data: pd.DataFrame) -> Optional[str]:
        """청산 조건 확인"""
        try:
            if not position or not hasattr(position, 'entry_price'):
                return None
            
            current_price = self.safe_last(price_data, "close")
            pnl_ratio = (current_price - position.entry_price) / position.entry_price
            
            # 손절/익절
            if pnl_ratio <= -self.stop_loss_pct:
                return f"손절 ({pnl_ratio:.1%})"
            if pnl_ratio >= self.take_profit_pct:
                return f"익절 ({pnl_ratio:.1%})"
            
            # 볼린저밴드 상단 도달
            bb = self.get_cached_indicator("bollinger", price_data, period=self.bb_period, std_dev=self.bb_std_dev)
            if bb is not None:
                bb_upper = float(bb['upper'].iloc[-1])
                if current_price >= bb_upper * 0.98:  # 상단 2% 내
                    return f"BB상단도달 ({current_price:.2f}>={bb_upper:.2f})"
            
            # 스토캐스틱 과매수
            stoch = self.get_cached_indicator("stochastic", price_data, k_period=self.stoch_k_period, d_period=self.stoch_d_period)
            if stoch is not None:
                stoch_k = float(stoch['k'].iloc[-1])
                if stoch_k >= self.stoch_overbought:
                    return f"Stoch과매수 ({stoch_k:.1f})"
            
            # 시간 초과
            if hasattr(position, 'holding_time'):
                holding_bars = position.holding_time.total_seconds() / 300
                if holding_bars >= self.max_hold_bars:
                    return f"시간초과 ({holding_bars:.0f}봉)"
            
            return None
            
        except Exception as e:
            logger.error(f"RuleStrategyC 청산 조건 확인 오류: {e}")
            return None

class RuleStrategyD(BaseRuleStrategy):
    """MACD + 이동평균 수렴/발산 전략"""
    
    def __init__(self):
        super().__init__(name="RuleStrategyD")
        
        # MACD 설정
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 이동평균 설정
        self.ma_period = 50
        
        # 리스크 관리
        self.take_profit_pct = 0.03
        self.stop_loss_pct = 0.016
        self.max_hold_bars = 15
        
        logger.info(f"RuleStrategyD 초기화: MACD({self.macd_fast},{self.macd_slow},{self.macd_signal}), MA({self.ma_period})")
    
    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """진입 조건 확인"""
        try:
            if len(price_data) < max(self.macd_slow + self.macd_signal, self.ma_period):
                return None
            
            # 지표 계산
            macd_data = self.get_cached_indicator("macd", price_data, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            ma = self.get_cached_indicator("sma", price_data, period=self.ma_period)
            
            if macd_data is None or ma is None:
                return None
            
            # 현재 값들
            current_price = self.safe_last(price_data, "close")
            macd_val = float(macd_data['macd'].iloc[-1])
            signal_val = float(macd_data['signal'].iloc[-1])
            histogram = float(macd_data['histogram'].iloc[-1])
            ma_val = float(ma.iloc[-1])
            
            # 이전 값들 (크로스오버 확인용)
            if len(macd_data['macd']) >= 2:
                prev_macd = float(macd_data['macd'].iloc[-2])
                prev_signal = float(macd_data['signal'].iloc[-2])
            else:
                prev_macd = prev_signal = 0
            
            # 진입 조건
            conditions = {
                "macd_bullish_cross": prev_macd <= prev_signal and macd_val > signal_val,  # MACD 상향 크로스
                "macd_above_zero": macd_val > 0,  # MACD 0선 위
                "price_above_ma": current_price > ma_val,  # 가격이 이평선 위
                "histogram_positive": histogram > 0  # 히스토그램 양수
            }
            
            if not all(conditions.values()):
                return None
            
            # 신뢰도 계산
            confidence = 0.6  # MACD 크로스는 강한 신호
            confidence += min(0.2, macd_val / 100)  # MACD 크기
            confidence += min(0.2, (current_price - ma_val) / ma_val * 10)  # MA 대비 위치
            
            return {
                "side": "LONG",
                "confidence": min(confidence, 1.0),
                "reason": f"MACD골든크로스({macd_val:.3f}>{signal_val:.3f}), MA상향돌파",
                "stop_loss": current_price * (1 - self.stop_loss_pct),
                "take_profit": current_price * (1 + self.take_profit_pct)
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyD 진입 조건 확인 오류: {e}")
            return None
    
    def should_exit(self, position, price_data: pd.DataFrame) -> Optional[str]:
        """청산 조건 확인"""
        try:
            if not position or not hasattr(position, 'entry_price'):
                return None
            
            current_price = self.safe_last(price_data, "close")
            pnl_ratio = (current_price - position.entry_price) / position.entry_price
            
            # 손절/익절
            if pnl_ratio <= -self.stop_loss_pct:
                return f"손절 ({pnl_ratio:.1%})"
            if pnl_ratio >= self.take_profit_pct:
                return f"익절 ({pnl_ratio:.1%})"
            
            # MACD 데드 크로스
            macd_data = self.get_cached_indicator("macd", price_data, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd_data is not None and len(macd_data['macd']) >= 2:
                macd_val = float(macd_data['macd'].iloc[-1])
                signal_val = float(macd_data['signal'].iloc[-1])
                prev_macd = float(macd_data['macd'].iloc[-2])
                prev_signal = float(macd_data['signal'].iloc[-2])
                
                if prev_macd >= prev_signal and macd_val < signal_val:  # MACD 하향 크로스
                    return f"MACD데드크로스 ({macd_val:.3f}<{signal_val:.3f})"
            
            # 시간 초과
            if hasattr(position, 'holding_time'):
                holding_bars = position.holding_time.total_seconds() / 300
                if holding_bars >= self.max_hold_bars:
                    return f"시간초과 ({holding_bars:.0f}봉)"
            
            return None
            
        except Exception as e:
            logger.error(f"RuleStrategyD 청산 조건 확인 오류: {e}")
            return None

class RuleStrategyE(BaseRuleStrategy):
    """다중 지표 종합 분석 전략"""
    
    def __init__(self):
        super().__init__(name="RuleStrategyE")
        
        # 다중 지표 설정
        self.ema_short = 8
        self.ema_long = 21
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std_dev = 2
        self.adx_period = 14
        
        # 종합 점수 임계값
        self.min_score = 0.6
        
        # 리스크 관리
        self.take_profit_pct = 0.035
        self.stop_loss_pct = 0.02
        self.max_hold_bars = 20
        
        logger.info(f"RuleStrategyE 초기화: 다중지표 종합분석 전략")
    
    def _calculate_composite_score(self, price_data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """종합 점수 계산"""
        try:
            scores = {}
            current_price = self.safe_last(price_data, "close")
            
            # 1. EMA 점수
            ema_short = self.get_cached_indicator("ema", price_data, period=self.ema_short)
            ema_long = self.get_cached_indicator("ema", price_data, period=self.ema_long)
            if ema_short is not None and ema_long is not None:
                ema_short_val = float(ema_short.iloc[-1])
                ema_long_val = float(ema_long.iloc[-1])
                if ema_short_val > ema_long_val:
                    scores['ema'] = min(1.0, (ema_short_val - ema_long_val) / ema_long_val * 100)
                else:
                    scores['ema'] = 0.0
            
            # 2. RSI 점수
            rsi = self.get_cached_indicator("rsi", price_data, period=self.rsi_period)
            if rsi is not None:
                rsi_val = float(rsi.iloc[-1])
                if 30 <= rsi_val <= 70:  # 적정 범위
                    scores['rsi'] = 0.5 + (50 - abs(rsi_val - 50)) / 50 * 0.5
                else:
                    scores['rsi'] = 0.0
            
            # 3. 볼린저밴드 점수
            bb = self.get_cached_indicator("bollinger", price_data, period=self.bb_period, std_dev=self.bb_std_dev)
            if bb is not None:
                bb_lower = float(bb['lower'].iloc[-1])
                bb_middle = float(bb['middle'].iloc[-1])
                bb_upper = float(bb['upper'].iloc[-1])
                
                if current_price < bb_middle:
                    # 하단에서 중간으로 향하는 경우
                    bb_position = (current_price - bb_lower) / (bb_middle - bb_lower)
                    scores['bb'] = max(0, min(1.0, bb_position))
                else:
                    scores['bb'] = 0.5  # 중성
            
            # 4. ADX 점수
            adx = self.get_cached_indicator("adx", price_data, period=self.adx_period)
            if adx is not None:
                adx_val = float(adx.iloc[-1])
                scores['adx'] = min(1.0, max(0, (adx_val - 20) / 30))  # 20~50 범위를 0~1로 정규화
            
            # 5. 거래량 점수
            if "volume" in price_data.columns and len(price_data) >= 20:
                avg_volume = price_data["volume"].tail(20).mean()
                current_volume = price_data["volume"].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                scores['volume'] = min(1.0, max(0, (volume_ratio - 0.8) / 1.2))  # 0.8~2.0을 0~1로
            
            # 가중 평균 계산
            weights = {
                'ema': 0.25,
                'rsi': 0.2,
                'bb': 0.2,
                'adx': 0.2,
                'volume': 0.15
            }
            
            total_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())
            return total_score, scores
            
        except Exception as e:
            logger.error(f"종합 점수 계산 오류: {e}")
            return 0.0, {}
    
    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """진입 조건 확인"""
        try:
            if len(price_data) < max(self.ema_long, self.bb_period, self.adx_period + 1):
                return None
            
            # 종합 점수 계산
            total_score, individual_scores = self._calculate_composite_score(price_data)
            
            if total_score < self.min_score:
                return None
            
            current_price = self.safe_last(price_data, "close")
            
            # 추가 필터 (최소 조건)
            ema_short = self.get_cached_indicator("ema", price_data, period=self.ema_short)
            if ema_short is not None and current_price < float(ema_short.iloc[-1]):
                return None  # 가격이 단기 EMA 아래
            
            return {
                "side": "LONG",
                "confidence": total_score,
                "reason": f"종합점수({total_score:.2f}) = EMA:{individual_scores.get('ema', 0):.2f}, RSI:{individual_scores.get('rsi', 0):.2f}, BB:{individual_scores.get('bb', 0):.2f}",
                "stop_loss": current_price * (1 - self.stop_loss_pct),
                "take_profit": current_price * (1 + self.take_profit_pct),
                "scores": individual_scores
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyE 진입 조건 확인 오류: {e}")
            return None
    
    def should_exit(self, position, price_data: pd.DataFrame) -> Optional[str]:
        """청산 조건 확인"""
        try:
            if not position or not hasattr(position, 'entry_price'):
                return None
            
            current_price = self.safe_last(price_data, "close")
            pnl_ratio = (current_price - position.entry_price) / position.entry_price
            
            # 손절/익절
            if pnl_ratio <= -self.stop_loss_pct:
                return f"손절 ({pnl_ratio:.1%})"
            if pnl_ratio >= self.take_profit_pct:
                return f"익절 ({pnl_ratio:.1%})"
            
            # 종합 점수 악화
            total_score, _ = self._calculate_composite_score(price_data)
            if total_score < 0.3:  # 점수가 너무 낮아짐
                return f"종합점수악화 ({total_score:.2f})"
            
            # 시간 초과
            if hasattr(position, 'holding_time'):
                holding_bars = position.holding_time.total_seconds() / 300
                if holding_bars >= self.max_hold_bars:
                    return f"시간초과 ({holding_bars:.0f}봉)"
            
            return None
            
        except Exception as e:
            logger.error(f"RuleStrategyE 청산 조건 확인 오류: {e}")
            return None

# 전략 레지스트리
RULE_STRATEGIES = {
    "RuleStrategyA": RuleStrategyA,
    "RuleStrategyB": RuleStrategyB,
    "RuleStrategyC": RuleStrategyC,
    "RuleStrategyD": RuleStrategyD,
    "RuleStrategyE": RuleStrategyE
}

def get_rule_strategy(strategy_name: str) -> Optional[BaseRuleStrategy]:
    """룰 전략 인스턴스 생성"""
    if strategy_name in RULE_STRATEGIES:
        try:
            return RULE_STRATEGIES[strategy_name]()
        except Exception as e:
            logger.error(f"전략 {strategy_name} 생성 실패: {e}")
            return None
    else:
        logger.warning(f"알 수 없는 전략: {strategy_name}")
        return None

def get_available_strategies() -> List[str]:
    """사용 가능한 전략 목록 반환"""
    return list(RULE_STRATEGIES.keys())

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 AuroraQ Rule Strategies 테스트")
    
    # 모든 전략 테스트
    for strategy_name in get_available_strategies():
        print(f"\n📋 {strategy_name} 테스트")
        try:
            strategy = get_rule_strategy(strategy_name)
            if strategy:
                print(f"✅ {strategy_name} 생성 성공")
            else:
                print(f"❌ {strategy_name} 생성 실패")
        except Exception as e:
            print(f"❌ {strategy_name} 오류: {e}")
    
    print(f"\n📊 총 {len(get_available_strategies())}개 전략 등록 완료")
    print(f"등록된 전략: {', '.join(get_available_strategies())}")