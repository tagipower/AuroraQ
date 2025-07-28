"""
Optimized RuleStrategyE - 백테스트 v2용 최적화된 전략
최적화된 파라미터를 사용하는 RuleStrategyE 래퍼
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import os
import sys

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strategy.base_rule_strategy import BaseRuleStrategy
from utils.logger import get_logger
from backtest.v2.config.optimized_param_loader import get_optimized_rule_params

logger = get_logger("OptimizedRuleStrategyE")


class OptimizedRuleStrategyE(BaseRuleStrategy):
    """
    최적화된 RuleStrategyE
    
    주요 개선사항:
    - 최적화된 파라미터 사용
    - 더 엄격한 진입 조건
    - 개선된 리스크/보상 비율
    - 강화된 필터링 시스템
    
    성능 목표:
    - Win Rate: 35%+ (기존 13.51%에서 개선)
    - Profit Factor: 0.6+ (기존 0.15에서 개선)
    - Sharpe Ratio: 0.5+
    """

    def __init__(self):
        super().__init__(name="OptimizedRuleStrategyE")
        
        # 최적화된 파라미터 로드
        params = get_optimized_rule_params("RuleE")
        
        if not params:
            logger.warning("최적화된 파라미터 로드 실패 - 기본값 사용")
            params = self._get_fallback_params()
        
        logger.info(f"최적화된 파라미터 로드 완료: {len(params)}개 설정")
        
        # === 브레이크아웃 설정 (더 보수적) ===
        self.breakout_window_short = params.get("breakout_window_short", 12)      # 8→12 (더 긴 확인)
        self.breakout_window_medium = params.get("breakout_window_medium", 20)    # 15→20 (강한 트렌드)
        self.breakout_window_long = params.get("breakout_window_long", 35)        # 30→35 (장기 필터)
        self.breakout_buffer = params.get("breakout_buffer", 0.002)               # 0.0015→0.002 (강한 돌파)
        
        # === 레인지/변동성 (더 엄격한 기준) ===
        self.range_window = params.get("range_window", 10)                       # 15→10 (민감도 향상)
        self.range_std_threshold = params.get("range_std_threshold", 0.0015)     # 0.002→0.0015 (엄격)
        self.range_atr_threshold = params.get("range_atr_threshold", 0.003)      # 0.004→0.003 (타이트)
        self.squeeze_window = params.get("squeeze_window", 30)                   # 25→30 (긴 베이스)
        self.expansion_ratio = params.get("expansion_ratio", 2.2)                # 1.8→2.2 (강한 확장)
        
        # === 거래량 확인 (더 강한 요구사항) ===
        self.volume_window = params.get("volume_window", 10)                     # 8→10 (더 나은 트렌드)
        self.volume_spike_ratio = params.get("volume_spike_ratio", 2.0)          # 1.5→2.0 (큰 스파이크)
        self.volume_ma_window = params.get("volume_ma_window", 20)               # 15→20 (부드러운 베이스)
        self.obv_window = params.get("obv_window", 15)                           # 12→15 (더 나은 OBV)
        self.min_volume_ratio = params.get("min_volume_ratio", 1.5)              # 1.2→1.5 (높은 최소값)
        
        # === 모멘텀 지표 최적화 ===
        # RSI 설정 (더 보수적)
        self.rsi_window = params.get("rsi_window", 14)                           # 표준 유지
        self.rsi_breakout_threshold = params.get("rsi_breakout_threshold", 60)   # 55→60 (강한 모멘텀)
        self.rsi_overbought = params.get("rsi_overbought", 75)                   # 새로운 필터
        self.rsi_oversold = params.get("rsi_oversold", 25)                       # 새로운 필터
        
        # MACD 설정 (더 나은 시그널 필터링)
        self.macd_fast = params.get("macd_fast", 12)                             # 표준 유지
        self.macd_slow = params.get("macd_slow", 26)                             # 표준 유지
        self.macd_signal = params.get("macd_signal", 9)                          # 표준 유지
        self.macd_hist_threshold = params.get("macd_hist_threshold", 0.0001)     # 새로운 최소값
        
        # 모멘텀 강도
        self.momentum_window = params.get("momentum_window", 8)                  # 5→8 (긴 확인)
        self.momentum_threshold = params.get("momentum_threshold", 0.008)        # 0.005→0.008 (강한 모멘텀)
        self.min_confidence = params.get("min_confidence", 0.8)                  # 0.75→0.8 (높은 임계값)
        
        # === 리스크 관리 최적화 ===
        # Take Profit (개선된 R:R)
        self.take_profit_pct = params.get("take_profit_pct", 0.025)              # 0.03→0.025 (달성 가능)
        self.vol_multiplier_tp = params.get("vol_multiplier_tp", 2.0)            # 2.5→2.0 (덜 조정)
        
        # Stop Loss (더 타이트하게)
        self.stop_loss_pct = params.get("stop_loss_pct", 0.012)                  # 0.018→0.012 (더 나은 R:R)
        self.vol_multiplier_sl = params.get("vol_multiplier_sl", 1.5)            # 1.8→1.5 (덜 조정)
        
        # 포지션 관리
        self.max_hold_bars = params.get("max_hold_bars", 8)                      # 6→8 (더 많은 시간)
        self.trailing_stop_activation = params.get("trailing_stop_activation", 0.012)  # 0.015→0.012 (빠른 시작)
        self.trailing_stop_distance = params.get("trailing_stop_distance", 0.006)      # 0.008→0.006 (타이트)
        
        # === 마켓 구조 필터 ===
        self.support_resistance_window = params.get("support_resistance_window", 120)   # 100→120 (긴 히스토리)
        self.sr_touch_threshold = params.get("sr_touch_threshold", 0.0005)       # 0.0008→0.0005 (정확한 S/R)
        self.trend_filter_window = params.get("trend_filter_window", 60)         # 50→60 (긴 트렌드)
        
        # === 재진입 제어 ===
        self.enable_reentry = params.get("enable_reentry", False)                # 비활성화 유지
        self.reentry_cooldown = params.get("reentry_cooldown", 900)              # 600→900 (긴 쿨다운)
        self.max_reentries = params.get("max_reentries", 1)                      # 보수적 유지
        
        # === 추가 필터 (새로운 기능) ===
        # 멀티 타임프레임 확인
        self.require_higher_tf_trend = params.get("require_higher_tf_trend", True)       # 상위 TF 정렬
        self.higher_tf_period = params.get("higher_tf_period", 100)              # 상위 TF 기간
        
        # 마켓 레짐 필터
        self.enable_regime_filter = params.get("enable_regime_filter", True)     # 변동성 마켓 회피
        self.regime_window = params.get("regime_window", 50)                     # 레짐 감지 기간
        self.min_trend_strength = params.get("min_trend_strength", 0.3)          # 최소 트렌드 강도
        
        # 변동성 필터
        self.max_volatility_threshold = params.get("max_volatility_threshold", 0.08)     # 극한 변동성 회피
        self.min_volatility_threshold = params.get("min_volatility_threshold", 0.005)   # 데드 마켓 회피
        
        # 시간 기반 필터
        self.avoid_market_open_minutes = params.get("avoid_market_open_minutes", 30)     # 오픈 30분 회피
        self.avoid_market_close_minutes = params.get("avoid_market_close_minutes", 30)   # 클로즈 30분 회피
        
        # 성능 목표
        self.performance_targets = params.get("performance_targets", {})
        
        # 지표 캐시
        self.indicator_cache = {}
        self.last_cache_update = None
        
        logger.info(f"최적화된 전략 초기화 완료:")
        logger.info(f"  - Win Rate 목표: {self.performance_targets.get('target_win_rate', 0.35)*100:.1f}%")
        logger.info(f"  - Profit Factor 목표: {self.performance_targets.get('target_profit_factor', 0.6)}")
        logger.info(f"  - Risk/Reward 비율: {self.take_profit_pct/self.stop_loss_pct:.2f}:1")
    
    def _get_fallback_params(self) -> Dict[str, Any]:
        """대체 파라미터 (파일 로드 실패 시)"""
        return {
            "breakout_window_short": 12,
            "breakout_window_medium": 20,
            "breakout_window_long": 35,
            "breakout_buffer": 0.002,
            "range_window": 10,
            "range_std_threshold": 0.0015,
            "range_atr_threshold": 0.003,
            "squeeze_window": 30,
            "expansion_ratio": 2.2,
            "volume_window": 10,
            "volume_spike_ratio": 2.0,
            "volume_ma_window": 20,
            "obv_window": 15,
            "min_volume_ratio": 1.5,
            "rsi_window": 14,
            "rsi_breakout_threshold": 60,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "macd_hist_threshold": 0.0001,
            "momentum_window": 8,
            "momentum_threshold": 0.008,
            "min_confidence": 0.8,
            "take_profit_pct": 0.025,
            "vol_multiplier_tp": 2.0,
            "stop_loss_pct": 0.012,
            "vol_multiplier_sl": 1.5,
            "max_hold_bars": 8,
            "trailing_stop_activation": 0.012,
            "trailing_stop_distance": 0.006,
            "performance_targets": {
                "target_win_rate": 0.35,
                "target_profit_factor": 0.6,
                "target_sharpe": 0.5,
                "max_drawdown_limit": 0.15
            }
        }
    
    def evaluate_entry_signal(self, price_data: pd.DataFrame, current_position: str, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        최적화된 진입 신호 평가
        더 엄격한 조건으로 높은 품질의 신호만 선택
        """
        try:
            if len(price_data) < max(self.breakout_window_long, self.trend_filter_window):
                return {"action": "HOLD", "strength": 0.0, "reason": "데이터 부족"}
            
            current_price = price_data['close'].iloc[-1]
            
            # === 1. 변동성 필터 (새로운 기능) ===
            volatility = self._calculate_volatility(price_data)
            if volatility > self.max_volatility_threshold or volatility < self.min_volatility_threshold:
                return {"action": "HOLD", "strength": 0.0, "reason": f"변동성 필터: {volatility:.4f}"}
            
            # === 2. 마켓 레짐 필터 (새로운 기능) ===
            if self.enable_regime_filter:
                trend_strength = self._calculate_trend_strength(price_data)
                if trend_strength < self.min_trend_strength:
                    return {"action": "HOLD", "strength": 0.0, "reason": f"약한 트렌드: {trend_strength:.3f}"}
            
            # === 3. 멀티 타임프레임 필터 (새로운 기능) ===
            if self.require_higher_tf_trend:
                htf_trend = self._check_higher_timeframe_trend(price_data)
                if not htf_trend:
                    return {"action": "HOLD", "strength": 0.0, "reason": "상위 TF 트렌드 불일치"}
            
            # === 4. 브레이크아웃 조건 (강화됨) ===
            breakout_signals = self._check_breakout_conditions(price_data, indicators)
            if not breakout_signals["valid"]:
                return {"action": "HOLD", "strength": 0.0, "reason": "브레이크아웃 조건 미충족"}
            
            # === 5. 레인지 확장 조건 (더 엄격) ===
            range_expansion = self._check_range_expansion(price_data, indicators)
            if not range_expansion["valid"]:
                return {"action": "HOLD", "strength": 0.0, "reason": "레인지 확장 미감지"}
            
            # === 6. 거래량 조건 (더 강한 요구사항) ===
            volume_confirmation = self._check_volume_conditions(price_data, indicators)
            if not volume_confirmation["valid"]:
                return {"action": "HOLD", "strength": 0.0, "reason": "거래량 조건 미충족"}
            
            # === 7. 모멘텀 조건 (개선된 필터) ===
            momentum_signals = self._check_momentum_conditions(price_data, indicators)
            if not momentum_signals["valid"]:
                return {"action": "HOLD", "strength": 0.0, "reason": "모멘텀 조건 미충족"}
            
            # === 8. RSI 필터 (새로운 오버바잇/오버솔드 회피) ===
            rsi_value = indicators.get("rsi", 50)
            if rsi_value > self.rsi_overbought or rsi_value < self.rsi_oversold:
                return {"action": "HOLD", "strength": 0.0, "reason": f"RSI 극값 회피: {rsi_value:.1f}"}
            
            # === 9. MACD 히스토그램 필터 (새로운 기능) ===
            macd_hist = indicators.get("macd_histogram", 0)
            if abs(macd_hist) < self.macd_hist_threshold:
                return {"action": "HOLD", "strength": 0.0, "reason": f"MACD 히스토그램 약함: {macd_hist:.6f}"}
            
            # === 10. 서포트/레지스턴스 필터 ===
            sr_check = self._check_support_resistance(price_data, current_price)
            if sr_check["near_resistance"]:
                return {"action": "HOLD", "strength": 0.0, "reason": "레지스턴스 근처"}
            
            # === 모든 조건 통과 - 신호 방향 결정 ===
            signal_direction = self._determine_signal_direction(breakout_signals, momentum_signals, indicators)
            
            # === 신뢰도 계산 (더 엄격한 기준) ===
            confidence = self._calculate_signal_confidence(
                breakout_signals, range_expansion, volume_confirmation, 
                momentum_signals, volatility, trend_strength if self.enable_regime_filter else 0.7
            )
            
            if confidence < self.min_confidence:
                return {"action": "HOLD", "strength": confidence, "reason": f"신뢰도 부족: {confidence:.3f}"}
            
            # === 성공적인 진입 신호 ===
            return {
                "action": signal_direction,
                "strength": confidence,
                "reason": "모든 조건 충족",
                "details": {
                    "breakout_strength": breakout_signals["strength"],
                    "volume_ratio": volume_confirmation["ratio"],
                    "momentum_score": momentum_signals["score"],
                    "volatility": volatility,
                    "trend_strength": trend_strength if self.enable_regime_filter else 0.7,
                    "risk_reward_ratio": self.take_profit_pct / self.stop_loss_pct
                }
            }
            
        except Exception as e:
            logger.error(f"진입 신호 평가 오류: {e}")
            return {"action": "HOLD", "strength": 0.0, "reason": f"평가 오류: {str(e)}"}
    
    def _calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """변동성 계산"""
        returns = price_data['close'].pct_change().dropna()
        return returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else 0.02
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """트렌드 강도 계산"""
        if len(price_data) < self.regime_window:
            return 0.5
        
        # 가격 기울기 계산
        recent_prices = price_data['close'].iloc[-self.regime_window:]
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        
        # 정규화된 트렌드 강도
        price_range = recent_prices.max() - recent_prices.min()
        trend_strength = abs(slope) / (price_range / self.regime_window) if price_range > 0 else 0
        
        return min(trend_strength, 1.0)
    
    def _check_higher_timeframe_trend(self, price_data: pd.DataFrame) -> bool:
        """상위 타임프레임 트렌드 확인"""
        if len(price_data) < self.higher_tf_period:
            return True  # 데이터 부족시 통과
        
        # 장기 이동평균 트렌드
        long_ma = price_data['close'].rolling(window=self.higher_tf_period).mean()
        current_price = price_data['close'].iloc[-1]
        
        # 현재 가격이 장기 MA 위에 있으면 상승 트렌드
        return current_price > long_ma.iloc[-1]
    
    def _check_breakout_conditions(self, price_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """강화된 브레이크아웃 조건 확인"""
        current_price = price_data['close'].iloc[-1]
        
        # 다중 기간 최고가/최저가 확인
        short_high = price_data['high'].rolling(window=self.breakout_window_short).max().iloc[-1]
        medium_high = price_data['high'].rolling(window=self.breakout_window_medium).max().iloc[-1]
        long_high = price_data['high'].rolling(window=self.breakout_window_long).max().iloc[-1]
        
        short_low = price_data['low'].rolling(window=self.breakout_window_short).min().iloc[-1]
        medium_low = price_data['low'].rolling(window=self.breakout_window_medium).min().iloc[-1]
        long_low = price_data['low'].rolling(window=self.breakout_window_long).min().iloc[-1]
        
        # 버퍼 적용
        upside_breakout = current_price > (short_high * (1 + self.breakout_buffer))
        downside_breakout = current_price < (short_low * (1 - self.breakout_buffer))
        
        # 더 강한 확인: 중기간 브레이크아웃도 필요
        medium_upside = current_price > medium_high
        medium_downside = current_price < medium_low
        
        strength = 0.0
        valid = False
        direction = None
        
        if upside_breakout and medium_upside:
            strength = min((current_price - short_high) / short_high / 0.01, 1.0)  # 1% 기준
            valid = True
            direction = "BUY"
        elif downside_breakout and medium_downside:
            strength = min((short_low - current_price) / short_low / 0.01, 1.0)
            valid = True
            direction = "SELL"
        
        return {
            "valid": valid,
            "direction": direction,
            "strength": strength,
            "short_breakout": upside_breakout or downside_breakout,
            "medium_confirmation": medium_upside or medium_downside
        }
    
    def _check_range_expansion(self, price_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """레인지 확장 조건 확인 (더 엄격한 기준)"""
        if len(price_data) < self.squeeze_window:
            return {"valid": False, "reason": "데이터 부족"}
        
        # ATR 기반 변동성 측정
        atr = indicators.get("atr", 0)
        if atr <= 0:
            return {"valid": False, "reason": "ATR 계산 실패"}
        
        # 최근 변동성 vs 과거 변동성
        recent_atr = atr
        historical_atr = price_data['close'].rolling(window=self.squeeze_window).apply(
            lambda x: np.std(x.pct_change().dropna()) * np.sqrt(252)
        ).mean()
        
        # 확장 비율 계산
        expansion_ratio = recent_atr / historical_atr if historical_atr > 0 else 0
        
        # 더 강한 확장 요구
        valid = expansion_ratio >= self.expansion_ratio
        
        return {
            "valid": valid,
            "expansion_ratio": expansion_ratio,
            "required_ratio": self.expansion_ratio,
            "recent_atr": recent_atr,
            "historical_atr": historical_atr
        }
    
    def _check_volume_conditions(self, price_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """거래량 조건 확인 (더 강한 요구사항)"""
        if 'volume' not in price_data.columns:
            return {"valid": True, "ratio": 1.0, "reason": "거래량 데이터 없음"}
        
        current_volume = price_data['volume'].iloc[-1]
        avg_volume = price_data['volume'].rolling(window=self.volume_ma_window).mean().iloc[-1]
        
        if avg_volume <= 0:
            return {"valid": False, "ratio": 0, "reason": "평균 거래량 0"}
        
        volume_ratio = current_volume / avg_volume
        
        # 더 강한 거래량 스파이크 요구
        spike_valid = volume_ratio >= self.volume_spike_ratio
        
        # 최소 거래량 비율 확인
        minimum_valid = volume_ratio >= self.min_volume_ratio
        
        # OBV 트렌드 확인
        obv = indicators.get("obv", 0)
        obv_trend = True  # 간단한 구현을 위해 일단 True
        
        valid = spike_valid and minimum_valid and obv_trend
        
        return {
            "valid": valid,
            "ratio": volume_ratio,
            "spike_valid": spike_valid,
            "minimum_valid": minimum_valid,
            "obv_trend": obv_trend,
            "required_spike": self.volume_spike_ratio,
            "required_minimum": self.min_volume_ratio
        }
    
    def _check_momentum_conditions(self, price_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """모멘텀 조건 확인 (개선된 필터)"""
        # RSI 확인
        rsi = indicators.get("rsi", 50)
        rsi_valid = rsi >= self.rsi_breakout_threshold
        
        # MACD 확인
        macd_line = indicators.get("macd_line", 0)
        macd_signal = indicators.get("macd_signal", 0)
        macd_hist = indicators.get("macd_histogram", 0)
        
        macd_bullish = macd_line > macd_signal and macd_hist > self.macd_hist_threshold
        macd_bearish = macd_line < macd_signal and macd_hist < -self.macd_hist_threshold
        macd_valid = macd_bullish or macd_bearish
        
        # 가격 모멘텀 확인
        if len(price_data) >= self.momentum_window:
            momentum = price_data['close'].pct_change(self.momentum_window).iloc[-1]
            momentum_valid = abs(momentum) >= self.momentum_threshold
        else:
            momentum = 0
            momentum_valid = False
        
        # 종합 점수
        score = 0.0
        if rsi_valid:
            score += 0.4
        if macd_valid:
            score += 0.4
        if momentum_valid:
            score += 0.2
        
        return {
            "valid": score >= 0.6,  # 적어도 2개 조건 충족
            "score": score,
            "rsi_valid": rsi_valid,
            "rsi_value": rsi,
            "macd_valid": macd_valid,
            "macd_bullish": macd_bullish,
            "macd_bearish": macd_bearish,
            "momentum_valid": momentum_valid,
            "momentum_value": momentum
        }
    
    def _check_support_resistance(self, price_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """서포트/레지스턴스 확인"""
        if len(price_data) < self.support_resistance_window:
            return {"near_resistance": False, "near_support": False}
        
        # 최근 고점/저점 식별
        highs = price_data['high'].rolling(window=self.support_resistance_window)
        lows = price_data['low'].rolling(window=self.support_resistance_window)
        
        recent_high = highs.max().iloc[-1]
        recent_low = lows.min().iloc[-1]
        
        # 현재가가 주요 레지스턴스 근처인지 확인
        resistance_distance = abs(current_price - recent_high) / current_price
        support_distance = abs(current_price - recent_low) / current_price
        
        near_resistance = resistance_distance <= self.sr_touch_threshold
        near_support = support_distance <= self.sr_touch_threshold
        
        return {
            "near_resistance": near_resistance,
            "near_support": near_support,
            "resistance_level": recent_high,
            "support_level": recent_low,
            "resistance_distance": resistance_distance,
            "support_distance": support_distance
        }
    
    def _determine_signal_direction(self, breakout_signals: Dict, momentum_signals: Dict, indicators: Dict) -> str:
        """신호 방향 결정"""
        # 브레이크아웃 방향 우선
        if breakout_signals.get("direction"):
            return breakout_signals["direction"]
        
        # 모멘텀 기반 방향
        if momentum_signals.get("macd_bullish"):
            return "BUY"
        elif momentum_signals.get("macd_bearish"):
            return "SELL"
        
        # 기본값
        return "BUY"
    
    def _calculate_signal_confidence(self, breakout_signals: Dict, range_expansion: Dict, 
                                   volume_confirmation: Dict, momentum_signals: Dict,
                                   volatility: float, trend_strength: float) -> float:
        """신호 신뢰도 계산 (더 엄격한 기준)"""
        confidence = 0.0
        
        # 브레이크아웃 강도 (30%)
        confidence += breakout_signals.get("strength", 0) * 0.3
        
        # 거래량 확인 (25%)
        volume_ratio = volume_confirmation.get("ratio", 1.0)
        volume_score = min(volume_ratio / self.volume_spike_ratio, 1.0)
        confidence += volume_score * 0.25
        
        # 모멘텀 점수 (25%)
        confidence += momentum_signals.get("score", 0) * 0.25
        
        # 트렌드 강도 (10%)
        confidence += trend_strength * 0.1
        
        # 변동성 적정성 (10%)
        vol_score = 1.0 - abs(volatility - 0.03) / 0.05  # 3% 변동성이 최적
        vol_score = max(0, min(vol_score, 1.0))
        confidence += vol_score * 0.1
        
        return min(confidence, 1.0)
    
    def calculate_exit_conditions(self, price_data: pd.DataFrame, position: Dict[str, Any], 
                                current_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """청산 조건 계산 (개선된 리스크 관리)"""
        try:
            current_price = price_data['close'].iloc[-1]
            entry_price = position.get("entry_price", current_price)
            entry_time = position.get("entry_time", 0)
            current_time = len(price_data) - 1
            position_side = position.get("side", "LONG")
            
            # 현재 수익률 계산
            if position_side == "LONG":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # 변동성 기반 동적 TP/SL
            volatility = self._calculate_volatility(price_data)
            dynamic_tp = self.take_profit_pct * (1 + volatility * self.vol_multiplier_tp)
            dynamic_sl = self.stop_loss_pct * (1 + volatility * self.vol_multiplier_sl)
            
            # 1. Take Profit 확인
            if pnl_pct >= dynamic_tp:
                return {
                    "should_exit": True,
                    "reason": "Take Profit 달성",
                    "exit_type": "PROFIT",
                    "pnl_pct": pnl_pct,
                    "target_pnl": dynamic_tp
                }
            
            # 2. Stop Loss 확인
            if pnl_pct <= -dynamic_sl:
                return {
                    "should_exit": True,
                    "reason": "Stop Loss 도달",
                    "exit_type": "STOP_LOSS",
                    "pnl_pct": pnl_pct,
                    "target_sl": -dynamic_sl
                }
            
            # 3. 트레일링 스톱 확인
            if pnl_pct >= self.trailing_stop_activation:
                # 최고 수익에서 일정 비율 하락시 청산
                max_profit = position.get("max_profit", pnl_pct)
                if pnl_pct < max_profit - self.trailing_stop_distance:
                    return {
                        "should_exit": True,
                        "reason": "트레일링 스톱",
                        "exit_type": "TRAILING",
                        "pnl_pct": pnl_pct,
                        "max_profit": max_profit
                    }
            
            # 4. 최대 보유 기간 확인
            holding_period = current_time - entry_time
            if holding_period >= self.max_hold_bars:
                return {
                    "should_exit": True,
                    "reason": "최대 보유 기간 초과",
                    "exit_type": "TIME_LIMIT",
                    "pnl_pct": pnl_pct,
                    "holding_period": holding_period
                }
            
            # 5. 모멘텀 소진 확인
            momentum_exhausted = self._check_momentum_exhaustion(price_data, current_indicators, position_side)
            if momentum_exhausted:
                return {
                    "should_exit": True,
                    "reason": "모멘텀 소진",
                    "exit_type": "MOMENTUM",
                    "pnl_pct": pnl_pct
                }
            
            # 6. 브레이크아웃 실패 확인
            breakout_failed = self._check_breakout_failure(price_data, position)
            if breakout_failed:
                return {
                    "should_exit": True,
                    "reason": "브레이크아웃 실패",
                    "exit_type": "FAILURE",
                    "pnl_pct": pnl_pct
                }
            
            # 포지션 유지
            return {
                "should_exit": False,
                "reason": "조건 미충족",
                "pnl_pct": pnl_pct,
                "dynamic_tp": dynamic_tp,
                "dynamic_sl": dynamic_sl,
                "holding_period": holding_period
            }
            
        except Exception as e:
            logger.error(f"청산 조건 계산 오류: {e}")
            return {
                "should_exit": True,
                "reason": f"계산 오류: {str(e)}",
                "exit_type": "ERROR",
                "pnl_pct": 0
            }
    
    def _check_momentum_exhaustion(self, price_data: pd.DataFrame, indicators: Dict, position_side: str) -> bool:
        """모멘텀 소진 확인"""
        rsi = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        
        if position_side == "LONG":
            # 롱 포지션: RSI 과매수 or MACD 히스토그램 하락
            return rsi > 75 or macd_hist < 0
        else:
            # 숏 포지션: RSI 과매도 or MACD 히스토그램 상승
            return rsi < 25 or macd_hist > 0
    
    def _check_breakout_failure(self, price_data: pd.DataFrame, position: Dict) -> bool:
        """브레이크아웃 실패 확인"""
        entry_price = position.get("entry_price", 0)
        current_price = price_data['close'].iloc[-1]
        position_side = position.get("side", "LONG")
        
        # 간단한 실패 기준: 진입가 대비 일정 비율 이상 역행
        failure_threshold = 0.01  # 1%
        
        if position_side == "LONG":
            return (entry_price - current_price) / entry_price > failure_threshold
        else:
            return (current_price - entry_price) / entry_price > failure_threshold


# 편의 함수
def create_optimized_strategy() -> OptimizedRuleStrategyE:
    """최적화된 전략 인스턴스 생성"""
    return OptimizedRuleStrategyE()


if __name__ == "__main__":
    # 테스트
    strategy = create_optimized_strategy()
    print(f"전략 초기화 완료: {strategy.name}")
    print(f"Risk/Reward 비율: {strategy.take_profit_pct/strategy.stop_loss_pct:.2f}:1")
    print(f"성능 목표:")
    for key, value in strategy.performance_targets.items():
        print(f"  - {key}: {value}")