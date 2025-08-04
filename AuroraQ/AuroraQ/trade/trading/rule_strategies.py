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


# VPS 배포 시스템 경로 설정
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        
        # Profit Factor 추적
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.profit_factor = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        
        # 기본 설정 로드
        self.config = load_config(component_type="production")
        
        # 지표 캐시
        self._indicator_cache = {}
        
        logger.info(f"{self.name} 전략 초기화 완료")
        
        # Rule 전략 로깅 시스템 초기화
        try:
            from trading.rule_strategy_logger import get_rule_strategy_logger
            self.rule_logger = get_rule_strategy_logger()
        except ImportError:
            self.rule_logger = None
            logger.warning(f"{self.name}: Rule 전략 로거를 불러올 수 없습니다")
    
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
    
    def _log_strategy_signal(self, signal: Dict[str, Any], selected: bool = False, 
                           selection_rank: int = 0) -> None:
        """전략 신호 로깅"""
        try:
            if not self.rule_logger:
                return
            
            # 점수 계산
            metadata = signal.get('metadata', {})
            composite_score = metadata.get('composite_score', 0.0)
            confidence = metadata.get('confidence', 0.5)
            
            # 로깅 실행
            self.rule_logger.log_strategy_signal(
                strategy_name=self.name,
                action=signal.get('action', 'HOLD'),
                strength=signal.get('strength', 0.0),
                strategy_score=composite_score,
                confidence=confidence,
                selected=selected,
                selection_rank=selection_rank,
                total_trades=self.position_count,
                win_rate=self.win_rate,
                profit_factor=self.profit_factor,
                total_pnl=self.total_pnl
            )
            
        except Exception as e:
            logger.warning(f"{self.name} 신호 로깅 실패: {e}")
    
    def add_market_outcome(self, outcome: float) -> None:
        """시장 결과 피드백 추가 (성과 업데이트)"""
        try:
            self.position_count += 1
            self.total_pnl += outcome
            
            if outcome > 0:
                self.winning_trades += 1
                self.total_profit += outcome
            else:
                self.losing_trades += 1
                self.total_loss += abs(outcome)
            
            # Win rate 업데이트
            if self.position_count > 0:
                self.win_rate = self.winning_trades / self.position_count
            
            # Profit factor 업데이트
            if self.total_loss > 0:
                self.profit_factor = self.total_profit / self.total_loss
            else:
                self.profit_factor = float('inf') if self.total_profit > 0 else 1.0
            
            logger.debug(f"{self.name} 성과 업데이트: outcome={outcome:.3f}, "
                        f"win_rate={self.win_rate:.3f}, profit_factor={self.profit_factor:.3f}")
            
        except Exception as e:
            logger.error(f"{self.name} 시장 결과 처리 실패: {e}")

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
    
    def score(self, price_data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """전략A 점수 계산: EMA 크로스오버 + ADX 트렌드 강도"""
        try:
            scores = {}
            
            if len(price_data) < max(self.ema_long_len, self.adx_window + 1):
                return 0.0, scores
            
            # 1. EMA 크로스오버 점수
            ema_short = self.get_cached_indicator("ema", price_data, period=self.ema_short_len)
            ema_long = self.get_cached_indicator("ema", price_data, period=self.ema_long_len)
            
            if ema_short is not None and ema_long is not None:
                ema_short_val = float(ema_short.iloc[-1])
                ema_long_val = float(ema_long.iloc[-1])
                ema_diff_pct = (ema_short_val - ema_long_val) / ema_long_val
                
                if ema_diff_pct > 0:
                    scores['ema_cross'] = min(1.0, ema_diff_pct * 100)  # 0-1 정규화
                else:
                    scores['ema_cross'] = 0.0
            else:
                scores['ema_cross'] = 0.0
            
            # 2. ADX 트렌드 강도 점수
            adx = self.get_cached_indicator("adx", price_data, period=self.adx_window)
            if adx is not None:
                adx_val = float(adx.iloc[-1])
                scores['adx_strength'] = min(1.0, max(0.0, (adx_val - 20) / 30))  # 20-50 범위를 0-1로
            else:
                scores['adx_strength'] = 0.0
            
            # 3. 가격 모멘텀 점수
            current_price = self.safe_last(price_data, "close")
            if len(price_data) >= 5:
                price_change_5 = (current_price - price_data['close'].iloc[-5]) / price_data['close'].iloc[-5]
                scores['momentum'] = min(1.0, max(0.0, price_change_5 * 20 + 0.5))  # -2.5%~+2.5% → 0~1
            else:
                scores['momentum'] = 0.5
            
            # 4. 거래량 점수
            if "volume" in price_data.columns and len(price_data) >= 20:
                avg_volume = price_data["volume"].tail(20).mean()
                current_volume = price_data["volume"].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                scores['volume'] = min(1.0, max(0.0, (volume_ratio - 0.5) / 1.5))  # 0.5~2.0 → 0~1
            else:
                scores['volume'] = 0.5
            
            # 종합 점수 계산 (가중 평균)
            weights = {
                'ema_cross': 0.4,
                'adx_strength': 0.3,
                'momentum': 0.2,
                'volume': 0.1
            }
            
            composite_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())
            
            return composite_score, scores
            
        except Exception as e:
            logger.error(f"RuleStrategyA 점수 계산 오류: {e}")
            return 0.0, {}
    
    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """VPS용 신호 생성 메서드"""
        try:
            if len(price_data) < 50:
                return {
                    "action": "HOLD",
                    "strength": 0.0,
                    "price": self.safe_last(price_data, "close"),
                    "metadata": {"reason": "insufficient_data", "strategy": "RuleStrategyA"}
                }
            
            current_price = self.safe_last(price_data, "close")
            
            # 진입 신호 확인
            entry_signal = self.should_enter(price_data)
            if entry_signal:
                # 점수 계산으로 신호 강도 보정
                composite_score, detail_scores = self.score(price_data)
                
                return {
                    "action": "BUY" if entry_signal["side"] == "LONG" else "SELL",
                    "strength": min(entry_signal.get("confidence", 0.5) * composite_score, 1.0),
                    "price": current_price,
                    "metadata": {
                        "strategy": "RuleStrategyA",
                        "reason": entry_signal.get("reason", ""),
                        "confidence": entry_signal.get("confidence", 0.5),
                        "composite_score": composite_score,
                        "detail_scores": detail_scores,
                        "stop_loss": entry_signal.get("stop_loss"),
                        "take_profit": entry_signal.get("take_profit")
                    }
                }
            
            # 신호 없음
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": current_price,
                "metadata": {"strategy": "RuleStrategyA", "reason": "no_signal"}
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyA 신호 생성 오류: {e}")
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": self.safe_last(price_data, "close") if len(price_data) > 0 else 0.0,
                "metadata": {"error": str(e), "strategy": "RuleStrategyA"}
            }

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
    
    def score(self, price_data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """전략B 점수 계산: 거래량 스파이크 + RSI + 이동평균"""
        try:
            scores = {}
            
            if len(price_data) < max(self.ma_long_len, self.rsi_period + 1, 20):
                return 0.0, scores
            
            # 1. 이동평균 크로스오버 점수
            ma_short = self.get_cached_indicator("sma", price_data, period=self.ma_short_len)
            ma_long = self.get_cached_indicator("sma", price_data, period=self.ma_long_len)
            
            if ma_short is not None and ma_long is not None:
                ma_short_val = float(ma_short.iloc[-1])
                ma_long_val = float(ma_long.iloc[-1])
                ma_diff_pct = (ma_short_val - ma_long_val) / ma_long_val
                
                if ma_diff_pct > 0:
                    scores['ma_cross'] = min(1.0, ma_diff_pct * 50)  # 0-1 정규화
                else:
                    scores['ma_cross'] = 0.0
            else:
                scores['ma_cross'] = 0.0
            
            # 2. RSI 점수 (과매도/과매수 영역 피하기)
            rsi = self.get_cached_indicator("rsi", price_data, period=self.rsi_period)
            if rsi is not None:
                rsi_val = float(rsi.iloc[-1])
                # 30-70 구간에서 최고 점수, 극단적 값에서 점수 감소
                if 30 <= rsi_val <= 70:
                    scores['rsi'] = 1.0 - abs(rsi_val - 50) / 20  # 50에서 최고, 30/70에서 최저
                elif rsi_val < 30:
                    scores['rsi'] = 0.3  # 과매도에서 반등 기대
                else:  # rsi_val > 70
                    scores['rsi'] = 0.1  # 과매수에서 하락 위험
            else:
                scores['rsi'] = 0.5
            
            # 3. 거래량 스파이크 점수
            if "volume" in price_data.columns and len(price_data) >= 20:
                avg_volume = price_data["volume"].tail(20).mean()
                current_volume = price_data["volume"].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                # 거래량 비율이 높을수록 좋음
                if volume_ratio >= self.volume_spike_ratio:
                    scores['volume_spike'] = min(1.0, (volume_ratio - 1.0) / 2.0)  # 1.5이상에서 고득점
                else:
                    scores['volume_spike'] = max(0.0, (volume_ratio - 0.5) / 1.0)  # 0.5 이하에서 0점
            else:
                scores['volume_spike'] = 0.3
            
            # 4. 가격 모멘텀 점수
            current_price = self.safe_last(price_data, "close")
            if len(price_data) >= 3:
                price_change_3 = (current_price - price_data['close'].iloc[-3]) / price_data['close'].iloc[-3]
                scores['momentum'] = min(1.0, max(0.0, price_change_3 * 30 + 0.5))  # -1.7%~+1.7% → 0~1
            else:
                scores['momentum'] = 0.5
            
            # 종합 점수 계산 (가중 평균)
            weights = {
                'ma_cross': 0.3,
                'rsi': 0.3,
                'volume_spike': 0.3,
                'momentum': 0.1
            }
            
            composite_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())
            
            return composite_score, scores
            
        except Exception as e:
            logger.error(f"RuleStrategyB 점수 계산 오류: {e}")
            return 0.0, {}
    
    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """VPS용 신호 생성 메서드"""
        try:
            if len(price_data) < 50:
                return {
                    "action": "HOLD",
                    "strength": 0.0,
                    "price": self.safe_last(price_data, "close"),
                    "metadata": {"reason": "insufficient_data", "strategy": "RuleStrategyB"}
                }
            
            current_price = self.safe_last(price_data, "close")
            
            # 진입 신호 확인
            entry_signal = self.should_enter(price_data)
            if entry_signal:
                # 점수 계산으로 신호 강도 보정
                composite_score, detail_scores = self.score(price_data)
                
                return {
                    "action": "BUY" if entry_signal["side"] == "LONG" else "SELL",
                    "strength": min(entry_signal.get("confidence", 0.5) * composite_score, 1.0),
                    "price": current_price,
                    "metadata": {
                        "strategy": "RuleStrategyB",
                        "reason": entry_signal.get("reason", ""),
                        "confidence": entry_signal.get("confidence", 0.5),
                        "composite_score": composite_score,
                        "detail_scores": detail_scores,
                        "stop_loss": entry_signal.get("stop_loss"),
                        "take_profit": entry_signal.get("take_profit")
                    }
                }
            
            # 신호 없음
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": current_price,
                "metadata": {"strategy": "RuleStrategyB", "reason": "no_signal"}
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyB 신호 생성 오류: {e}")
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": self.safe_last(price_data, "close") if len(price_data) > 0 else 0.0,
                "metadata": {"error": str(e), "strategy": "RuleStrategyB"}
            }

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
    
    def score(self, price_data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """전략C 점수 계산: 볼린저밴드 + 스토캐스틱"""
        try:
            scores = {}
            
            if len(price_data) < max(self.bb_period, self.stoch_k_period + 1):
                return 0.0, scores
            
            current_price = self.safe_last(price_data, "close")
            
            # 1. 볼린저밴드 위치 점수
            bb = self.get_cached_indicator("bollinger", price_data, period=self.bb_period, std_dev=self.bb_std_dev)
            if bb is not None:
                bb_lower = float(bb['lower'].iloc[-1])
                bb_middle = float(bb['middle'].iloc[-1])
                bb_upper = float(bb['upper'].iloc[-1])
                
                # 밴드 내에서의 위치 계산 (0: 하단, 0.5: 중간, 1: 상단)
                if bb_upper > bb_lower:
                    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                    bb_position = max(0.0, min(1.0, bb_position))
                    
                    # 하단 및 중간 근처에서 고득점, 상단에서 저득점
                    if bb_position <= 0.3:  # 하단 근처
                        scores['bb_position'] = 0.8 + (0.3 - bb_position) * 0.7  # 0.8~1.0
                    elif bb_position <= 0.7:  # 중간 영역
                        scores['bb_position'] = 0.4 + (0.7 - abs(bb_position - 0.5)) * 0.8  # 0.4~0.8
                    else:  # 상단 근처
                        scores['bb_position'] = max(0.1, 0.5 - (bb_position - 0.7) * 1.3)  # 0.1~0.5
                else:
                    scores['bb_position'] = 0.5
            else:
                scores['bb_position'] = 0.5
            
            # 2. 스토캐스틱 점수
            stoch = self.get_cached_indicator("stochastic", price_data, k_period=self.stoch_k_period, d_period=self.stoch_d_period)
            if stoch is not None:
                stoch_k = float(stoch['%K'].iloc[-1])
                stoch_d = float(stoch['%D'].iloc[-1])
                
                # 과매도 영역에서 상승 전환 신호
                if stoch_k <= self.stoch_oversold and stoch_k > stoch_d:
                    scores['stoch'] = 0.9  # 과매도에서 상승 전환
                elif stoch_k <= self.stoch_oversold:
                    scores['stoch'] = 0.7  # 과매도 영역
                elif stoch_k >= self.stoch_overbought:
                    scores['stoch'] = 0.2  # 과매수 영역
                else:
                    scores['stoch'] = 0.5  # 중립 영역
            else:
                scores['stoch'] = 0.5
            
            # 3. 볼린저밴드 폭 점수 (변동성 측정)
            if bb is not None:
                bb_width = (bb_upper - bb_lower) / bb_middle
                # 좌은 밴드는 러배운스, 넓은 밴드는 안정 신호
                if bb_width < 0.02:  # 2% 이하
                    scores['bb_squeeze'] = 0.8  # 러배운스 준비
                elif bb_width < 0.05:  # 5% 이하
                    scores['bb_squeeze'] = 0.6
                else:
                    scores['bb_squeeze'] = 0.4
            else:
                scores['bb_squeeze'] = 0.5
            
            # 4. 가격 모멘텀 점수
            if len(price_data) >= 5:
                price_change_5 = (current_price - price_data['close'].iloc[-5]) / price_data['close'].iloc[-5]
                scores['momentum'] = min(1.0, max(0.0, price_change_5 * 25 + 0.5))  # -2%~+2% → 0~1
            else:
                scores['momentum'] = 0.5
            
            # 종합 점수 계산 (가중 평균)
            weights = {
                'bb_position': 0.4,   # 볼린저밴드 위치
                'stoch': 0.3,         # 스토캐스틱
                'bb_squeeze': 0.2,    # 볼린저밴드 압축
                'momentum': 0.1       # 모멘텀
            }
            
            composite_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())
            
            return composite_score, scores
            
        except Exception as e:
            logger.error(f"RuleStrategyC 점수 계산 오류: {e}")
            return 0.0, {}
    
    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """VPS용 신호 생성 메서드"""
        try:
            if len(price_data) < 50:
                return {
                    "action": "HOLD",
                    "strength": 0.0,
                    "price": self.safe_last(price_data, "close"),
                    "metadata": {"reason": "insufficient_data", "strategy": "RuleStrategyC"}
                }
            
            current_price = self.safe_last(price_data, "close")
            
            # 진입 신호 확인
            entry_signal = self.should_enter(price_data)
            if entry_signal:
                # 점수 계산으로 신호 강도 보정
                composite_score, detail_scores = self.score(price_data)
                
                return {
                    "action": "BUY" if entry_signal["side"] == "LONG" else "SELL",
                    "strength": min(entry_signal.get("confidence", 0.5) * composite_score, 1.0),
                    "price": current_price,
                    "metadata": {
                        "strategy": "RuleStrategyC",
                        "reason": entry_signal.get("reason", ""),
                        "confidence": entry_signal.get("confidence", 0.5),
                        "composite_score": composite_score,
                        "detail_scores": detail_scores,
                        "stop_loss": entry_signal.get("stop_loss"),
                        "take_profit": entry_signal.get("take_profit")
                    }
                }
            
            # 신호 없음
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": current_price,
                "metadata": {"strategy": "RuleStrategyC", "reason": "no_signal"}
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyC 신호 생성 오류: {e}")
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": self.safe_last(price_data, "close") if len(price_data) > 0 else 0.0,
                "metadata": {"error": str(e), "strategy": "RuleStrategyC"}
            }

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
    
    def score(self, price_data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """전략D 점수 계산: MACD + 이동평균 수렴/발산"""
        try:
            scores = {}
            
            if len(price_data) < max(self.macd_slow, self.ma_period):
                return 0.0, scores
            
            current_price = self.safe_last(price_data, "close")
            
            # 1. MACD 골든 크로스 점수
            macd_data = self.get_cached_indicator("macd", price_data, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd_data is not None and len(macd_data['macd']) >= 2:
                macd_val = float(macd_data['macd'].iloc[-1])
                signal_val = float(macd_data['signal'].iloc[-1])
                prev_macd = float(macd_data['macd'].iloc[-2])
                prev_signal = float(macd_data['signal'].iloc[-2])
                
                # MACD 골든 크로스 및 상승 모멘텀
                if macd_val > signal_val:
                    if prev_macd <= prev_signal:  # 방금 골든 크로스
                        scores['macd_cross'] = 1.0
                    else:  # 이미 골든 크로스 상태
                        scores['macd_cross'] = 0.8
                else:
                    scores['macd_cross'] = max(0.0, (macd_val - signal_val) / abs(signal_val) + 0.5) if signal_val != 0 else 0.3
                
                # MACD 히스토그램 점수 (모멘텀)
                histogram = macd_val - signal_val
                if len(macd_data['macd']) >= 3:
                    prev_histogram = prev_macd - prev_signal
                    if histogram > 0 and histogram > prev_histogram:  # 상승 모멘텀
                        scores['macd_momentum'] = min(1.0, abs(histogram) * 1000 + 0.6)
                    elif histogram > 0:  # 양수지만 약화
                        scores['macd_momentum'] = 0.5
                    else:  # 음수
                        scores['macd_momentum'] = max(0.0, 0.4 + histogram * 500)
                else:
                    scores['macd_momentum'] = 0.5 if histogram > 0 else 0.3
            else:
                scores['macd_cross'] = 0.5
                scores['macd_momentum'] = 0.5
            
            # 2. 이동평균 위치 점수
            ma = self.get_cached_indicator("sma", price_data, period=self.ma_period)
            if ma is not None:
                ma_val = float(ma.iloc[-1])
                price_ma_ratio = current_price / ma_val if ma_val > 0 else 1.0
                
                # 이동평균 위에 있을 때 가점
                if price_ma_ratio > 1.0:
                    scores['ma_position'] = min(1.0, 0.6 + (price_ma_ratio - 1.0) * 20)  # 1%당 0.2점 가산
                else:
                    scores['ma_position'] = max(0.0, 0.4 * price_ma_ratio)  # 이평선 아래는 감점
            else:
                scores['ma_position'] = 0.5
            
            # 3. 이동평균 기울기 점수 (트렌드 방향)
            if ma is not None and len(ma) >= 5:
                ma_current = float(ma.iloc[-1])
                ma_prev = float(ma.iloc[-5])
                ma_slope = (ma_current - ma_prev) / ma_prev if ma_prev > 0 else 0.0
                
                # 상승 기울기일 때 가점
                if ma_slope > 0:
                    scores['ma_trend'] = min(1.0, 0.5 + ma_slope * 100)  # 1% 상승당 1점
                else:
                    scores['ma_trend'] = max(0.0, 0.5 + ma_slope * 50)   # 하락 시 감점
            else:
                scores['ma_trend'] = 0.5
            
            # 4. 가격 모멘텀 점수
            if len(price_data) >= 3:
                price_change_3 = (current_price - price_data['close'].iloc[-3]) / price_data['close'].iloc[-3]
                scores['price_momentum'] = min(1.0, max(0.0, price_change_3 * 50 + 0.5))  # -1%~+1% → 0~1
            else:
                scores['price_momentum'] = 0.5
            
            # 종합 점수 계산 (가중 평균)
            weights = {
                'macd_cross': 0.35,        # MACD 크로스 신호
                'macd_momentum': 0.25,     # MACD 모멘텀
                'ma_position': 0.2,        # 이동평균 위치
                'ma_trend': 0.15,          # 이동평균 기울기
                'price_momentum': 0.05     # 단기 모멘텀
            }
            
            composite_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())
            
            return composite_score, scores
            
        except Exception as e:
            logger.error(f"RuleStrategyD 점수 계산 오류: {e}")
            return 0.0, {}
    
    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """VPS용 신호 생성 메서드"""
        try:
            if len(price_data) < 50:
                return {
                    "action": "HOLD",
                    "strength": 0.0,
                    "price": self.safe_last(price_data, "close"),
                    "metadata": {"reason": "insufficient_data", "strategy": "RuleStrategyD"}
                }
            
            current_price = self.safe_last(price_data, "close")
            
            # 진입 신호 확인
            entry_signal = self.should_enter(price_data)
            if entry_signal:
                # 점수 계산으로 신호 강도 보정
                composite_score, detail_scores = self.score(price_data)
                
                return {
                    "action": "BUY" if entry_signal["side"] == "LONG" else "SELL",
                    "strength": min(entry_signal.get("confidence", 0.5) * composite_score, 1.0),
                    "price": current_price,
                    "metadata": {
                        "strategy": "RuleStrategyD",
                        "reason": entry_signal.get("reason", ""),
                        "confidence": entry_signal.get("confidence", 0.5),
                        "composite_score": composite_score,
                        "detail_scores": detail_scores,
                        "stop_loss": entry_signal.get("stop_loss"),
                        "take_profit": entry_signal.get("take_profit")
                    }
                }
            
            # 신호 없음
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": current_price,
                "metadata": {"strategy": "RuleStrategyD", "reason": "no_signal"}
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyD 신호 생성 오류: {e}")
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": self.safe_last(price_data, "close") if len(price_data) > 0 else 0.0,
                "metadata": {"error": str(e), "strategy": "RuleStrategyD"}
            }

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
    
    def score(self, price_data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """전략E 점수 계산 (일관성을 위한 score 메서드)"""
        return self._calculate_composite_score(price_data)
    
    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """VPS용 신호 생성 메서드"""
        try:
            if len(price_data) < 50:
                return {
                    "action": "HOLD",
                    "strength": 0.0,
                    "price": self.safe_last(price_data, "close"),
                    "metadata": {"reason": "insufficient_data", "strategy": "RuleStrategyE"}
                }
            
            current_price = self.safe_last(price_data, "close")
            
            # 진입 신호 확인
            entry_signal = self.should_enter(price_data)
            if entry_signal:
                # 점수 계산으로 신호 강도 보정
                composite_score, detail_scores = self.score(price_data)
                
                return {
                    "action": "BUY" if entry_signal["side"] == "LONG" else "SELL",
                    "strength": min(entry_signal.get("confidence", 0.5) * composite_score, 1.0),
                    "price": current_price,
                    "metadata": {
                        "strategy": "RuleStrategyE",
                        "reason": entry_signal.get("reason", ""),
                        "confidence": entry_signal.get("confidence", 0.5),
                        "composite_score": composite_score,
                        "detail_scores": detail_scores,
                        "stop_loss": entry_signal.get("stop_loss"),
                        "take_profit": entry_signal.get("take_profit")
                    }
                }
            
            # 신호 없음
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": current_price,
                "metadata": {"strategy": "RuleStrategyE", "reason": "no_signal"}
            }
            
        except Exception as e:
            logger.error(f"RuleStrategyE 신호 생성 오류: {e}")
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": self.safe_last(price_data, "close") if len(price_data) > 0 else 0.0,
                "metadata": {"error": str(e), "strategy": "RuleStrategyE"}
            }

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