"""
시장 체제 감지기 - Bull/Bear/Sideways 시장 분류
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """시장 체제 분류"""
    BULL = "bull"          # 상승장
    BEAR = "bear"          # 하락장  
    SIDEWAYS = "sideways"  # 횡보장
    VOLATILE = "volatile"   # 고변동성장
    UNKNOWN = "unknown"     # 불명확


@dataclass
class RegimeSignal:
    """체제 분석 결과"""
    regime: MarketRegime
    confidence: float      # 0~1, 신뢰도
    strength: float        # 0~1, 강도
    duration_days: int     # 지속 기간
    next_regime_prob: Dict[MarketRegime, float]  # 다음 체제 전환 확률


class RegimeDetector:
    """
    시장 체제 감지 및 분류
    - 다중 지표 기반 체제 분류
    - 머신러닝 기반 체제 전환 예측
    - 자산별 체제 분석
    """
    
    def __init__(
        self,
        lookback_window: int = 252,  # 1년
        trend_window: int = 50,      # 추세 판단 윈도우
        volatility_window: int = 20   # 변동성 윈도우
    ):
        self.lookback_window = lookback_window
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        
        # 체제 분류 임계값
        self.thresholds = {
            'bull_trend': 0.15,      # 연 15% 이상 상승시 강세장
            'bear_trend': -0.15,     # 연 15% 이상 하락시 약세장
            'high_volatility': 0.25,  # 연 변동성 25% 이상시 고변동성
            'sideways_range': 0.10    # ±10% 범위 내 횡보장
        }
        
        # 히스토리 저장
        self.regime_history = {}
        
    def detect_regime(
        self,
        price_data: pd.DataFrame,
        asset_name: str = "ASSET"
    ) -> RegimeSignal:
        """
        현재 시장 체제 감지
        
        Args:
            price_data: OHLCV 데이터
            asset_name: 자산명
            
        Returns:
            RegimeSignal: 체제 분석 결과
        """
        if len(price_data) < self.lookback_window:
            logger.warning(f"Insufficient data for {asset_name}: {len(price_data)} < {self.lookback_window}")
            return RegimeSignal(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                strength=0.0, 
                duration_days=0,
                next_regime_prob={}
            )
        
        # 1. 추세 분석
        trend_signal = self._analyze_trend(price_data)
        
        # 2. 변동성 분석  
        volatility_signal = self._analyze_volatility(price_data)
        
        # 3. 모멘텀 분석
        momentum_signal = self._analyze_momentum(price_data)
        
        # 4. 체제 분류
        regime = self._classify_regime(trend_signal, volatility_signal, momentum_signal)
        
        # 5. 신뢰도 계산
        confidence = self._calculate_confidence(trend_signal, volatility_signal, momentum_signal)
        
        # 6. 강도 계산
        strength = self._calculate_strength(trend_signal, volatility_signal)
        
        # 7. 지속 기간 추정
        duration = self._estimate_duration(price_data, regime)
        
        # 8. 다음 체제 전환 확률
        next_prob = self._predict_regime_transition(price_data, regime)
        
        result = RegimeSignal(
            regime=regime,
            confidence=confidence,
            strength=strength,
            duration_days=duration,
            next_regime_prob=next_prob
        )
        
        # 히스토리 업데이트
        self._update_history(asset_name, result)
        
        return result
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, float]:
        """추세 분석"""
        prices = data['close']
        
        # 1. 이동평균 기반 추세
        ma_short = prices.rolling(20).mean()
        ma_long = prices.rolling(50).mean()
        ma_trend = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        
        # 2. 선형 회귀 기반 추세
        x = np.arange(len(prices))
        y = prices.values
        slope = np.polyfit(x[-self.trend_window:], y[-self.trend_window:], 1)[0]
        linear_trend = slope / prices.iloc[-1] * 252  # 연환산
        
        # 3. 가격 위치 (최근 최고/최저 대비)
        recent_high = prices.rolling(self.trend_window).max().iloc[-1]
        recent_low = prices.rolling(self.trend_window).min().iloc[-1]
        current_price = prices.iloc[-1]
        
        if recent_high != recent_low:
            price_position = (current_price - recent_low) / (recent_high - recent_low)
        else:
            price_position = 0.5
        
        return {
            'ma_trend': ma_trend,
            'linear_trend': linear_trend,
            'price_position': price_position
        }
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """변동성 분석"""
        returns = data['close'].pct_change().dropna()
        
        # 1. 현재 변동성
        current_vol = returns.rolling(self.volatility_window).std().iloc[-1] * np.sqrt(252)
        
        # 2. 역사적 변동성 대비
        historical_vol = returns.rolling(252).std().mean() * np.sqrt(252)
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # 3. VIX 스타일 변동성 (GARCH 대신 단순화)
        vol_forecast = self._simple_vol_forecast(returns)
        
        return {
            'current_volatility': current_vol,
            'volatility_ratio': vol_ratio,
            'volatility_forecast': vol_forecast
        }
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """모멘텀 분석"""
        prices = data['close']
        
        # 1. RSI
        rsi = self._calculate_rsi(prices)
        
        # 2. MACD
        macd, signal = self._calculate_macd(prices)
        macd_signal = (macd - signal) / prices.iloc[-1]
        
        # 3. 가격 모멘텀
        mom_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 21 else 0
        mom_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) if len(prices) > 63 else 0
        
        return {
            'rsi': rsi,
            'macd_signal': macd_signal,
            'momentum_1m': mom_1m,
            'momentum_3m': mom_3m
        }
    
    def _classify_regime(
        self,
        trend: Dict[str, float],
        volatility: Dict[str, float],
        momentum: Dict[str, float]
    ) -> MarketRegime:
        """체제 분류 로직"""
        
        # 고변동성 우선 체크
        if volatility['current_volatility'] > self.thresholds['high_volatility']:
            return MarketRegime.VOLATILE
        
        # 추세 기반 분류
        trend_score = (
            trend['linear_trend'] * 0.4 +
            trend['ma_trend'] * 0.3 +
            (trend['price_position'] - 0.5) * 2 * 0.3
        )
        
        if trend_score > self.thresholds['bull_trend']:
            return MarketRegime.BULL
        elif trend_score < self.thresholds['bear_trend']:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    def _calculate_confidence(
        self,
        trend: Dict[str, float],
        volatility: Dict[str, float], 
        momentum: Dict[str, float]
    ) -> float:
        """신뢰도 계산"""
        
        # 지표 간 일관성
        trend_consistency = 1.0 - np.std([
            trend['linear_trend'],
            trend['ma_trend'],
            (trend['price_position'] - 0.5) * 2
        ])
        
        # 변동성 안정성 (낮은 변동성일수록 높은 신뢰도)
        vol_stability = 1.0 / (1.0 + volatility['volatility_ratio'])
        
        # 모멘텀 강도
        momentum_strength = min(1.0, abs(momentum['momentum_1m']) * 2)
        
        confidence = (
            trend_consistency * 0.4 +
            vol_stability * 0.3 +
            momentum_strength * 0.3
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def _calculate_strength(
        self,
        trend: Dict[str, float],
        volatility: Dict[str, float]
    ) -> float:
        """체제 강도 계산"""
        trend_strength = abs(trend['linear_trend'])
        vol_strength = min(1.0, volatility['current_volatility'] / 0.5)
        
        strength = (trend_strength + vol_strength) / 2
        return float(np.clip(strength, 0, 1))
    
    def _estimate_duration(self, data: pd.DataFrame, regime: MarketRegime) -> int:
        """체제 지속 기간 추정"""
        # 간단한 휴리스틱 (실제로는 더 복잡한 모델 필요)
        regime_durations = {
            MarketRegime.BULL: 180,      # 6개월
            MarketRegime.BEAR: 120,      # 4개월
            MarketRegime.SIDEWAYS: 90,   # 3개월
            MarketRegime.VOLATILE: 30,   # 1개월
            MarketRegime.UNKNOWN: 0
        }
        
        return regime_durations.get(regime, 0)
    
    def _predict_regime_transition(
        self,
        data: pd.DataFrame,
        current_regime: MarketRegime
    ) -> Dict[MarketRegime, float]:
        """체제 전환 확률 예측"""
        # 간단한 마르코프 체인 기반 (실제로는 ML 모델 사용)
        transition_matrix = {
            MarketRegime.BULL: {
                MarketRegime.BULL: 0.7,
                MarketRegime.SIDEWAYS: 0.2,
                MarketRegime.BEAR: 0.05,
                MarketRegime.VOLATILE: 0.05
            },
            MarketRegime.BEAR: {
                MarketRegime.BEAR: 0.6,
                MarketRegime.SIDEWAYS: 0.25,
                MarketRegime.BULL: 0.1,
                MarketRegime.VOLATILE: 0.05
            },
            MarketRegime.SIDEWAYS: {
                MarketRegime.SIDEWAYS: 0.5,
                MarketRegime.BULL: 0.25,
                MarketRegime.BEAR: 0.2,
                MarketRegime.VOLATILE: 0.05
            },
            MarketRegime.VOLATILE: {
                MarketRegime.VOLATILE: 0.3,
                MarketRegime.SIDEWAYS: 0.3,
                MarketRegime.BULL: 0.2,
                MarketRegime.BEAR: 0.2
            }
        }
        
        return transition_matrix.get(current_regime, {})
    
    def _simple_vol_forecast(self, returns: pd.Series) -> float:
        """간단한 변동성 예측"""
        # EWMA 기반 변동성 예측
        lambda_param = 0.94
        ewma_var = returns.ewm(alpha=1-lambda_param).var().iloc[-1]
        return np.sqrt(ewma_var * 252)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float]:
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        
        return float(macd.iloc[-1]), float(signal_line.iloc[-1])
    
    def _update_history(self, asset_name: str, signal: RegimeSignal):
        """체제 히스토리 업데이트"""
        if asset_name not in self.regime_history:
            self.regime_history[asset_name] = []
            
        history = self.regime_history[asset_name]
        
        # 현재 시간과 함께 저장
        history.append({
            'timestamp': datetime.now(),
            'regime': signal.regime,
            'confidence': signal.confidence,
            'strength': signal.strength
        })
        
        # 최대 100개 항목만 유지
        if len(history) > 100:
            history.pop(0)
    
    def get_regime_history(self, asset_name: str) -> List[Dict]:
        """특정 자산의 체제 히스토리 반환"""
        return self.regime_history.get(asset_name, [])
    
    def get_cross_asset_correlation(self, regimes: Dict[str, RegimeSignal]) -> float:
        """자산 간 체제 상관관계"""
        if len(regimes) < 2:
            return 0.0
            
        regime_values = []
        for signal in regimes.values():
            # 체제를 숫자로 변환 (Bull=1, Sideways=0, Bear=-1)
            if signal.regime == MarketRegime.BULL:
                regime_values.append(1.0)
            elif signal.regime == MarketRegime.BEAR:
                regime_values.append(-1.0)
            else:
                regime_values.append(0.0)
        
        # 분산이 0이면 완전 상관
        if np.var(regime_values) == 0:
            return 1.0
            
        return float(np.corrcoef(regime_values, regime_values)[0, 1])