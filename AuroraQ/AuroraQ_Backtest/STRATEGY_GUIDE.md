# 🎯 전략 개발 가이드

## 📋 개요

이 가이드는 AuroraQ Backtest 프레임워크에서 거래 전략을 개발하는 방법을 설명합니다. 초보자부터 전문가까지 단계별로 전략을 구현할 수 있도록 구성되어 있습니다.

## 🏗️ 전략 아키텍처

### 기본 구조

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from core.trade_executor import OrderSignal, MarketData

class BaseStrategy(ABC):
    """모든 전략의 기본 클래스"""
    
    def __init__(self, **params):
        self.params = params
        self.name = self.__class__.__name__
        self.history = []
        self.indicators = {}
        
    @abstractmethod
    def generate_signal(self, 
                       market_data: MarketData, 
                       position: Dict, 
                       equity: float) -> OrderSignal:
        """거래 신호 생성 (필수 구현)"""
        pass
    
    def initialize(self, data: pd.DataFrame):
        """전략 초기화 (선택적 구현)"""
        pass
    
    def on_trade_executed(self, trade: Dict):
        """거래 실행 후 호출 (선택적 구현)"""
        pass
    
    def finalize(self):
        """백테스트 종료 시 호출 (선택적 구현)"""
        pass
```

## 🚀 1단계: 단순 전략

### 1.1 Buy and Hold 전략

```python
class BuyAndHoldStrategy(BaseStrategy):
    """매수 후 보유 전략"""
    
    def __init__(self, position_size=0.95, **params):
        super().__init__(**params)
        self.position_size = position_size
        self.has_bought = False
    
    def generate_signal(self, market_data, position, equity):
        # 아직 매수하지 않았고 포지션이 없다면 매수
        if not self.has_bought and position['is_flat']:
            self.has_bought = True
            return OrderSignal(
                action='buy',
                size=self.position_size,
                confidence=1.0,
                reason="Buy and Hold: Initial purchase"
            )
        
        # 이후에는 보유
        return OrderSignal('hold', 0)

# 사용 예제
strategy = BuyAndHoldStrategy(position_size=0.8)
```

### 1.2 단순 이동평균 전략

```python
class SimpleMovingAverageStrategy(BaseStrategy):
    """단순 이동평균 전략"""
    
    def __init__(self, short_window=20, long_window=50, position_size=0.5, **params):
        super().__init__(**params)
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
    
    def generate_signal(self, market_data, position, equity):
        # 히스토리에 현재 데이터 추가
        self.history.append(market_data.close)
        
        # 충분한 데이터가 없으면 대기
        if len(self.history) < self.long_window:
            return OrderSignal('hold', 0)
        
        # 이동평균 계산
        short_ma = sum(self.history[-self.short_window:]) / self.short_window
        long_ma = sum(self.history[-self.long_window:]) / self.long_window
        
        # 신호 생성
        if short_ma > long_ma and position['is_flat']:
            # 골든 크로스 - 매수
            return OrderSignal(
                action='buy',
                size=self.position_size,
                confidence=0.7,
                reason=f"Golden Cross: SMA{self.short_window}({short_ma:.2f}) > SMA{self.long_window}({long_ma:.2f})"
            )
        elif short_ma < long_ma and position['is_long']:
            # 데드 크로스 - 매도
            return OrderSignal(
                action='sell',
                size=1.0,  # 전체 포지션 매도
                confidence=0.7,
                reason=f"Dead Cross: SMA{self.short_window}({short_ma:.2f}) < SMA{self.long_window}({long_ma:.2f})"
            )
        
        return OrderSignal('hold', 0)

# 사용 예제
strategy = SimpleMovingAverageStrategy(
    short_window=10,
    long_window=30,
    position_size=0.6
)
```

## 📊 2단계: 기술 지표 활용

### 2.1 RSI 전략

```python
import numpy as np

class RSIStrategy(BaseStrategy):
    """RSI 기반 전략"""
    
    def __init__(self, 
                 rsi_period=14, 
                 oversold=30, 
                 overbought=70,
                 position_size=0.4,
                 **params):
        super().__init__(**params)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_size = position_size
        
    def calculate_rsi(self, prices):
        """RSI 계산"""
        if len(prices) < self.rsi_period + 1:
            return 50  # 중립값 반환
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 평균 계산 (Wilder's smoothing)
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, market_data, position, equity):
        self.history.append(market_data.close)
        
        if len(self.history) < self.rsi_period + 1:
            return OrderSignal('hold', 0)
        
        rsi = self.calculate_rsi(self.history)
        
        # RSI 기반 신호
        if rsi < self.oversold and position['is_flat']:
            return OrderSignal(
                action='buy',
                size=self.position_size,
                confidence=min(0.9, (self.oversold - rsi) / 10),  # RSI가 낮을수록 높은 신뢰도
                reason=f"RSI Oversold: {rsi:.1f} < {self.oversold}"
            )
        elif rsi > self.overbought and position['is_long']:
            return OrderSignal(
                action='sell',
                size=1.0,
                confidence=min(0.9, (rsi - self.overbought) / 10),
                reason=f"RSI Overbought: {rsi:.1f} > {self.overbought}"
            )
        
        return OrderSignal('hold', 0)
```

### 2.2 볼린저 밴드 전략

```python
class BollingerBandsStrategy(BaseStrategy):
    """볼린저 밴드 평균회귀 전략"""
    
    def __init__(self, 
                 period=20, 
                 std_dev=2.0,
                 position_size=0.3,
                 **params):
        super().__init__(**params)
        self.period = period
        self.std_dev = std_dev
        self.position_size = position_size
    
    def calculate_bollinger_bands(self, prices):
        """볼린저 밴드 계산"""
        if len(prices) < self.period:
            return None, None, None
        
        recent_prices = prices[-self.period:]
        middle_band = np.mean(recent_prices)  # 중심선 (SMA)
        std = np.std(recent_prices)
        
        upper_band = middle_band + (self.std_dev * std)
        lower_band = middle_band - (self.std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def generate_signal(self, market_data, position, equity):
        self.history.append(market_data.close)
        
        if len(self.history) < self.period:
            return OrderSignal('hold', 0)
        
        upper, middle, lower = self.calculate_bollinger_bands(self.history)
        current_price = market_data.close
        
        # 볼린저 밴드 기반 신호
        if current_price <= lower and position['is_flat']:
            # 하단 밴드 터치 - 매수 (평균회귀 기대)
            distance_ratio = (lower - current_price) / (upper - lower)
            confidence = min(0.8, 0.5 + distance_ratio)
            
            return OrderSignal(
                action='buy',
                size=self.position_size,
                confidence=confidence,
                reason=f"BB Lower Band Touch: {current_price:.2f} <= {lower:.2f}"
            )
        elif current_price >= upper and position['is_long']:
            # 상단 밴드 터치 - 매도
            distance_ratio = (current_price - upper) / (upper - lower)
            confidence = min(0.8, 0.5 + distance_ratio)
            
            return OrderSignal(
                action='sell',
                size=0.5,  # 부분 매도
                confidence=confidence,
                reason=f"BB Upper Band Touch: {current_price:.2f} >= {upper:.2f}"
            )
        elif position['is_long'] and current_price <= middle:
            # 중심선까지 하락 시 전체 매도
            return OrderSignal(
                action='sell',
                size=1.0,
                confidence=0.6,
                reason=f"BB Middle Band Cross Down: {current_price:.2f} <= {middle:.2f}"
            )
        
        return OrderSignal('hold', 0)
```

## 🔄 3단계: 다중 지표 조합

### 3.1 지표 조합 전략

```python
class MultiIndicatorStrategy(BaseStrategy):
    """여러 지표를 조합한 전략"""
    
    def __init__(self, 
                 ma_short=10, 
                 ma_long=30,
                 rsi_period=14,
                 rsi_oversold=30,
                 rsi_overbought=70,
                 position_size=0.4,
                 **params):
        super().__init__(**params)
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.position_size = position_size
    
    def calculate_indicators(self):
        """모든 지표 계산"""
        if len(self.history) < max(self.ma_long, self.rsi_period + 1):
            return None
        
        # 이동평균
        ma_short = np.mean(self.history[-self.ma_short:])
        ma_long = np.mean(self.history[-self.ma_long:])
        
        # RSI
        deltas = np.diff(self.history)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return {
            'ma_short': ma_short,
            'ma_long': ma_long,
            'rsi': rsi,
            'ma_signal': 1 if ma_short > ma_long else -1,
            'rsi_signal': 1 if rsi < self.rsi_oversold else (-1 if rsi > self.rsi_overbought else 0)
        }
    
    def generate_signal(self, market_data, position, equity):
        self.history.append(market_data.close)
        
        indicators = self.calculate_indicators()
        if indicators is None:
            return OrderSignal('hold', 0)
        
        # 다중 지표 신호 조합
        total_signal = indicators['ma_signal'] + indicators['rsi_signal']
        
        # 강한 매수 신호 (두 지표 모두 매수)
        if total_signal >= 2 and position['is_flat']:
            confidence = 0.8 if total_signal == 2 else 0.6
            return OrderSignal(
                action='buy',
                size=self.position_size,
                confidence=confidence,
                reason=f"Multi-Indicator Buy: MA({indicators['ma_signal']}) + RSI({indicators['rsi_signal']})"
            )
        
        # 강한 매도 신호 (두 지표 모두 매도)
        elif total_signal <= -2 and position['is_long']:
            confidence = 0.8 if total_signal == -2 else 0.6
            return OrderSignal(
                action='sell',
                size=1.0,
                confidence=confidence,
                reason=f"Multi-Indicator Sell: MA({indicators['ma_signal']}) + RSI({indicators['rsi_signal']})"
            )
        
        # 약한 매도 신호 (부분 매도)
        elif total_signal == -1 and position['is_long']:
            return OrderSignal(
                action='sell',
                size=0.3,
                confidence=0.4,
                reason="Weak Sell Signal: Partial Exit"
            )
        
        return OrderSignal('hold', 0)
```

## 🤖 4단계: 고급 전략

### 4.1 평균회귀 전략

```python
class MeanReversionStrategy(BaseStrategy):
    """평균회귀 전략"""
    
    def __init__(self, 
                 lookback_period=30,
                 entry_threshold=2.0,  # 표준편차 배수
                 exit_threshold=0.5,
                 max_position_size=0.6,
                 **params):
        super().__init__(**params)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_position_size = max_position_size
    
    def calculate_z_score(self, prices):
        """Z-Score 계산"""
        if len(prices) < self.lookback_period:
            return 0
        
        recent_prices = prices[-self.lookback_period:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        if std_price == 0:
            return 0
        
        current_price = prices[-1]
        z_score = (current_price - mean_price) / std_price
        
        return z_score
    
    def generate_signal(self, market_data, position, equity):
        self.history.append(market_data.close)
        
        if len(self.history) < self.lookback_period + 1:
            return OrderSignal('hold', 0)
        
        z_score = self.calculate_z_score(self.history)
        
        # 평균회귀 신호
        if z_score < -self.entry_threshold and position['is_flat']:
            # 가격이 평균보다 크게 낮음 - 매수 (상승 기대)
            confidence = min(0.9, abs(z_score) / 3)
            position_size = min(self.max_position_size, abs(z_score) / 4)
            
            return OrderSignal(
                action='buy',
                size=position_size,
                confidence=confidence,
                reason=f"Mean Reversion Buy: Z-Score = {z_score:.2f}"
            )
        elif z_score > self.entry_threshold and position['is_flat']:
            # 가격이 평균보다 크게 높음 - 공매도 (하락 기대)
            # 공매도가 허용된 경우에만
            confidence = min(0.9, abs(z_score) / 3)
            position_size = min(self.max_position_size, abs(z_score) / 4)
            
            return OrderSignal(
                action='sell',  # 공매도
                size=position_size,
                confidence=confidence,
                reason=f"Mean Reversion Short: Z-Score = {z_score:.2f}"
            )
        elif position['is_long'] and z_score > -self.exit_threshold:
            # 롱 포지션 청산
            return OrderSignal(
                action='sell',
                size=1.0,
                confidence=0.6,
                reason=f"Mean Reversion Exit Long: Z-Score = {z_score:.2f}"
            )
        elif position['is_short'] and z_score < self.exit_threshold:
            # 숏 포지션 청산
            return OrderSignal(
                action='buy',
                size=1.0,
                confidence=0.6,
                reason=f"Mean Reversion Exit Short: Z-Score = {z_score:.2f}"
            )
        
        return OrderSignal('hold', 0)
```

### 4.2 모멘텀 전략

```python
class MomentumStrategy(BaseStrategy):
    """모멘텀 추종 전략"""
    
    def __init__(self, 
                 momentum_period=20,
                 ranking_period=60,
                 top_percentile=0.8,  # 상위 20%
                 position_size=0.5,
                 **params):
        super().__init__(**params)
        self.momentum_period = momentum_period
        self.ranking_period = ranking_period
        self.top_percentile = top_percentile
        self.position_size = position_size
    
    def calculate_momentum_score(self, prices):
        """모멘텀 점수 계산"""
        if len(prices) < self.momentum_period + 1:
            return 0
        
        # 단기 모멘텀 (가격 변화율)
        short_momentum = (prices[-1] / prices[-self.momentum_period] - 1) * 100
        
        # 추세 강도 (회귀선의 기울기)
        x = np.arange(self.momentum_period)
        y = prices[-self.momentum_period:]
        slope = np.polyfit(x, y, 1)[0]
        trend_strength = slope / np.mean(y) * 100
        
        # 변동성 조정 모멘텀
        volatility = np.std(np.diff(prices[-self.momentum_period:])) / np.mean(prices[-self.momentum_period:])
        risk_adjusted_momentum = short_momentum / max(volatility, 0.01)
        
        return {
            'momentum': short_momentum,
            'trend_strength': trend_strength,
            'risk_adjusted': risk_adjusted_momentum,
            'composite': (short_momentum + trend_strength + risk_adjusted_momentum) / 3
        }
    
    def generate_signal(self, market_data, position, equity):
        self.history.append(market_data.close)
        
        if len(self.history) < max(self.momentum_period, self.ranking_period) + 1:
            return OrderSignal('hold', 0)
        
        momentum_metrics = self.calculate_momentum_score(self.history)
        
        # 과거 데이터로 모멘텀 순위 계산
        historical_momentums = []
        for i in range(self.momentum_period, min(len(self.history), self.ranking_period)):
            hist_prices = self.history[:i+1]
            hist_momentum = self.calculate_momentum_score(hist_prices)
            historical_momentums.append(hist_momentum['composite'])
        
        if len(historical_momentums) < 10:
            return OrderSignal('hold', 0)
        
        # 현재 모멘텀의 순위 계산
        current_momentum = momentum_metrics['composite']
        percentile_rank = (np.sum(np.array(historical_momentums) < current_momentum) 
                          / len(historical_momentums))
        
        # 모멘텀 기반 신호
        if percentile_rank > self.top_percentile and position['is_flat']:
            # 상위 모멘텀 - 매수
            confidence = min(0.9, percentile_rank)
            return OrderSignal(
                action='buy',
                size=self.position_size,
                confidence=confidence,
                reason=f"High Momentum: {current_momentum:.2f} (Rank: {percentile_rank:.2%})"
            )
        elif percentile_rank < (1 - self.top_percentile) and position['is_long']:
            # 하위 모멘텀 - 매도
            confidence = min(0.9, 1 - percentile_rank)
            return OrderSignal(
                action='sell',
                size=1.0,
                confidence=confidence,
                reason=f"Low Momentum: {current_momentum:.2f} (Rank: {percentile_rank:.2%})"
            )
        elif position['is_long'] and percentile_rank < 0.3:
            # 모멘텀 약화 시 부분 매도
            return OrderSignal(
                action='sell',
                size=0.4,
                confidence=0.5,
                reason=f"Weakening Momentum: Partial Exit"
            )
        
        return OrderSignal('hold', 0)
```

## 🔧 5단계: 리스크 관리 통합

### 5.1 스톱로스 & 테이크프로핏

```python
class RiskManagedStrategy(BaseStrategy):
    """리스크 관리가 통합된 전략"""
    
    def __init__(self, 
                 base_strategy,  # 기본 전략
                 stop_loss_pct=0.05,  # 5% 손절
                 take_profit_pct=0.15,  # 15% 익절
                 trailing_stop_pct=0.03,  # 3% 추적 손절
                 max_drawdown_pct=0.10,  # 10% 최대 낙폭
                 **params):
        super().__init__(**params)
        self.base_strategy = base_strategy
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        # 리스크 관리 상태
        self.entry_price = None
        self.highest_price = None
        self.peak_equity = None
    
    def check_risk_management(self, market_data, position, equity):
        """리스크 관리 규칙 확인"""
        current_price = market_data.close
        
        # 포지션이 있는 경우에만 체크
        if not position['is_flat']:
            # 스톱로스 체크
            if position['is_long']:
                if self.entry_price and (current_price <= self.entry_price * (1 - self.stop_loss_pct)):
                    return OrderSignal(
                        action='sell',
                        size=1.0,
                        confidence=1.0,
                        reason=f"Stop Loss: {current_price:.2f} <= {self.entry_price * (1 - self.stop_loss_pct):.2f}"
                    )
                
                # 테이크 프로핏 체크
                if self.entry_price and (current_price >= self.entry_price * (1 + self.take_profit_pct)):
                    return OrderSignal(
                        action='sell',
                        size=1.0,
                        confidence=0.8,
                        reason=f"Take Profit: {current_price:.2f} >= {self.entry_price * (1 + self.take_profit_pct):.2f}"
                    )
                
                # 추적 손절 체크
                if self.highest_price:
                    trailing_stop_price = self.highest_price * (1 - self.trailing_stop_pct)
                    if current_price <= trailing_stop_price:
                        return OrderSignal(
                            action='sell',
                            size=1.0,
                            confidence=0.9,
                            reason=f"Trailing Stop: {current_price:.2f} <= {trailing_stop_price:.2f}"
                        )
        
        # 최대 낙폭 체크
        if self.peak_equity and equity < self.peak_equity * (1 - self.max_drawdown_pct):
            return OrderSignal(
                action='sell',
                size=1.0,
                confidence=1.0,
                reason=f"Max Drawdown: {equity:.2f} < {self.peak_equity * (1 - self.max_drawdown_pct):.2f}"
            )
        
        return None
    
    def generate_signal(self, market_data, position, equity):
        # 상태 업데이트
        if position['is_long']:
            if self.highest_price is None or market_data.close > self.highest_price:
                self.highest_price = market_data.close
        else:
            self.highest_price = None
        
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        
        # 리스크 관리 우선 체크
        risk_signal = self.check_risk_management(market_data, position, equity)
        if risk_signal:
            if risk_signal.action == 'sell':
                self.entry_price = None
                self.highest_price = None
            return risk_signal
        
        # 기본 전략 신호
        base_signal = self.base_strategy.generate_signal(market_data, position, equity)
        
        # 매수 신호인 경우 진입가 기록
        if base_signal.action == 'buy' and position['is_flat']:
            self.entry_price = market_data.close
            self.highest_price = market_data.close
        
        return base_signal
```

### 5.2 포지션 사이징

```python
class PositionSizingStrategy(BaseStrategy):
    """동적 포지션 사이징 전략"""
    
    def __init__(self, 
                 base_strategy,
                 sizing_method='kelly',  # 'fixed', 'volatility', 'kelly'
                 base_size=0.1,  # 기본 포지션 크기
                 max_size=0.5,   # 최대 포지션 크기
                 lookback_period=30,
                 **params):
        super().__init__(**params)
        self.base_strategy = base_strategy
        self.sizing_method = sizing_method
        self.base_size = base_size
        self.max_size = max_size
        self.lookback_period = lookback_period
        self.trade_history = []
    
    def calculate_kelly_fraction(self):
        """켈리 공식으로 최적 포지션 크기 계산"""
        if len(self.trade_history) < 10:
            return self.base_size
        
        returns = [trade['return'] for trade in self.trade_history[-20:]]  # 최근 20 거래
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        if not wins or not losses:
            return self.base_size
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # 켈리 공식: f = (bp - q) / b
        # b = 승리 시 수익률, p = 승률, q = 패율
        if avg_loss == 0:
            return self.base_size
        
        kelly_fraction = (avg_win * win_rate - (1 - win_rate)) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.max_size))
        
        return kelly_fraction
    
    def calculate_volatility_sizing(self):
        """변동성 기반 포지션 사이징"""
        if len(self.history) < self.lookback_period:
            return self.base_size
        
        recent_prices = self.history[-self.lookback_period:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        
        # 변동성이 낮을수록 큰 포지션
        target_volatility = 0.02  # 2% 일일 변동성
        size_multiplier = target_volatility / max(volatility, 0.005)
        
        position_size = self.base_size * size_multiplier
        return min(position_size, self.max_size)
    
    def generate_signal(self, market_data, position, equity):
        self.history.append(market_data.close)
        
        # 기본 전략 신호
        base_signal = self.base_strategy.generate_signal(market_data, position, equity)
        
        if base_signal.action in ['buy', 'sell'] and base_signal.size > 0:
            # 포지션 사이징 적용
            if self.sizing_method == 'kelly':
                optimal_size = self.calculate_kelly_fraction()
            elif self.sizing_method == 'volatility':
                optimal_size = self.calculate_volatility_sizing()
            else:  # fixed
                optimal_size = self.base_size
            
            # 신호 강도 반영
            confidence_adjusted_size = optimal_size * base_signal.confidence
            
            # 새로운 신호 생성
            return OrderSignal(
                action=base_signal.action,
                size=min(confidence_adjusted_size, self.max_size),
                confidence=base_signal.confidence,
                reason=f"{base_signal.reason} | Size: {confidence_adjusted_size:.3f}"
            )
        
        return base_signal
    
    def on_trade_executed(self, trade):
        """거래 실행 후 기록"""
        # 거래 수익률 계산 및 기록
        if trade['side'] == 'sell' and hasattr(self, 'last_buy_price'):
            trade_return = (trade['price'] - self.last_buy_price) / self.last_buy_price
            self.trade_history.append({
                'return': trade_return,
                'timestamp': trade['timestamp']
            })
        elif trade['side'] == 'buy':
            self.last_buy_price = trade['price']
```

## 📈 6단계: 전략 조합 및 앙상블

### 6.1 다중 전략 앙상블

```python
class EnsembleStrategy(BaseStrategy):
    """여러 전략을 조합한 앙상블 전략"""
    
    def __init__(self, 
                 strategies,  # 전략 리스트
                 weights=None,  # 가중치
                 voting_method='weighted',  # 'weighted', 'majority', 'consensus'
                 confidence_threshold=0.6,
                 **params):
        super().__init__(**params)
        self.strategies = strategies
        self.weights = weights or [1/len(strategies)] * len(strategies)
        self.voting_method = voting_method
        self.confidence_threshold = confidence_threshold
        
        # 가중치 정규화
        total_weight = sum(self.weights)
        self.weights = [w/total_weight for w in self.weights]
    
    def generate_signal(self, market_data, position, equity):
        # 모든 전략에서 신호 수집
        signals = []
        for strategy in self.strategies:
            signal = strategy.generate_signal(market_data, position, equity)
            signals.append(signal)
        
        # 앙상블 방법에 따른 신호 조합
        if self.voting_method == 'weighted':
            return self._weighted_voting(signals)
        elif self.voting_method == 'majority':
            return self._majority_voting(signals)
        else:  # consensus
            return self._consensus_voting(signals)
    
    def _weighted_voting(self, signals):
        """가중 투표"""
        buy_score = 0
        sell_score = 0
        total_confidence = 0
        reasons = []
        
        for signal, weight in zip(signals, self.weights):
            if signal.action == 'buy':
                buy_score += weight * signal.confidence
                reasons.append(f"Buy({signal.confidence:.2f})")
            elif signal.action == 'sell':
                sell_score += weight * signal.confidence
                reasons.append(f"Sell({signal.confidence:.2f})")
            else:
                reasons.append("Hold")
        
        # 최종 신호 결정
        if buy_score > sell_score and buy_score > self.confidence_threshold:
            # 포지션 크기는 평균 계산
            avg_size = np.mean([s.size for s in signals if s.action == 'buy' and s.size > 0])
            return OrderSignal(
                action='buy',
                size=avg_size * buy_score,  # 신뢰도로 조정
                confidence=buy_score,
                reason=f"Ensemble Buy: {' | '.join(reasons)}"
            )
        elif sell_score > buy_score and sell_score > self.confidence_threshold:
            avg_size = np.mean([s.size for s in signals if s.action == 'sell' and s.size > 0])
            return OrderSignal(
                action='sell',
                size=avg_size,
                confidence=sell_score,
                reason=f"Ensemble Sell: {' | '.join(reasons)}"
            )
        
        return OrderSignal('hold', 0)
    
    def _majority_voting(self, signals):
        """다수결 투표"""
        buy_votes = sum(1 for s in signals if s.action == 'buy')
        sell_votes = sum(1 for s in signals if s.action == 'sell')
        
        if buy_votes > len(signals) // 2:
            avg_confidence = np.mean([s.confidence for s in signals if s.action == 'buy'])
            avg_size = np.mean([s.size for s in signals if s.action == 'buy'])
            return OrderSignal(
                action='buy',
                size=avg_size,
                confidence=avg_confidence,
                reason=f"Majority Buy: {buy_votes}/{len(signals)} votes"
            )
        elif sell_votes > len(signals) // 2:
            avg_confidence = np.mean([s.confidence for s in signals if s.action == 'sell'])
            avg_size = np.mean([s.size for s in signals if s.action == 'sell'])
            return OrderSignal(
                action='sell',
                size=avg_size,
                confidence=avg_confidence,
                reason=f"Majority Sell: {sell_votes}/{len(signals)} votes"
            )
        
        return OrderSignal('hold', 0)
    
    def _consensus_voting(self, signals):
        """합의 투표 (모든 전략이 동의해야 실행)"""
        actions = [s.action for s in signals]
        
        if all(action == 'buy' for action in actions):
            avg_confidence = np.mean([s.confidence for s in signals])
            avg_size = np.mean([s.size for s in signals])
            return OrderSignal(
                action='buy',
                size=avg_size,
                confidence=avg_confidence,
                reason="Consensus Buy: All strategies agree"
            )
        elif all(action == 'sell' for action in actions):
            avg_confidence = np.mean([s.confidence for s in signals])
            avg_size = np.mean([s.size for s in signals])
            return OrderSignal(
                action='sell',
                size=avg_size,
                confidence=avg_confidence,
                reason="Consensus Sell: All strategies agree"
            )
        
        return OrderSignal('hold', 0)

# 사용 예제
strategies = [
    SimpleMovingAverageStrategy(short_window=10, long_window=30),
    RSIStrategy(rsi_period=14),
    BollingerBandsStrategy(period=20)
]

ensemble = EnsembleStrategy(
    strategies=strategies,
    weights=[0.4, 0.3, 0.3],
    voting_method='weighted'
)
```

## 🧪 7단계: 전략 테스트 및 검증

### 7.1 전략 백테스트

```python
# 전략 백테스트 예제
from core import BacktestEngine, BacktestConfig
from utils import DataManager

# 데이터 로딩
data_manager = DataManager()
data = data_manager.load_data('BTC-USD', start_date='2023-01-01')

# 백테스트 설정
config = BacktestConfig(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

# 전략 생성
strategy = MultiIndicatorStrategy(
    ma_short=10,
    ma_long=30,
    rsi_period=14
)

# 백테스트 실행
engine = BacktestEngine(config)
result = engine.run(strategy, data)

# 결과 분석
print(f"총 수익률: {result.total_return:.2%}")
print(f"샤프 비율: {result.sharpe_ratio:.2f}")
print(f"최대 낙폭: {result.max_drawdown:.2%}")
print(f"승률: {result.win_rate:.1%}")
```

### 7.2 파라미터 최적화

```python
# 파라미터 최적화 예제
param_grid = {
    'ma_short': [5, 10, 15, 20],
    'ma_long': [20, 30, 40, 50],
    'rsi_period': [10, 14, 21],
    'position_size': [0.2, 0.4, 0.6]
}

best_params, best_result = engine.optimize(
    MultiIndicatorStrategy,
    param_grid,
    data,
    metric='sharpe_ratio'
)

print(f"최적 파라미터: {best_params}")
print(f"최적 샤프 비율: {best_result.sharpe_ratio:.2f}")
```

### 7.3 Walk-Forward 검증

```python
# Walk-Forward 분석
results = engine.walk_forward_analysis(
    strategy=MultiIndicatorStrategy(**best_params),
    data=data,
    train_period=252,  # 1년 훈련
    test_period=63,    # 3개월 테스트
    step=21           # 1개월씩 이동
)

# 안정성 평가
returns = [r.total_return for r in results]
print(f"평균 수익률: {np.mean(returns):.2%}")
print(f"수익률 표준편차: {np.std(returns):.2%}")
print(f"최대 수익률: {np.max(returns):.2%}")
print(f"최소 수익률: {np.min(returns):.2%}")
```

## 💡 전략 개발 팁

### 1. 과최적화 방지
- 샘플 외 테스트 필수
- 파라미터 수 최소화
- 경제적 논리 기반 전략 개발

### 2. 리스크 관리
- 항상 스톱로스 설정
- 포지션 사이징 고려
- 최대 낙폭 제한

### 3. 성과 평가
- 벤치마크와 비교
- 리스크 조정 수익률 중시
- 다양한 시장 환경에서 테스트

### 4. 실전 고려사항
- 거래 비용 현실적 설정
- 슬리피지 고려
- 유동성 제약 인식

이 가이드를 통해 단계별로 전략을 개발하고 검증해보세요!