# 📚 AuroraQ Backtest 사용자 매뉴얼

## 🎯 시작하기

### 기본 사용법

```python
from core import BacktestEngine, BacktestConfig
from utils import DataManager
from strategies import YourStrategy

# 1. 데이터 로딩
data_manager = DataManager()
data = data_manager.load_data('BTC-USD', source='yahoo', start_date='2023-01-01')

# 2. 백테스트 설정
config = BacktestConfig(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

# 3. 백테스트 엔진 생성
engine = BacktestEngine(config)

# 4. 전략 생성
strategy = YourStrategy(param1=value1, param2=value2)

# 5. 백테스트 실행
result = engine.run(strategy, data)

# 6. 결과 확인
print(result.summary())
```

## 📊 데이터 관리

### 지원하는 데이터 소스

#### 1. Yahoo Finance
```python
data = data_manager.load_data(
    symbol='BTC-USD',
    source='yahoo',
    start_date='2023-01-01',
    end_date='2023-12-31',
    interval='1d'
)
```

#### 2. Binance API
```python
data = data_manager.load_data(
    symbol='BTCUSDT',
    source='binance',
    start_date='2023-01-01',
    interval='1h'
)
```

#### 3. CSV 파일
```python
# CSV 파일은 다음 형식이어야 합니다:
# timestamp,open,high,low,close,volume

data = data_manager.load_data(
    'your_data.csv',
    source='csv'
)
```

### 데이터 전처리

```python
# 데이터 정보 확인
info = data_manager.get_data_info(data)
print(info)

# 데이터 분할 (훈련/검증/테스트)
train_data, val_data, test_data = data_manager.split_data(
    data, 
    train_ratio=0.7, 
    validation_ratio=0.15
)

# 리샘플링 (시간 간격 변경)
hourly_data = data_manager.resample_data(data, '1h')
```

## 🔧 백테스트 설정

### BacktestConfig 옵션

```python
config = BacktestConfig(
    initial_capital=100000.0,     # 초기 자본
    commission=0.001,             # 수수료 (0.1%)
    slippage=0.0005,             # 슬리피지 (0.05%)
    min_order_size=0.0001,       # 최소 주문 크기
    max_position_size=0.95,      # 최대 포지션 크기 (자본의 95%)
    enable_short=False,          # 공매도 허용 여부
    enable_leverage=False,       # 레버리지 허용 여부
    max_leverage=1.0,           # 최대 레버리지
    risk_free_rate=0.02         # 무위험 수익률 (연 2%)
)
```

### 고급 설정 예제

```python
# 레버리지 거래 설정
leverage_config = BacktestConfig(
    initial_capital=50000,
    enable_leverage=True,
    max_leverage=3.0,
    commission=0.0015,  # 레버리지 거래는 수수료가 높음
    slippage=0.001
)

# 공매도 허용 설정
short_config = BacktestConfig(
    enable_short=True,
    max_position_size=0.8,  # 공매도시 더 보수적
    commission=0.0012
)
```

## 📈 전략 개발

### 기본 전략 구조

```python
from abc import ABC, abstractmethod
from core.trade_executor import OrderSignal

class BaseStrategy(ABC):
    def __init__(self, **params):
        self.params = params
        self.history = []
    
    @abstractmethod
    def generate_signal(self, market_data, position, equity) -> OrderSignal:
        """거래 신호 생성"""
        pass
    
    def update_history(self, market_data):
        """히스토리 업데이트"""
        self.history.append(market_data)
        # 메모리 관리를 위해 일정 길이 유지
        if len(self.history) > 1000:
            self.history = self.history[-500:]
```

### 단순 이동평균 전략 예제

```python
class SimpleMAStrategy(BaseStrategy):
    def __init__(self, short_window=20, long_window=50, **params):
        super().__init__(**params)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, market_data, position, equity):
        self.update_history(market_data)
        
        if len(self.history) < self.long_window:
            return OrderSignal('hold', 0)
        
        # 이동평균 계산
        prices = [h.close for h in self.history[-self.long_window:]]
        short_ma = sum(prices[-self.short_window:]) / self.short_window
        long_ma = sum(prices) / self.long_window
        
        current_price = market_data.close
        
        # 신호 생성
        if short_ma > long_ma and position.is_flat:
            # 골든 크로스 - 매수
            return OrderSignal(
                action='buy',
                size=0.5,  # 자본의 50%
                confidence=0.7,
                reason=f"Golden Cross: {short_ma:.2f} > {long_ma:.2f}"
            )
        elif short_ma < long_ma and position.is_long:
            # 데드 크로스 - 매도
            return OrderSignal(
                action='sell',
                size=1.0,  # 전체 포지션
                confidence=0.7,
                reason=f"Dead Cross: {short_ma:.2f} < {long_ma:.2f}"
            )
        
        return OrderSignal('hold', 0)
```

### 기술 지표 활용 전략

```python
import pandas as pd
import numpy as np

class RSIStrategy(BaseStrategy):
    def __init__(self, rsi_period=14, oversold=30, overbought=70, **params):
        super().__init__(**params)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices):
        """RSI 계산"""
        if len(prices) < self.rsi_period + 1:
            return 50  # 기본값
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, market_data, position, equity):
        self.update_history(market_data)
        
        if len(self.history) < self.rsi_period + 1:
            return OrderSignal('hold', 0)
        
        prices = [h.close for h in self.history]
        rsi = self.calculate_rsi(prices)
        
        # RSI 기반 신호
        if rsi < self.oversold and position.is_flat:
            return OrderSignal(
                action='buy',
                size=0.3,
                confidence=0.6,
                reason=f"RSI Oversold: {rsi:.1f}"
            )
        elif rsi > self.overbought and position.is_long:
            return OrderSignal(
                action='sell',
                size=0.5,  # 부분 매도
                confidence=0.6,
                reason=f"RSI Overbought: {rsi:.1f}"
            )
        
        return OrderSignal('hold', 0)
```

## 🔍 백테스트 실행

### 단일 전략 백테스트

```python
# 기본 실행
result = engine.run(strategy, data)

# 진행률 콜백과 함께 실행
def progress_callback(progress):
    print(f"Progress: {progress:.1%}")

result = engine.run(
    strategy, 
    data, 
    progress_callback=progress_callback
)

# 특정 기간만 테스트
result = engine.run(
    strategy, 
    data,
    start_date='2023-06-01',
    end_date='2023-12-31'
)
```

### 다중 전략 백테스트

```python
strategies = [
    SimpleMAStrategy(short_window=10, long_window=30),
    SimpleMAStrategy(short_window=20, long_window=50),
    RSIStrategy(rsi_period=14),
    RSIStrategy(rsi_period=21)
]

# 순차 실행
results = engine.run_multiple(strategies, data, parallel=False)

# 병렬 실행 (더 빠름)
results = engine.run_multiple(strategies, data, parallel=True)

# 결과 비교
for name, result in results.items():
    print(f"{name}: {result.total_return:.2%}")
```

### 파라미터 최적화

```python
# 파라미터 그리드 정의
param_grid = {
    'short_window': [10, 15, 20],
    'long_window': [30, 40, 50],
    'position_size': [0.3, 0.5, 0.7]
}

# 최적화 실행
best_params, best_result = engine.optimize(
    SimpleMAStrategy,
    param_grid,
    data,
    metric='sharpe_ratio'  # 최적화 기준
)

print(f"Best parameters: {best_params}")
print(f"Best Sharpe ratio: {best_result.sharpe_ratio:.2f}")
```

### Walk-Forward 분석

```python
# Walk-forward 분석 실행
results = engine.walk_forward_analysis(
    strategy=SimpleMAStrategy(short_window=20, long_window=50),
    data=data,
    train_period=252,  # 1년 훈련
    test_period=63,    # 3개월 테스트
    step=21           # 1개월씩 이동
)

# 각 기간별 결과 확인
for i, result in enumerate(results):
    print(f"Period {i+1}: {result.total_return:.2%}")
```

## 📊 결과 분석

### 기본 결과 확인

```python
# 요약 출력
print(result.summary())

# 개별 지표 접근
print(f"총 수익률: {result.total_return:.2%}")
print(f"샤프 비율: {result.sharpe_ratio:.2f}")
print(f"최대 낙폭: {result.max_drawdown:.2%}")
print(f"승률: {result.win_rate:.1%}")
```

### 상세 분석

```python
# 자산가치 곡선
equity_curve = result.equity_curve
print(equity_curve.head())

# 일별 수익률
daily_returns = result.returns
print(f"평균 일수익률: {daily_returns.mean():.4f}")
print(f"수익률 표준편차: {daily_returns.std():.4f}")

# 거래 내역
trades = result.trades
print(f"총 거래 수: {len(trades)}")
print(trades.head())

# 포지션 히스토리
positions = result.positions
print(positions.head())
```

### 월별/연도별 분석

```python
# 월별 수익률
monthly_returns = result.monthly_returns
print("월별 수익률:")
print(monthly_returns)

# 연도별 수익률
yearly_returns = result.yearly_returns
print("연도별 수익률:")
print(yearly_returns)

# 최고/최악 월
best_month = monthly_returns.max()
worst_month = monthly_returns.min()
print(f"최고 월: {best_month:.2%}")
print(f"최악 월: {worst_month:.2%}")
```

## 📈 시각화

### 기본 차트

```python
from reports import BacktestVisualizer

visualizer = BacktestVisualizer()

# 자산가치 곡선
fig = visualizer.plot_equity_curve(
    result.equity_curve,
    drawdown=result.drawdown_series
)
fig.show()

# 수익률 분포
fig = visualizer.plot_returns_distribution(result.returns)
fig.show()

# 거래 분석
fig = visualizer.plot_trade_analysis(
    result.trades,
    result.equity_curve
)
fig.show()
```

### 종합 대시보드

```python
# 모든 차트를 포함한 대시보드
metrics = {
    'total_return': result.total_return,
    'sharpe_ratio': result.sharpe_ratio,
    'max_drawdown': result.max_drawdown,
    'monthly_returns': result.monthly_returns,
    'yearly_returns': result.yearly_returns
}

fig = visualizer.create_dashboard(
    result.equity_curve,
    result.returns,
    result.trades,
    metrics
)
fig.show()
```

## 📋 리포트 생성

### 종합 리포트

```python
from reports import ReportGenerator

report_gen = ReportGenerator()

# 종합 리포트 생성
report_path = report_gen.generate_comprehensive_report(
    result,
    strategy_name="Simple MA Strategy",
    include_charts=True
)

print(f"리포트가 생성되었습니다: {report_path}")
```

### 다중 전략 비교

```python
# 여러 전략 결과 비교
comparison_path = report_gen.compare_strategies({
    'MA_10_30': result1,
    'MA_20_50': result2,
    'RSI_14': result3
})

print(f"비교 리포트: {comparison_path}")
```

## ⚠️ 주의사항 및 팁

### 1. 데이터 품질 확인

```python
# 데이터 검증
info = data_manager.get_data_info(data)
print(f"결측값: {info['missing_values']}")
print(f"데이터 기간: {info['duration_days']}일")

# 이상치 확인
price_range = info['price_range']
if price_range['max'] / price_range['min'] > 100:
    print("⚠️ 가격 범위가 매우 넓습니다. 데이터를 확인해주세요.")
```

### 2. 과최적화 방지

```python
# 과최적화 방지를 위한 검증
# 1. 샘플 외 테스트
train_data = data.iloc[:int(len(data) * 0.7)]
test_data = data.iloc[int(len(data) * 0.7):]

# 훈련 데이터로 최적화
best_params, _ = engine.optimize(Strategy, param_grid, train_data)

# 테스트 데이터로 검증
strategy = Strategy(**best_params)
test_result = engine.run(strategy, test_data)

print(f"테스트 성과: {test_result.total_return:.2%}")
```

### 3. 실제 거래와의 차이점

```python
# 현실적인 설정 사용
realistic_config = BacktestConfig(
    initial_capital=10000,     # 실제 투자 가능 금액
    commission=0.0025,         # 실제 거래소 수수료
    slippage=0.001,           # 실제 슬리피지
    min_order_size=0.001,     # 실제 최소 주문
    max_position_size=0.8     # 보수적 포지션 크기
)
```

### 4. 성과 해석

```python
# 벤치마크와 비교
benchmark_return = 0.10  # 연 10% (예시)

if result.annualized_return > benchmark_return:
    print("✅ 벤치마크를 상회하는 성과")
else:
    print("❌ 벤치마크 대비 저조한 성과")

# 리스크 조정 수익률 확인
if result.sharpe_ratio > 1.0:
    print("✅ 양호한 리스크 조정 수익률")
else:
    print("⚠️ 리스크 대비 수익률 개선 필요")
```

### 5. 메모리 관리

```python
# 큰 데이터셋 처리시
import gc

# 사용하지 않는 변수 정리
del large_dataset
gc.collect()

# 데이터 청크 단위 처리
chunk_size = 10000
for i in range(0, len(data), chunk_size):
    chunk = data.iloc[i:i+chunk_size]
    # 처리...
```

## 🚀 고급 기능

### 커스텀 성과 지표

```python
from utils.performance_metrics import PerformanceAnalyzer

# 커스텀 지표 계산
analyzer = PerformanceAnalyzer()

# 사용자 정의 지표
def custom_metric(returns):
    # 연속 손실 일수 계산
    consecutive_losses = 0
    max_consecutive_losses = 0
    
    for ret in returns:
        if ret < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    return max_consecutive_losses

max_loss_streak = custom_metric(result.returns)
print(f"최대 연속 손실 일수: {max_loss_streak}")
```

### 실시간 모니터링

```python
import time

def real_time_callback(equity, trades_count):
    """실시간 모니터링 콜백"""
    current_time = time.strftime("%H:%M:%S")
    print(f"[{current_time}] 자산: ${equity:,.0f}, 거래수: {trades_count}")

# 실시간 모니터링과 함께 백테스트
# (실제 구현은 백테스트 엔진에 콜백 기능 추가 필요)
```

이것으로 AuroraQ Backtest 사용자 매뉴얼을 마칩니다. 추가 질문이나 고급 기능에 대한 문의는 개발팀에 연락해주세요.