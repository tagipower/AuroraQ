# 🏗️ AuroraQ Backtest 아키텍처 가이드

## 📋 시스템 개요

AuroraQ Backtest는 고성능 백테스팅을 위한 모듈화된 프레임워크입니다. 실제 거래 환경을 정확하게 시뮬레이션하며, 확장 가능한 아키텍처를 제공합니다.

## 🎯 설계 원칙

### 1. 모듈화 (Modularity)
- 각 컴포넌트는 독립적으로 개발 및 테스트 가능
- 명확한 인터페이스를 통한 컴포넌트 간 통신
- 플러그인 방식의 전략 및 지표 추가

### 2. 확장성 (Scalability)
- 대용량 데이터 처리 지원
- 병렬 처리 및 분산 컴퓨팅 준비
- 메모리 효율적인 스트리밍 처리

### 3. 정확성 (Accuracy)
- 실제 거래 환경의 정확한 시뮬레이션
- 슬리피지, 수수료, 시장 충격 모델링
- 현실적인 주문 실행 로직

### 4. 성능 (Performance)
- 벡터화 연산을 통한 고속 처리
- 메모리 효율적인 데이터 구조
- 캐싱 및 최적화 기법 활용

## 🔧 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    AuroraQ Backtest                        │
├─────────────────────────────────────────────────────────────┤
│  User Interface Layer                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │   CLI Tool   │ │  Jupyter     │ │   Web API    │      │
│  │              │ │  Notebooks   │ │              │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Backtest     │ │ Strategy     │ │ Report       │      │
│  │ Orchestrator │ │ Manager      │ │ Generator    │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  Core Engine Layer                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Backtest     │ │ Portfolio    │ │ Trade        │      │
│  │ Engine       │ │ Manager      │ │ Executor     │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Market       │ │ Risk         │ │ Performance  │      │
│  │ Simulator    │ │ Manager      │ │ Analyzer     │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Data         │ │ Market Data  │ │ Historical   │      │
│  │ Manager      │ │ Providers    │ │ Database     │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ Logging      │ │ Configuration│ │ Monitoring   │      │
│  │ System       │ │ Management   │ │ & Metrics    │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 📦 컴포넌트 상세

### Core Engine Layer

#### 1. BacktestEngine
**역할**: 백테스트 실행의 중앙 조정자
**주요 기능**:
- 백테스트 라이프사이클 관리
- 시간 진행 및 이벤트 순서 제어
- 컴포넌트 간 데이터 흐름 조정
- 병렬 처리 및 최적화 관리

```python
class BacktestEngine:
    """
    시간 진행: data[t] → market_update
                     ↓
    신호 생성: strategy.generate_signal()
                     ↓
    주문 실행: trade_executor.execute()
                     ↓
    포트폴리오 업데이트: portfolio.update()
                     ↓
    성과 기록: performance_tracker.record()
    """
```

#### 2. Portfolio
**역할**: 자본, 포지션, 수익률 관리
**주요 기능**:
- 실시간 자산가치 계산
- 포지션 및 현금 추적
- 거래 이력 관리
- 리스크 메트릭 계산

**데이터 구조**:
```python
@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position]
    equity_history: List[Tuple[datetime, float]]
    trades: List[Trade]
    unrealized_pnl: float
    realized_pnl: float
```

#### 3. TradeExecutor
**역할**: 주문 처리 및 체결 시뮬레이션
**주요 기능**:
- 다양한 주문 타입 지원 (Market, Limit, Stop)
- 슬리피지 및 수수료 적용
- 부분 체결 시뮬레이션
- 거래 검증 및 리스크 체크

**주문 처리 플로우**:
```
OrderSignal → Validation → Slippage Calculation → 
Execution → Commission Application → Portfolio Update
```

#### 4. MarketSimulator
**역할**: 실제 시장 조건 시뮬레이션
**주요 기능**:
- 시장 마이크로구조 모델링
- 유동성 및 시장 충격 시뮬레이션
- 호가창 및 스프레드 모델링
- 시장 상태 분류 (Bullish/Bearish/Sideways)

### Data Layer

#### 1. DataManager
**역할**: 데이터 소스 통합 관리
**지원 데이터 소스**:
- Yahoo Finance API
- Binance API
- CSV 파일
- 사용자 정의 데이터 소스

**데이터 파이프라인**:
```
Raw Data → Validation → Cleaning → Normalization → Cache → Engine
```

#### 2. MarketDataProvider
**역할**: 실시간/배치 데이터 제공
**기능**:
- 다중 소스 데이터 병합
- 데이터 품질 검증
- 캐싱 및 성능 최적화

### Analysis Layer

#### 1. PerformanceAnalyzer
**역할**: 종합적인 성과 분석
**계산 지표** (50+ 지표):

**수익률 지표**:
- 총 수익률 (Total Return)
- 연환산 수익률 (Annualized Return)
- 복리 성장률 (CAGR)
- 위험 조정 수익률 (Risk-Adjusted Return)

**리스크 지표**:
- 최대 낙폭 (Maximum Drawdown)
- 변동성 (Volatility)
- VaR (Value at Risk) - 95%, 99%
- CVaR (Conditional VaR)
- 베타 (Beta) - 벤치마크 대비

**효율성 지표**:
- 샤프 비율 (Sharpe Ratio)
- 소르티노 비율 (Sortino Ratio)
- 칼마 비율 (Calmar Ratio)
- 정보 비율 (Information Ratio)
- 트레이너 비율 (Treynor Ratio)

**거래 지표**:
- 승률 (Win Rate)
- 손익비 (Profit Factor)
- 평균 승리/손실
- 최대 연속 승리/손실
- 거래 빈도 (Trade Frequency)

#### 2. RiskAnalyzer
**역할**: 리스크 측정 및 관리
**기능**:
- 실시간 리스크 모니터링
- 리스크 한도 관리
- 스트레스 테스트
- 시나리오 분석

### Visualization & Reporting

#### 1. BacktestVisualizer
**역할**: 차트 및 그래프 생성
**차트 타입**:
- 자산가치 곡선 (Equity Curve)
- 낙폭 차트 (Drawdown Chart)
- 수익률 분포 (Returns Distribution)
- 거래 분석 (Trade Analysis)
- 성과 히트맵 (Performance Heatmap)
- 롤링 지표 (Rolling Metrics)

#### 2. ReportGenerator
**역할**: 종합 리포트 생성
**리포트 형식**:
- HTML 대화형 리포트
- PDF 인쇄용 리포트
- JSON 데이터 익스포트
- CSV 원시 데이터

## 🔄 데이터 플로우

### 1. 백테스트 실행 플로우

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  데이터     │    │   전략      │    │   설정      │
│  로딩       │    │   초기화    │    │   검증      │
└─────────────┘    └─────────────┘    └─────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                          │
                ┌─────────────┐
                │ 백테스트    │
                │ 엔진 초기화 │
                └─────────────┘
                          │
                ┌─────────────┐
                │ 시간 루프   │
                │ 시작        │
                └─────────────┘
                          │
        ┌─────────────────────────────────────────┐
        │              시간 t에서               │
        │                                       │
        │  ┌─────────────┐  ┌─────────────┐   │
        │  │ 시장 데이터 │  │   전략      │   │
        │  │   업데이트  │→ │ 신호 생성   │   │
        │  └─────────────┘  └─────────────┘   │
        │           │               │         │
        │           │       ┌─────────────┐   │
        │           │       │ 주문 실행   │   │
        │           │       └─────────────┘   │
        │           │               │         │
        │           │       ┌─────────────┐   │
        │           │       │ 포트폴리오  │   │
        │           │       │   업데이트  │   │
        │           │       └─────────────┘   │
        │           │               │         │
        │           │       ┌─────────────┐   │
        │           │       │ 성과 기록   │   │
        │           │       └─────────────┘   │
        └─────────────────────────────────────────┘
                          │
                ┌─────────────┐
                │ 다음 시점   │
                │ 진행        │
                └─────────────┘
                          │
                ┌─────────────┐
                │ 결과 분석   │
                │ 및 리포트   │
                └─────────────┘
```

### 2. 실시간 데이터 플로우

```
Market Data → MarketSimulator → TradeExecutor → Portfolio → PerformanceTracker
     │              │                │             │              │
     │              │                │             │              │
   OHLCV        Market State     Order Signals   Position      Metrics
  Timestamp      Volatility      Order Types     Cash         Returns
   Volume        Liquidity       Slippage        Equity       Drawdown
                 Impact          Commission      P&L          Ratios
```

## 🚀 성능 최적화

### 1. 메모리 최적화

```python
# 청크 단위 데이터 처리
class StreamingBacktest:
    def __init__(self, chunk_size=10000):
        self.chunk_size = chunk_size
    
    def process_chunk(self, data_chunk):
        # 청크별 처리로 메모리 사용량 제한
        pass

# 메모리 효율적인 데이터 구조 사용
import numpy as np
from collections import deque

class RollingBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def add(self, value):
        self.buffer.append(value)
```

### 2. 연산 최적화

```python
# NumPy 벡터화 연산 활용
def calculate_returns_vectorized(prices):
    """벡터화된 수익률 계산 (루프 대비 10x 빠름)"""
    return np.diff(prices) / prices[:-1]

# Numba JIT 컴파일 적용
from numba import jit

@jit(nopython=True)
def fast_moving_average(prices, window):
    """JIT 컴파일된 이동평균 (100x 빠름)"""
    result = np.empty(len(prices))
    for i in range(window-1, len(prices)):
        result[i] = np.mean(prices[i-window+1:i+1])
    return result
```

### 3. 병렬 처리

```python
from multiprocessing import Pool
import concurrent.futures

class ParallelBacktest:
    def run_multiple_strategies(self, strategies, data):
        """다중 전략 병렬 실행"""
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.run_single, strategy, data): strategy
                for strategy in strategies
            }
            
            results = {}
            for future in concurrent.futures.as_completed(futures):
                strategy = futures[future]
                result = future.result()
                results[strategy.name] = result
            
            return results
```

## 🔒 확장성 설계

### 1. 플러그인 아키텍처

```python
# 전략 플러그인 인터페이스
class StrategyPlugin:
    @abstractmethod
    def generate_signal(self, market_data, portfolio):
        pass
    
    @abstractmethod
    def get_parameters(self):
        pass

# 지표 플러그인 인터페이스
class IndicatorPlugin:
    @abstractmethod
    def calculate(self, data):
        pass
    
    @abstractmethod
    def get_config(self):
        pass
```

### 2. 설정 관리

```python
# 계층적 설정 시스템
class ConfigManager:
    def __init__(self):
        self.config_hierarchy = [
            'user_config.yaml',
            'environment_config.yaml',
            'default_config.yaml'
        ]
    
    def get_config(self, key):
        # 상위 설정부터 순차 검색
        pass
```

### 3. 이벤트 시스템

```python
# 이벤트 기반 아키텍처
class EventBus:
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def publish(self, event):
        for callback in self.subscribers.get(event.type, []):
            callback(event)

# 이벤트 타입
class BacktestEvents:
    TRADE_EXECUTED = "trade_executed"
    POSITION_CHANGED = "position_changed"
    DRAWDOWN_THRESHOLD = "drawdown_threshold"
    PROFIT_TARGET = "profit_target"
```

## 🧪 테스트 전략

### 1. 단위 테스트
- 각 컴포넌트별 독립 테스트
- Mock 데이터를 사용한 격리 테스트
- 엣지 케이스 및 오류 시나리오 테스트

### 2. 통합 테스트
- 컴포넌트 간 상호작용 테스트
- 실제 데이터를 사용한 종단간 테스트
- 성능 벤치마크 테스트

### 3. 검증 테스트
- 알려진 전략의 결과 재현
- 금융 이론과의 일치성 검증
- 실제 거래 결과와의 비교

## 📊 모니터링 및 로깅

### 1. 구조화된 로깅
```python
import structlog

logger = structlog.get_logger()

# 성과 로깅
logger.info("backtest_completed",
           strategy="MA_Strategy",
           total_return=0.15,
           sharpe_ratio=1.2,
           max_drawdown=0.08)

# 거래 로깅
logger.info("trade_executed",
           side="buy",
           size=100,
           price=50000,
           commission=50)
```

### 2. 메트릭 수집
```python
from prometheus_client import Counter, Histogram, Gauge

# 성과 메트릭
backtest_duration = Histogram('backtest_duration_seconds')
total_trades = Counter('total_trades')
current_equity = Gauge('current_equity')
```

## 🔮 향후 확장 계획

### 1. 분산 처리
- Apache Spark 통합
- Kubernetes 클러스터 지원
- 클라우드 네이티브 배포

### 2. 실시간 처리
- Apache Kafka 스트리밍
- 실시간 신호 생성
- 라이브 트레이딩 연결

### 3. 머신러닝 통합
- 자동 특성 추출
- 하이퍼파라미터 자동 튜닝
- 강화학습 에이전트 지원

### 4. 고급 분석
- 몬테카를로 시뮬레이션
- 시나리오 분석
- 스트레스 테스트

이 아키텍처 가이드는 AuroraQ Backtest의 설계 철학과 구현 세부사항을 제공합니다. 각 컴포넌트는 모듈화되어 있어 독립적으로 개선하고 확장할 수 있습니다.