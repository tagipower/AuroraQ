# 🏗️ QuantumAI 시스템 아키텍처 가이드 v2.0

## 📋 시스템 개요

QuantumAI는 **AuroraQ**(단기) + **MacroQ**(중장기) 두 AI Agent가 **SharedCore**를 통해 데이터를 공유하는 독립적이면서 협조적인 자산 운용 시스템입니다.

## 🎯 핵심 설계 원칙

### 1. **독립성 보장**
- 각 Agent는 완전히 독립적인 의사결정
- SharedCore는 데이터만 제공, 직접 제어하지 않음
- Agent 간 직접적인 상호작용 없음

### 2. **읽기 전용 데이터 공유**
- 모든 공통 데이터는 SharedCore에서 일원화
- 감정 점수, 거시 이벤트, 시장 데이터 통합 관리
- 캐싱을 통한 효율적 데이터 접근

### 3. **리소스 효율성**
- VPS 환경(4 CPU, 8GB RAM)에서 실행 가능
- 학습/운영 환경 분리로 비용 최적화
- 배치 처리 및 캐싱으로 성능 향상

## 🔧 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      QuantumAI System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 SharedCore                          │   │
│  │  (공통 데이터 레이어 - 읽기 전용)                   │   │
│  │                                                     │   │
│  │  📊 UnifiedDataProvider                            │   │
│  │    • Multi-asset market data                       │   │
│  │    • Redis caching (5min TTL)                      │   │
│  │    • Crypto, Stock, ETF, Bond                      │   │
│  │                                                     │   │
│  │  📰 SentimentAggregator                           │   │
│  │    • FinBERT batch processing                      │   │
│  │    • News + Social + Forum sentiment              │   │
│  │    • Time-decay weighted scores                    │   │
│  │                                                     │   │
│  │  📅 EventCalendar                                 │   │
│  │    • FOMC, CPI, G20 schedules                     │   │
│  │    • Economic indicator releases                   │   │
│  │    • Event impact quantification                   │   │
│  │                                                     │   │
│  │  🛡️ RiskManagement                               │   │
│  │    • Integrated risk monitoring                    │   │
│  │    • VaR/CVaR calculation                         │   │
│  │    • Cross-agent exposure tracking                │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                 ┌────────────┴────────────┐                │
│                 │                         │                │
│  ┌──────────────▼──────────────┐ ┌────────▼────────────┐   │
│  │        AuroraQ              │ │      MacroQ         │   │
│  │    (단기 트레이딩)          │ │   (중장기 포트폴리오) │   │
│  │                             │ │                     │   │
│  │  🎯 Target: Crypto         │ │ 🎯 Target: Multi   │   │
│  │  ⏱️ Timeframe: 1m-1h       │ │ ⏱️ Timeframe: 1d-3m │   │
│  │  🤖 Strategy: PPO+Rules    │ │ 🤖 Strategy: TFT    │   │
│  │                             │ │                     │   │
│  │  Components:                │ │ Components:         │   │
│  │  • PPO Agent               │ │ • LightweightTFT    │   │
│  │  • Rule Strategies A~E     │ │ • PortfolioOptimizer│   │
│  │  • RealtimeSystem          │ │ • RegimeDetector    │   │
│  │  • BacktestEngine          │ │ • RiskParity        │   │
│  └─────────────────────────────┘ └─────────────────────┘   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    Execution Layer                         │
│                                                             │
│  🖥️ VPS Runtime (CPU Only)     📊 Training (Local/Cloud)   │
│    • Model inference            • GPU accelerated          │
│    • Real-time trading          • Model compression        │
│    • Risk monitoring            • Hyperparameter tuning   │
│    • Performance tracking       • Backtesting research     │
└─────────────────────────────────────────────────────────────┘
```

## 📦 모듈 상세 구조

### SharedCore (공통 인프라)
```
SharedCore/
├── data_layer/
│   ├── unified_data_provider.py     # 읽기 전용 데이터 인터페이스
│   ├── market_data/
│   │   ├── crypto_collector.py      # Binance API
│   │   ├── stocks_collector.py      # Yahoo Finance/Alpha Vantage
│   │   ├── macro_collector.py       # FRED 경제 지표
│   │   └── cache_manager.py         # Redis 캐싱 관리
│   └── event_calendar/
│       ├── economic_events.py       # FOMC, CPI 등
│       └── event_parser.py          # 이벤트 정량화
│
├── sentiment_engine/
│   ├── batch_processor.py           # FinBERT 배치 처리
│   ├── sentiment_aggregator.py      # 다중 소스 통합
│   └── news_collectors/
│       ├── feedly_collector.py      # 뉴스 수집
│       ├── reddit_collector.py      # 소셜 감정
│       └── twitter_collector.py     # 트위터 감정
│
├── risk_management/                 # 통합 리스크 관리
├── monitoring/                      # 시스템 모니터링
└── utils/                          # 공통 유틸리티
```

### AuroraQ (단기 트레이딩)
```
AuroraQ/
├── agent.py                        # 메인 AI Agent
├── backtest/
│   ├── core/
│   │   ├── backtest_engine.py      # 백테스트 엔진
│   │   ├── portfolio.py            # 포트폴리오 관리
│   │   ├── trade_executor.py       # 거래 실행
│   │   └── market_simulator.py     # 시장 시뮬레이션
│   ├── strategies/                 # 백테스트용 전략
│   └── reports/                    # 성과 분석
│
├── production/
│   ├── core/
│   │   ├── realtime_system.py      # 실시간 시스템
│   │   ├── market_data.py          # 실시간 데이터
│   │   └── position_manager.py     # 포지션 관리
│   ├── strategies/
│   │   ├── ppo_strategy.py         # PPO 강화학습
│   │   ├── rule_strategies.py      # Rule A~E
│   │   └── strategy_adapter.py     # 전략 통합
│   ├── execution/
│   │   └── order_manager.py        # 주문 관리
│   └── sentiment/                  # 감정 분석 연동
│
└── config/
    └── default_config.yaml         # 기본 설정
```

### MacroQ (중장기 포트폴리오)
```
MacroQ/
├── core/
│   ├── tft_engine/
│   │   ├── lightweight_tft.py      # 경량 TFT 모델
│   │   ├── data_formatter.py       # 데이터 전처리
│   │   └── attention_analyzer.py   # 주의력 해석
│   └── regime_detector.py          # 시장 체제 분석
│
├── portfolio/
│   ├── optimizer.py                # 포트폴리오 최적화
│   ├── risk_parity.py             # 리스크 패리티
│   └── rebalancer.py              # 리밸런싱
│
├── training/                       # 오프라인 학습
│   ├── offline_trainer.py         # 주말 재학습
│   └── model_compression.py       # 모델 압축
│
├── inference/                      # 온라인 예측
│   ├── online_predictor.py        # 실시간 예측
│   └── prediction_cache.py        # 예측 캐싱
│
└── config/
    └── assets.yaml                # 지원 자산 목록
```

## 🔄 데이터 플로우

### 1. 데이터 공유 플로우
```
External APIs → SharedCore → Cache → AuroraQ/MacroQ (Read Only)
     ↓              ↓
  Binance        Redis Cache
  Yahoo Finance  (5min TTL)
  FRED API       
  Feedly API     
  Reddit API     
```

### 2. 의사결정 플로우

#### AuroraQ (단기)
```
Market Data → Sentiment Score → PPO/Rule Decision → Trade Execution
     ↓              ↓                 ↓               ↓
  BTC/USDT      News + Social     Buy/Sell/Hold    Binance API
  1h OHLCV      Sentiment         Position Size    (Simulation)
```

#### MacroQ (중장기)  
```
Multi-Asset → TFT Prediction → Portfolio Optimization → Rebalancing
     ↓              ↓                 ↓                    ↓
 SPY,QQQ,TLT   1w/1m/3m Return   Risk Parity Weights   Weekly
 GLD,BTC       Quantile Bands    Turnover Control      Execution
```

## ⚡ 성능 최적화

### 1. 메모리 관리
```python
# 배치 크기 자동 조정
class AdaptiveBatchProcessor:
    def adjust_batch_size(self):
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 70%:
            self.batch_size = max(4, self.batch_size // 2)
        elif memory_usage < 50%:
            self.batch_size = min(32, self.batch_size * 1.2)
```

### 2. 캐싱 전략
```python
# 계층적 캐싱
L1_Cache: 메모리 (1분)    # 실시간 데이터
L2_Cache: Redis (5분)     # 감정/뉴스 데이터  
L3_Cache: Disk (1시간)    # 거시경제 데이터
```

### 3. 비동기 처리
```python
# 병렬 데이터 수집
async def collect_all_data():
    tasks = [
        get_market_data(),
        get_sentiment_data(), 
        get_macro_events()
    ]
    return await asyncio.gather(*tasks)
```

## 🔒 리스크 관리 아키텍처

### 1. 통합 리스크 모니터링
```python
class IntegratedRiskMonitor:
    def check_total_exposure(self):
        aurora_exposure = self.get_aurora_positions()
        macro_exposure = self.get_macro_positions()
        
        total_risk = calculate_portfolio_var(
            aurora_exposure + macro_exposure
        )
        
        if total_risk > RISK_LIMIT:
            self.trigger_risk_reduction()
```

### 2. 실시간 위험 한도 관리
- **개별 Agent 한도**: AuroraQ 20%, MacroQ 80%
- **총 노출 한도**: 전체 자본의 95%
- **일일 손실 한도**: 자본의 2%
- **최대 낙폭 한도**: 15%

## 🚀 배포 아키텍처

### 개발 환경
```yaml
Development:
  Location: Local PC
  Purpose: Model training, research, backtesting
  Resources: GPU available, unlimited compute
  Tools: Jupyter, TensorBoard, MLflow
```

### 운영 환경  
```yaml
Production:
  Location: VPS (Contabo)
  Purpose: Live trading, real-time inference
  Resources: 4 CPU, 8GB RAM, no GPU
  Optimization: Model compression, CPU inference
  Monitoring: Telegram alerts, log aggregation
```

### 배포 파이프라인
```
1. 로컬 개발 → 2. 모델 학습 → 3. 모델 압축 → 4. VPS 배포
   ↓              ↓              ↓              ↓
Research       Weekend        Quantization    Production
Backtesting    Retraining     Pruning         Inference
```

## 📊 모니터링 및 관찰성

### 핵심 메트릭
- **AuroraQ**: Sharpe Ratio, Win Rate, 거래 빈도
- **MacroQ**: 포트폴리오 분산, 리밸런싱 빈도, 예측 정확도
- **SharedCore**: 캐시 적중률, API 응답 시간, 데이터 품질

### 알림 시스템
```python
# 중요도별 알림
CRITICAL: 리스크 한도 초과, 시스템 다운
HIGH: 큰 손실, API 장애
MEDIUM: 성과 업데이트, 리밸런싱
LOW: 일반 로그, 디버그 정보
```

이 아키텍처는 실제 운영 환경에서 안정적이고 효율적으로 동작하도록 설계되었으며, 점진적 확장이 가능한 구조입니다.