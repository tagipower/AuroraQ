# AuroraQ 백테스트 시스템 v2

적응형·확률적·다중프레임 백테스트 환경

## 📋 개요

AuroraQ 백테스트 시스템 v2는 5분봉 중저빈도 전략의 샘플 부족과 노이즈 문제를 해결하기 위해 설계된 고도화된 백테스트 환경입니다.

### 🎯 핵심 목표

- **실전 괴리 최소화**: 슬리피지, 수수료, 시장 영향을 현실적으로 모델링
- **적응형 신호 처리**: 시장 레짐과 감정에 따른 동적 임계값 조정
- **확률적 포지션 관리**: 신뢰도 기반 포지션 크기 및 Kelly 기준 적용
- **다중 타임프레임**: 5분봉, 15분봉, 1시간봉 통합 분석
- **학습 피드백**: PPO와 MAB에 백테스트 결과 자동 반영

## 🏗️ 아키텍처

### 5계층 구조

```
┌─────────────────────────────────────────────────────────┐
│                  Controller Layer                       │
│           (BacktestOrchestrator)                       │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                Evaluation Layer                         │
│     (StandardizedMetrics + Sample Weighting)           │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│              Execution Layer                            │
│   (Kelly Criterion + Slippage + Market Impact)         │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│               Signal Layer                              │
│  (Adaptive Entry + Probabilistic + Time Filters)       │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                Data Layer                               │
│     (Multi-Timeframe + Sentiment + Indicators)         │
└─────────────────────────────────────────────────────────┘
```

## 📦 주요 컴포넌트

### 1. 데이터 계층 (Data Layer)
- **MultiTimeframeData**: 5분봉 → 15분봉, 1시간봉 자동 변환
- **IndicatorCache**: 중복 계산 방지 (30-50% 성능 향상)
- **SentimentAlignment**: 가격 데이터와 감정 점수 시간 정렬

### 2. 시그널 처리 계층 (Signal Processing Layer)
- **AdaptiveEntrySystem**: 변동성/레짐/감정 기반 동적 임계값
- **ProbabilisticEntry**: 신뢰도 기반 포지션 크기 결정
- **TimeBasedFilter**: 세션별(아시아/유럽/미국) 노이즈 필터링
- **MultiTimeframeConfirmation**: 상위 타임프레임 신호 가중치

### 3. 실행 시뮬레이션 계층 (Execution Simulation Layer)
- **RiskAdjustedEntry**: Kelly 기준 기반 리스크 조정
- **SlippageModel**: 변동성과 거래량 기반 동적 슬리피지
- **FeeModel**: Binance 수수료 구조 모델링
- **MarketImpactSimulator**: 거래량 대비 시장 영향 계산

### 4. 평가 계층 (Evaluation Layer)
- **StandardizedMetrics**: ROI, Sharpe, Sortino, Profit Factor 등
- **SampleWeightCalculator**: 표본 수 기반 전략 점수 가중치
- **ReportGenerator**: CSV/JSON 형태 상세 보고서

### 5. 컨트롤러 계층 (Controller Layer)
- **BacktestController**: 단일 백테스트 관리
- **BacktestOrchestrator**: 복수 백테스트 병렬 실행
- **WalkForwardAnalysis**: 시간 분할 검증

## 🔄 PPO/MAB 피드백 루프

### BacktestFeedbackBridge
- **PPOExperienceBuffer**: 백테스트 경험을 PPO 학습 데이터로 변환
- **MABFeedbackProcessor**: 전략별 성과를 MAB 보상으로 반영
- **StateEncoder**: 시장 상태를 신경망 입력으로 인코딩

### 피드백 플로우
```
Backtest Results → Feedback Bridge → [PPO Buffer | MAB Rewards] → Strategy Learning
```

## 🚀 사용법

### 기본 백테스트

```python
from backtest.v2.layers.controller_layer import BacktestController, BacktestMode

# 컨트롤러 생성
controller = BacktestController(
    initial_capital=1000000,
    mode=BacktestMode.NORMAL,
    enable_multiframe=True,
    enable_exploration=False
)

# 전략 시스템 초기화
controller.initialize_strategies(
    sentiment_file=\"data/sentiment.csv\",
    enable_ppo=True
)

# 백테스트 실행
result = controller.run_backtest(
    price_data_path=\"data/price_data.csv\",
    sentiment_data_path=\"data/sentiment.csv\",
    window_size=100,
    indicators=[\"sma_20\", \"rsi\", \"macd\", \"atr\"]
)
```

### 복수 백테스트

```python
from backtest.v2.layers.controller_layer import BacktestOrchestrator

# 설정 목록
configurations = [
    {
        \"name\": \"normal_mode\",
        \"price_data_path\": \"data/price_data.csv\",
        \"mode\": BacktestMode.NORMAL,
        \"enable_exploration\": False
    },
    {
        \"name\": \"exploration_mode\",
        \"price_data_path\": \"data/price_data.csv\",
        \"mode\": BacktestMode.EXPLORATION,
        \"enable_exploration\": True
    }
]

# 병렬 실행
orchestrator = BacktestOrchestrator(n_workers=4)
results = orchestrator.run_multiple_backtests(configurations, parallel=True)
```

### 워크포워드 분석

```python
# 시간 분할 검증
wf_result = orchestrator.walk_forward_analysis(
    base_config={
        \"price_data_path\": \"data/price_data.csv\",
        \"start_date\": \"2023-01-01\",
        \"end_date\": \"2023-12-31\"
    },
    n_windows=10,
    train_ratio=0.7
)
```

## 📊 성능 특징

### 속도 최적화
- **지표 캐싱**: 중복 계산 제거로 30-50% 성능 향상
- **병렬 처리**: 다중 백테스트 동시 실행
- **메모리 효율**: 데이터 윈도우 기반 점진적 처리

### 정확도 개선
- **동적 슬리피지**: 변동성과 거래량 반영
- **시장 영향**: 주문 크기 대비 가격 영향 모델링
- **세션별 필터**: 시간대별 노이즈 수준 차등 적용

## 📈 메트릭 시스템

### 표준화된 메트릭
- **수익성**: ROI, 연율화 수익률
- **리스크 조정**: Sharpe, Sortino, Calmar Ratio
- **리스크**: Max Drawdown, VaR, CVaR
- **거래 통계**: 승률, Profit Factor, Expectancy
- **효율성**: 평균 보유시간, 거래 빈도

### 표본 기반 가중치
- **최소 표본**: 30개 거래 미만 시 페널티
- **최적 표본**: 100개 거래에서 만점
- **신뢰도 계산**: Wilson Score Interval 기반

## 🛠️ 의존성

### 필수 라이브러리
```bash
pip install pandas numpy talib tqdm
```

### 기존 AuroraQ 시스템
- core.strategy_selector
- strategy.mab_selector
- core.ppo_agent_proxy

## 📋 예제

`example_usage.py` 파일에서 다양한 사용 예제를 확인할 수 있습니다:

1. **간단한 백테스트**: 기본 기능 테스트
2. **탐색 모드**: 랜덤 전략 선택으로 데이터 수집
3. **복수 백테스트**: 여러 설정 병렬 비교
4. **워크포워드 분석**: 시간 분할 검증

## 🔧 확장성

### 새로운 지표 추가
`DataLayer.calculate_indicators()` 메서드 확장

### 새로운 슬리피지 모델
`SlippageModel` 클래스 상속 및 구현

### 새로운 평가 메트릭
`MetricsCalculator._calculate_composite_score()` 수정

## ⚠️ 주의사항

1. **데이터 품질**: OHLCV 데이터의 무결성 확인 필요
2. **메모리 사용량**: 큰 데이터셋의 경우 윈도우 크기 조정
3. **피드백 시스템**: PPO/MAB 통합 시 충분한 학습 데이터 확보

## 🎯 실제 전략 백테스트 실행하기

### 준비 단계

1. **데이터 파일 준비** (CSV 형식):
   ```
   data/btc_5m_sample.csv  (가격 데이터)
   data/sentiment_sample.csv  (감정 데이터, 선택사항)
   ```

2. **가격 데이터 형식**:
   ```csv
   timestamp,open,high,low,close,volume
   2024-01-01 00:00:00,50000,50100,49900,50050,1000
   ```

### 실행 명령어

#### Windows:
```cmd
# 배치 파일로 실행 (가장 간단)
run_backtest.bat

# 또는 Python 직접 실행
python run_backtest.py --price-data data/btc_5m_sample.csv --initial-capital 1000000
```

#### Linux/Mac:
```bash
# 쉘 스크립트로 실행
./run_backtest.sh

# 또는 Python 직접 실행
python3 run_backtest.py --price-data data/btc_5m_sample.csv --initial-capital 1000000
```

### 고급 옵션

```bash
# 날짜 범위 지정하여 백테스트
python run_backtest.py \
    --price-data data/btc_5m.csv \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --initial-capital 1000000 \
    --window-size 100 \
    --mode normal

# 탐색 모드로 다양한 전략 시도
python run_backtest.py \
    --price-data data/btc_5m.csv \
    --enable-exploration \
    --mode exploration

# 감정 데이터 포함
python run_backtest.py \
    --price-data data/btc_5m.csv \
    --sentiment-data data/sentiment.csv
```

### 등록된 전략들

다음 전략들이 자동으로 등록되어 백테스트에 사용됩니다:

1. **RuleStrategyA**: EMA 크로스오버 + ADX 트렌드 강도
2. **RuleStrategyB**: RSI 역전 + 볼륨 확인  
3. **RuleStrategyC**: 볼린저 밴드 + 모멘텀
4. **RuleStrategyD**: MACD + 가격 액션
5. **RuleStrategyE**: 복합 지표 + 시간 필터

### 결과 확인

백테스트 완료 후 `reports/backtest/` 디렉토리에서 결과를 확인할 수 있습니다:

- `*_trades.csv`: 개별 거래 기록
- `*_metrics.csv`: 성과 지표
- `*_report.json`: 종합 보고서
- `strategy_comparison_*.csv`: 전략별 비교

### 트러블슈팅

1. **모듈 임포트 오류**: 프로젝트 루트에서 실행하세요
2. **데이터 파일 없음**: data/ 디렉토리에 CSV 파일을 준비하세요
3. **전략 등록 실패**: strategy/ 디렉토리의 전략 파일들을 확인하세요

## 📞 지원

문제 발생 시 로그 레벨을 DEBUG로 설정하여 상세 정보 확인:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```