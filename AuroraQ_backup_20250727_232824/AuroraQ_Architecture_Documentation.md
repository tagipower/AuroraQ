# AuroraQ 아키텍처 문서

## 프로젝트 개요

AuroraQ는 **강화학습(PPO)과 Multi-Armed Bandit 알고리즘을 활용한 적응적 암호화폐 트레이딩 시스템**입니다. 뉴스 감정 분석과 기술적 지표를 결합하여 실시간으로 최적의 트레이딩 전략을 선택합니다.

### 핵심 철학
- **적응적 전략 선택**: MAB로 최적 전략 자동 선택
- **멀티모달 신호**: 가격 + 뉴스 감정 분석 결합  
- **리스크 관리**: 포지션 추적 + 위험 관리
- **연속 학습**: PPO 에이전트 지속 훈련
- **시나리오 테스팅**: 다양한 시장 조건 시뮬레이션

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    AuroraQ Trading System                   │
├─────────────────────────────────────────────────────────────┤
│ Orchestration (loops/)         │ Configuration (config/)    │
│ ├─ run_loop.py                │ ├─ env_loader.py           │
│ ├─ sentiment_loop.py          │ ├─ trade_config.yaml       │
│ ├─ train_loop.py              │ └─ strategy_config.yaml    │
│ └─ weight_update_loop.py       │                            │
├─────────────────────────────────────────────────────────────┤
│ Strategy Layer (strategy/)     │ Core Engine (core/)        │
│ ├─ StrategySelector            │ ├─ PPOAgentProxy           │
│ ├─ MABSelector                 │ ├─ OrderSignal             │
│ ├─ RuleStrategy A-E            │ ├─ RiskManager             │
│ └─ BaseRuleStrategy            │ └─ PositionTracker         │
├─────────────────────────────────────────────────────────────┤
│ Sentiment Analysis (sentiment/)│ RL Environment (envs/)     │
│ ├─ SentimentAnalyzer           │ ├─ BaseTradingEnv          │
│ ├─ SentimentRouter             │ ├─ RuleStrategyEnv A-E     │
│ └─ SentimentScoreRefiner       │ └─ SentimentTradingEnv     │
├─────────────────────────────────────────────────────────────┤
│ Backtesting (backtest/)        │ Utilities (utils/)         │
│ ├─ BacktestLoop                │ ├─ DataLoader              │
│ ├─ ScenarioRunner              │ ├─ Indicators              │
│ └─ ReportGenerator             │ └─ Logger                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 디렉토리별 상세 분석

## 📊 core/ - 메인 트레이딩 엔진

핵심 트레이딩 로직과 실행 엔진을 담당하는 모듈들입니다.

### 주요 컴포넌트

#### **PPOAgentProxy**
```python
class PPOAgentProxy:
    def load_model()          # PPO 모델 로드
    def generate_signal()     # RL 기반 시그널 생성
    def _preprocess()         # 상태 전처리
```
- **역할**: Stable-Baselines3 PPO 모델 래퍼
- **기능**: 강화학습 기반 트레이딩 시그널 생성
- **연동**: StatePreprocessor로 상태 정규화

#### **StrategySelector**
```python
class StrategySelector:
    def select()              # 메타 전략 선택 및 실행
    def adjust_score()        # 다중 요인 점수 조정
    def get_sentiment_score() # 감정 점수 조회
    def _select_strategy()    # MAB 기반 전략 선택
```
- **역할**: 메타 전략 선택기 (최고 중요도)
- **기능**: 
  - Rule A~E + PPO 전략 통합 관리
  - MAB 기반 동적 전략 선택
  - 감정 점수 통합 및 자동 정렬
  - 성과 기반 자동 가중치 조정

#### **OrderSignal**
```python
class OrderSignal:
    def execute()             # 주문 실행
    def __init__()           # 주문 신호 생성
```
- **역할**: 주문 실행 및 포지션 추적
- **기능**: Binance API 연동, 포지션 상태 파일 기록

#### **RiskManager**
```python
def evaluate_risk()          # 종합 리스크 평가
def allocate_capital()       # 자본 배분
def adjust_leverage()        # 레버리지 동적 조정
def calculate_position_size() # Kelly Criterion 기반 포지션 크기

class IntegratedRiskManager:
    def comprehensive_risk_check() # 종합 리스크 검사
```
- **역할**: 다층 리스크 관리 시스템
- **기능**: 
  - 가격/감정/변동성/전략별 리스크 평가
  - 동적 자본 배분 및 레버리지 조정
  - 향상된 포지션 사이징 (변동성/Kelly 기반)
  - 거래 비용 최적화

---

## 🎯 strategy/ - 전략 레이어

다양한 트레이딩 전략과 메타 선택 로직을 담당합니다.

### 전략 아키텍처

#### **BaseRuleStrategy (추상 베이스)**
```python
class BaseRuleStrategy(ABC):
    @abstractmethod
    def should_enter()        # 진입 조건 확인
    @abstractmethod  
    def should_exit()         # 청산 조건 확인
    @abstractmethod
    def calculate_indicators() # 기술적 지표 계산
    
    def generate_signal()     # 메인 시그널 생성
    def custom_observation()  # PPO 학습용 관측값
    def custom_reward()       # PPO 학습용 보상
```
- **역할**: 모든 룰 전략의 공통 인터페이스
- **기능**: 
  - 포지션 관리 및 리스크 통합
  - 상세한 시그널 정보 제공
  - PPO 인터페이스 지원
  - 성과 추적 및 상태 관리

#### **RuleStrategyA (EMA 크로스오버)**
```python
class RuleStrategyA(BaseRuleStrategy):
    def compute_ema()         # 지수이동평균 계산
    def compute_adx()         # ADX 트렌드 강도
    def dynamic_thresholds()  # 변동성 기반 동적 임계값
```
- **전략**: EMA 크로스오버 + ADX 트렌드 강도
- **진입**: 단기EMA > 장기EMA + ADX > 임계값 + 감정 ≥ 0
- **청산**: 동적 TP/SL, 감정 악화, 시간 초과

#### **MABSelector (Multi-Armed Bandit)**
```python
class MABSelector:
    def select()              # 전략 선택 (알고리즘별)
    def update()              # 보상 업데이트 (증분 통계)
    def _select_epsilon_greedy() # Epsilon-Greedy 선택
    def _select_ucb()         # Upper Confidence Bound
    def _select_thompson()    # Thompson Sampling
```
- **역할**: 메타 전략 선택 알고리즘
- **알고리즘**: Epsilon-Greedy, UCB, Thompson Sampling
- **기능**: 
  - 미탐색 전략 우선 선택
  - Epsilon decay 및 증분 통계 업데이트
  - 베타 분포 기반 Thompson Sampling

---

## 📰 sentiment/ - 감정 분석 시스템

뉴스 텍스트에서 시장 감정을 분석하는 AI 기반 시스템입니다.

### 감정 분석 파이프라인

#### **SentimentAnalyzer (FinBERT 기반)**
```python
class SentimentAnalyzer:
    def analyze()             # 기본 감정 점수 분석
    def analyze_detailed()    # 상세 감정 분석 + 메타데이터
    def analyze_batch()       # 배치 처리 (성능 최적화)
    def normalize_score()     # -1~1 범위 점수 정규화
    def extract_keywords()    # 키워드 추출 (LRU 캐시)
    def tag_scenario()        # 시나리오 태깅
```
- **모델**: ProsusAI/finbert (금융 특화 BERT)
- **기능**: 
  - GPU/CPU 자동 선택
  - 토큰 길이 체크 및 truncation
  - 시나리오별 감정 분류 (랠리, 공포, 인플레이션 등)

#### **SentimentRouter (모드별 라우팅)**
```python
class SentimentRouter:
    def get_score()           # 모드별 감정 점수 조회
    def switch_mode()         # 실시간/백테스트 모드 전환
    def get_score_with_metadata() # 메타데이터 포함 조회
```
- **모드**: Live (실시간 분석) / Backtest (저장된 데이터)
- **기능**: 캐싱, 모드 전환, 싱글톤 패턴

#### **SentimentScoreRefiner (키워드 기반)**
```python
class SentimentScoreRefiner:
    def analyze_text()        # 텍스트 감정 점수 계산
    def _analyze_sentence()   # 문장 단위 분석
    def get_keyword_impact()  # 키워드 영향도 분석
```
- **방식**: 정의된 키워드 사전 기반 점수 계산
- **특징**: 
  - 강화/약화 표현 처리 (very, slightly 등)
  - 부정 표현 처리 (not, never 등)
  - 시그모이드 함수로 0~1 정규화

---

## 🔧 utils/ - 유틸리티 모듈

공통 기능과 헬퍼 함수들을 제공합니다.

### 주요 유틸리티

- **logger.py**: 통합 로깅 시스템
- **indicators.py**: 기술적 지표 계산 (RSI, MACD, Bollinger Bands 등)
- **data_loader.py**: 시장 데이터 로드 및 전처리
- **reward_calculator.py**: 보상 함수 계산
- **telegram_notifier.py**: 텔레그램 알림 시스템

---

## 🔄 loops/ - 시스템 오케스트레이션

전체 시스템의 실행 흐름을 관리하는 루프들입니다.

### 주요 루프

- **run_loop.py**: 메인 트레이딩 루프
- **sentiment_loop.py**: 감정 분석 수집 루프  
- **train_loop.py**: PPO 모델 재훈련 루프
- **weight_update_loop.py**: 전략 가중치 업데이트

---

## 📊 backtest/ - 백테스팅 프레임워크

과거 데이터를 이용한 전략 성능 검증 시스템입니다.

### 백테스팅 컴포넌트

- **BacktestLoop**: 메인 백테스팅 엔진
- **ScenarioRunner**: 다양한 시나리오 테스트
- **ReportGenerator**: 성과 리포트 생성

---

## 🎮 envs/ - 강화학습 환경

OpenAI Gym 호환 트레이딩 환경들입니다.

### 환경 구조

- **BaseTradingEnv**: 공통 트레이딩 환경
- **RuleStrategyEnv A~E**: 각 룰 전략별 환경
- **SentimentTradingEnv**: 감정 분석 통합 환경

---

## ⚙️ config/ - 설정 관리

시스템 전체의 설정 파일과 로더들을 관리합니다.

### 설정 시스템

- **trade_config.yaml**: 거래 설정
- **strategy_config.yaml**: 전략 파라미터
- **reward_config.yaml**: 보상 설정
- **mab_config.yaml**: MAB 알고리즘 설정

---

## 🔀 시스템 플로우

### 1. 메인 트레이딩 플로우
```
Market Data → StrategySelector → MABSelector → Selected Strategy 
     ↓                                              ↓
Sentiment Analysis ←                      → Order Signal → Binance API
     ↓                                              ↓
SentimentRouter →                       → Position Tracker
```

### 2. 감정 분석 플로우
```
News Text → SentimentRouter → [Live: SentimentAnalyzer | Backtest: CSV] 
     ↓                                    ↓
Sentiment Score → Strategy Adjustment → Trading Decision
```

### 3. 강화학습 플로우
```
Market State → PPOAgentProxy → Action → Environment → Reward
     ↑                                           ↓
Training Loop ←                           → Model Update
```

---

## 🛡️ 리스크 관리 시스템

### 다층 리스크 아키텍처

1. **기본 리스크 평가**: 가격, 감정, 변동성 체크
2. **향상된 리스크 필터**: 시장 체제, 포지션 수 제한
3. **동적 포지션 사이징**: 변동성/Kelly Criterion 기반
4. **거래 비용 최적화**: 수수료/슬리피지 고려
5. **통합 리스크 관리**: 모든 검사 통합 실행

### 포지션 사이징 방법
- **Fixed Amount**: 고정 금액
- **Fixed Fractional**: 고정 비율 (15%)
- **Kelly Criterion**: 수학적 최적 비율
- **Volatility Based**: 변동성 역비례
- **Risk Parity**: 목표 변동성 기반

---

## 📈 성과 측정 시스템

### 핵심 메트릭
- **ROI**: 투자 수익률
- **Sharpe Ratio**: 위험 대비 수익률  
- **Maximum Drawdown**: 최대 낙폭
- **Win Rate**: 승률
- **Profit Factor**: 수익 배수
- **Calmar Ratio**: 연수익률/최대낙폭

### 전략별 성과 추적
- 실시간 성과 모니터링
- MAB 보상 기반 전략 가중치 자동 조정
- 백테스트 vs 실거래 성과 비교

---

## 🔧 설정 및 확장성

### 주요 설정 포인트
1. **전략 파라미터**: EMA 기간, ADX 임계값, TP/SL 비율
2. **MAB 설정**: Epsilon, decay rate, 알고리즘 선택
3. **리스크 파라미터**: 최대 포지션, 손절 비율, 레버리지
4. **감정 분석**: 모델 선택, 키워드 가중치, 정규화 방법

### 확장 가능성
- **새로운 전략 추가**: BaseRuleStrategy 상속
- **다른 자산 지원**: 환경 설정 변경
- **새로운 감정 모델**: SentimentAnalyzer 교체
- **추가 리스크 규칙**: RiskManager 확장

---

## 🚀 실행 방법

### 1. 실시간 트레이딩
```bash
python loops/run_loop.py
```

### 2. 백테스팅
```bash
python backtest/backtest_loop.py
```

### 3. 모델 훈련
```bash
python loops/train_loop.py
```

### 4. 감정 분석 수집
```bash
python loops/sentiment_loop.py
```

---

## 📝 요약

AuroraQ는 **현대적인 퀀트 트레이딩 시스템**으로, 다음과 같은 특징을 가집니다:

### 🎯 **핵심 강점**
- **적응적 전략 선택**: MAB 알고리즘으로 시장 상황에 맞는 최적 전략 자동 선택
- **멀티모달 분석**: 가격 데이터 + AI 감정 분석의 강력한 조합
- **고도화된 리스크 관리**: 다층 리스크 시스템으로 안전한 거래
- **연속 학습**: PPO 에이전트가 시장 변화에 적응하며 지속 개선
- **모듈화된 설계**: 각 컴포넌트의 독립적 개발/테스트 가능

### 🔄 **시스템 흐름**
1. **데이터 수집**: 가격 + 뉴스 데이터 실시간 처리
2. **감정 분석**: FinBERT 기반 시장 감정 스코어링
3. **전략 선택**: MAB가 최적 전략을 동적 선택
4. **리스크 관리**: 다층 검증으로 안전한 거래 결정
5. **주문 실행**: Binance API 연동으로 실제 거래
6. **성과 피드백**: 결과를 MAB와 PPO 모델에 반영

### 📊 **기술 스택**
- **ML/AI**: Stable-Baselines3 (PPO), Transformers (FinBERT)
- **수치 계산**: NumPy, Pandas, TA-Lib
- **API 연동**: Binance, 뉴스 피드
- **백테스팅**: 자체 개발 프레임워크
- **모니터링**: 텔레그램 알림, 상세 로깅

이 시스템은 **학술적 엄밀성과 실용적 효율성**을 모두 갖춘 전문가급 트레이딩 솔루션입니다.