# AuroraQ Production - 실시간 하이브리드 거래 시스템

> **중요**: 이 컴포넌트는 AuroraQ Shared 통합 모듈 시스템을 사용합니다. 중복 코드가 제거되고 공통 모듈이 통합되었습니다.

## 📋 시스템 개요

AuroraQ Production은 PPO(Proximal Policy Optimization) 강화학습과 Rule-based 전략을 결합한 실시간 하이브리드 거래 시스템입니다.

**통합 구조**: 
- 유틸리티, 리스크 관리, 포지션 관리 → `AuroraQ_Shared` 통합 모듈 사용
- Production 고유 기능: 전략, 실행, 최적화, 센티멘트 분석

## 🏗️ 패키지 구조

```
AuroraQ_Production/
├── README.md                 # 패키지 메인 문서
├── requirements.txt          # 의존성 패키지 목록
├── setup.py                 # 패키지 설치 스크립트
├── main.py                  # 메인 실행 파일
├── config.yaml              # 기본 설정 파일
│
├── core/                    # 핵심 시스템
│   ├── __init__.py
│   ├── realtime_system.py   # 실시간 거래 시스템
│   ├── hybrid_controller.py # 하이브리드 제어기
│   └── market_data.py       # 마켓 데이터 제공
│
├── strategies/              # 거래 전략 (Production 고유)
│   ├── __init__.py
│   ├── ppo_strategy.py      # PPO 강화학습 전략
│   ├── rule_strategies.py   # Rule-based 전략들
│   ├── strategy_adapter.py  # 전략 어댑터
│   └── strategy_registry.py # 전략 레지스트리
│
├── execution/               # 체결 시스템 (Production 고유)
│   ├── __init__.py
│   ├── execution_layer.py   # 실행 레이어
│   ├── order_manager.py     # 주문 관리
│   └── smart_execution.py   # 스마트 체결
│
├── optimization/            # 최적화 시스템 (Production 고유)
│   ├── __init__.py
│   ├── combination_optimizer.py # 조합 최적화
│   ├── parameter_tuner.py   # 파라미터 튜닝
│   └── results/             # 최적화 결과 저장
│
├── sentiment/               # 센티멘트 분석 (Production 고유)
│   ├── __init__.py
│   ├── sentiment_analyzer.py # 센티멘트 분석기
│   ├── news_collector.py    # 뉴스 수집
│   └── sentiment_scorer.py  # 센티멘트 점수화
│
├── data/                    # 데이터 관리
│   ├── __init__.py
│   ├── data_provider.py     # 데이터 제공자
│   ├── preprocessor.py      # 데이터 전처리
│   └── storage.py           # 데이터 저장
│
├── configs/                 # 설정 파일들
│   ├── trading_config.yaml  # 거래 설정
│   ├── risk_config.yaml     # 리스크 설정
│   └── strategy_config.yaml # 전략 설정
│
└── tests/                   # 테스트 코드
    ├── __init__.py
    ├── test_realtime.py     # 실시간 시스템 테스트
    ├── test_strategies.py   # 전략 테스트
    └── test_optimization.py # 최적화 테스트
```

## 🔗 통합 모듈 사용 (AuroraQ_Shared)

### 자동 통합되는 기능들
- **포지션 관리**: `AuroraQ_Shared.position_management.EnhancedPositionManager` 사용
- **리스크 관리**: `AuroraQ_Shared.risk_management.IntegratedRiskManager` 사용  
- **유틸리티**: `AuroraQ_Shared.utils` (logger, config_manager, metrics) 사용
- **테스트**: `AuroraQ_Shared.tests.unified_test_runner` 통합 테스트 사용

## 🚀 주요 기능

### 1. 하이브리드 거래 전략
- **PPO 강화학습**: 시장 패턴 학습 및 적응
- **Rule-based 전략**: 기술적 분석 기반 거래 규칙
- **하이브리드 모드**: Ensemble, Consensus, Competition

### 2. 실시간 거래 시스템
- **실시간 데이터 스트리밍**: 1초 간격 가격 데이터
- **자동 포지션 관리**: 손절/익절 자동 실행
- **리스크 제어**: 일일 거래 한도, 최대 포지션 크기

### 3. 최적화 시스템
- **그리드 서치**: 전략 조합 최적화
- **성과 분석**: Sharpe ratio, 승률, 최대 낙폭
- **자동 추천**: 최적 전략 조합 제안

### 4. 센티멘트 분석
- **뉴스 감정 분석**: 시장 심리 반영
- **소셜 미디어 모니터링**: 실시간 감정 추적
- **감정 지표 통합**: 거래 신호에 반영

## 📦 설치 및 실행

### 1. 통합 모듈 시스템 확인
```bash
# 통합 테스트 러너로 전체 시스템 검증
cd ../AuroraQ_Shared
python tests/unified_test_runner.py

# 또는 빠른 검증
python tests/quick_validation.py
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 설정 파일 편집
```bash
# config.yaml 파일에서 거래 설정 조정
nano config.yaml
```

### 4. 실행
```bash
# 실시간 거래 시스템 시작
python main.py

# 테스트 모드로 실행
python main.py --test

# 데모 모드로 실행
python main.py --demo
```

### 5. 통합 모듈 사용 예시
```python
# 통합 포지션 관리자 사용
from AuroraQ_Shared.position_management import EnhancedPositionManager

# 통합 리스크 관리자 사용  
from AuroraQ_Shared.risk_management import IntegratedRiskManager

# 통합 유틸리티 사용
from AuroraQ_Shared.utils import get_logger, load_config, calculate_performance_metrics

# 포지션 관리자 생성 (Legacy 호환)
position_manager = EnhancedPositionManager(
    initial_capital=1000000,
    use_legacy_interface=True  # Production 호환성
)

# 리스크 관리자 생성
risk_manager = IntegratedRiskManager(
    position_manager=position_manager
)
```

## ⚙️ 설정 옵션

### 거래 설정
- `max_position_size`: 최대 포지션 크기 (기본: 0.1)
- `emergency_stop_loss`: 긴급 손절선 (기본: 5%)
- `max_daily_trades`: 일일 최대 거래 횟수 (기본: 10)

### 전략 설정
- `hybrid_mode`: 하이브리드 모드 (ensemble/consensus/competition)
- `ppo_weight`: PPO 전략 가중치 (기본: 0.3)
- `rule_strategies`: 사용할 Rule 전략 목록

### 리스크 설정
- `risk_tolerance`: 리스크 허용도 (conservative/moderate/aggressive)
- `position_limits`: 포지션 한도 설정
- `drawdown_limit`: 최대 낙폭 한도

## 📊 성과 모니터링

### 실시간 지표
- 신호 생성률
- 거래 실행률
- 포지션 현황
- 손익 현황

### 일일 리포트
- HTML 리포트 자동 생성
- 전략별 기여도 분석
- 리스크 지표 추적

## 🔧 확장 가능성

### API 연동
- 바이낸스, 업비트 등 거래소 API
- 실시간 뉴스 API
- 소셜 미디어 API

### 고급 기능
- 포트폴리오 최적화
- 동적 리밸런싱
- 멀티 에셋 거래

## 📞 지원

문제 발생 시 로그 파일(`logs/`)을 확인하거나 테스트 모드로 실행하여 디버깅하세요.

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.