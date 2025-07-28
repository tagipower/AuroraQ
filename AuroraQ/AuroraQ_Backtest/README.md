# 🎯 AuroraQ Backtest - 고성능 백테스팅 프레임워크

> **중요**: 이 컴포넌트는 AuroraQ Shared 통합 모듈 시스템을 사용합니다. 중복 코드가 제거되고 공통 모듈이 통합되었습니다.

## 📌 개요

AuroraQ Backtest는 암호화폐 거래 전략을 검증하고 최적화하기 위한 전문 백테스팅 프레임워크입니다. 실제 거래 환경을 정확히 시뮬레이션하며, 다양한 전략을 동시에 테스트하고 비교할 수 있습니다.

**통합 구조**: 
- 성능 지표, 로깅, 리스크 계산 → `AuroraQ_Shared` 통합 모듈 사용
- Backtest 고유 기능: 백테스트 엔진, 데이터 관리, 전략 테스트

### 🌟 핵심 특징

- **고속 백테스팅**: 벡터화 연산으로 수년간의 데이터를 분 단위로 처리
- **정확한 시뮬레이션**: 슬리피지, 수수료, 시장 충격을 고려한 현실적 모델링
- **다중 전략 지원**: PPO, Rule-based, 하이브리드 전략 동시 테스트
- **포괄적 분석**: 50+ 성과 지표 자동 계산 (통합 metrics 모듈 사용)
- **시각화 리포트**: 대화형 차트와 상세 분석 리포트 자동 생성
- **실거래 동기화**: 실거래 데이터 기반 파라미터 자동 보정

## 🚀 빠른 시작

### 1분 백테스트 (통합 모듈 사용)

```python
# 통합 모듈 import
from AuroraQ_Shared.utils import get_logger, calculate_performance_metrics
from AuroraQ_Shared.position_management import EnhancedPositionManager
from AuroraQ_Shared.risk_management import IntegratedRiskManager

# Backtest 고유 모듈
from core.backtest_engine import BacktestEngine
from strategies.hybrid_strategy import HybridStrategy

# 통합 로거 사용
logger = get_logger("BacktestExample")

# 백테스트 엔진 초기화 (통합 모듈 연동)
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005,
    use_shared_modules=True  # 통합 모듈 사용
)

# 전략 설정
strategy = HybridStrategy(
    ppo_weight=0.3,
    rule_weight=0.7
)

# 백테스트 실행
results = engine.run(
    strategy=strategy,
    data_path="data/btc_usdt_1h.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# 통합 성능 지표로 결과 분석
performance_metrics = calculate_performance_metrics(results.trades)
logger.info(f"Sharpe Ratio: {performance_metrics.sharpe_ratio}")
logger.info(f"Win Rate: {performance_metrics.win_rate}")

# 결과 출력
print(results.summary())
```

### 실거래 동기화 백테스트

```python
from AuroraQ_Shared.calibration import CalibrationManager
from AuroraQ_Shared.integration import create_synchronized_backtest_environment

# 실거래 데이터로 파라미터 보정
calibration_manager = CalibrationManager()
calibration_result = calibration_manager.calibrate_parameters("BTC-USD")

# 보정된 파라미터로 백테스트 환경 생성
sync_engine = create_synchronized_backtest_environment(
    calibrated_params={
        'slippage': calibration_result.calibrated_slippage,
        'commission': calibration_result.calibrated_commission,
        'fill_rate': calibration_result.calibrated_fill_rate
    }
)

# 실거래와 동일한 조건으로 백테스트 실행
sync_results = sync_engine.run_risk_aware_backtest(
    strategy=strategy,
    data=market_data,
    enable_periodic_calibration=True
)
```

## 📊 주요 기능

### 1. 데이터 관리
- 다양한 시간 프레임 지원 (1분 ~ 1일)
- 자동 데이터 검증 및 정제
- 실시간 데이터 업데이트

### 2. 전략 백테스팅
- 단일/다중 전략 동시 테스트
- 파라미터 최적화
- Walk-forward 분석

### 3. 리스크 분석
- 최대 낙폭 (MDD) 분석
- VaR/CVaR 계산
- 몬테카를로 시뮬레이션

### 4. 성과 분석
- Sharpe/Sortino 비율
- 승률 및 손익비
- 월별/연도별 수익률

## 📁 프로젝트 구조

```
AuroraQ_Backtest/
├── core/               # 백테스트 엔진 핵심 (Backtest 고유)
├── data/               # 가격 데이터 관리
├── strategies/         # 전략 구현체
├── indicators/         # 기술 지표
├── reports/            # 리포트 생성
├── configs/            # 설정 파일
├── tests/              # 테스트 코드
└── examples/           # 예제 코드
```

## 🔗 통합 모듈 사용 (AuroraQ_Shared)

### 자동 통합되는 기능들
- **성능 지표**: `AuroraQ_Shared.utils.metrics` - 통일된 성과 계산
- **로깅 시스템**: `AuroraQ_Shared.utils.logger` - 일관된 로그 관리
- **리스크 분석**: `AuroraQ_Shared.risk_management` - VaR/CVaR 계산
- **실거래 보정**: `AuroraQ_Shared.calibration` - 파라미터 자동 조정
- **테스트**: `AuroraQ_Shared.tests.unified_test_runner` 통합 테스트 사용

## 🔧 설치

### 1. 통합 모듈 시스템 확인
```bash
# 통합 테스트 러너로 전체 시스템 검증
cd ../AuroraQ_Shared
python tests/unified_test_runner.py

# Backtest 고유 기능 테스트
python tests/test_production_modules.py
```

### 2. 의존성 설치
```bash
# 기본 설치
pip install -r requirements.txt

# GPU 가속 설치 (선택)
pip install -r requirements-gpu.txt
```

### 3. 통합 모듈 사용 설정
```python
# 통합 설정 관리자 사용
from AuroraQ_Shared.utils import load_config

# Backtest 컴포넌트용 설정 로드
config = load_config(component_type="backtest")

# 통합 로거 설정
from AuroraQ_Shared.utils import get_logger
logger = get_logger("AuroraQ_Backtest")
```

## 📈 성능 벤치마크

| 데이터 크기 | 전략 수 | 처리 시간 | 메모리 사용 |
|------------|---------|-----------|-------------|
| 1년 (1시간) | 1 | 0.5초 | 100MB |
| 5년 (1시간) | 1 | 2.3초 | 500MB |
| 1년 (1분) | 1 | 8.7초 | 2GB |
| 1년 (1시간) | 10 | 4.8초 | 1GB |

## 📚 문서

- [설치 가이드](INSTALLATION.md)
- [사용자 매뉴얼](USER_MANUAL.md)
- [전략 개발 가이드](STRATEGY_GUIDE.md)
- [API 레퍼런스](API_REFERENCE.md)

## 🤝 기여

백테스트 모듈 개선에 기여하실 수 있습니다:
- 새로운 전략 추가
- 성능 최적화
- 버그 리포트
- 문서 개선

## 📄 라이선스

MIT License - 자유롭게 사용하고 수정 가능합니다.