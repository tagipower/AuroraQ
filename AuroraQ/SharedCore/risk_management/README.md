# Risk Management Module

고도화된 리스크 관리 모듈로 VaR (Value at Risk), CVaR (Conditional VaR), MDD (Maximum Drawdown) 기반의 동적 리스크 관리를 제공합니다.

## 📋 주요 기능

### 1. 다차원 VaR 계산
- **Historical VaR**: 과거 수익률 분포 기반
- **Parametric VaR**: 정규분포 가정
- **Monte Carlo VaR**: 시뮬레이션 기반
- **Cornish-Fisher VaR**: 비정규분포 고려

### 2. 고급 리스크 지표
- **CVaR (Expected Shortfall)**: VaR 초과 손실의 평균
- **Component VaR**: 포지션별 VaR 기여도
- **Incremental VaR**: 신규 포지션 추가 시 VaR 변화
- **Rolling VaR**: 시간대별 VaR 변화 추적

### 3. 동적 포지션 관리
- **VaR 기반 포지션 사이징**: 리스크 한도 내에서 최적 포지션 크기 결정
- **MDD 기반 포지션 축소**: 낙폭 한도 도달 시 자동 포지션 축소
- **실시간 리스크 모니터링**: 거래 실행 시마다 리스크 지표 업데이트

### 4. 포트폴리오 리스크 분석
- **집중도 분석**: 허핀달 지수, 포지션 집중도
- **상관관계 분석**: 포지션 간 상관관계 및 다각화 효과
- **시나리오 분석**: 극단적 시장 상황에서의 포트폴리오 영향
- **효율적 투자선**: 리스크-수익률 최적화

## 🏗️ 모듈 구조

```
risk_management/
├── __init__.py                    # 모듈 초기화
├── advanced_risk_manager.py       # 고도화된 리스크 관리자
├── var_calculator.py             # VaR/CVaR 계산기
├── risk_models.py                # 리스크 데이터 모델
├── portfolio_risk_analyzer.py    # 포트폴리오 리스크 분석기
└── README.md                     # 모듈 문서
```

## 🚀 사용법

### 기본 설정

```python
from AuroraQ_Shared.risk_management import AdvancedRiskManager, RiskConfig
from AuroraQ_Shared.position_management import UnifiedPositionManager

# 리스크 설정
config = RiskConfig(
    var_limit_pct=0.05,           # 일일 5% VaR 한도
    cvar_limit_pct=0.08,          # 일일 8% CVaR 한도
    max_drawdown_limit=0.15,      # 15% 최대 낙폭
    drawdown_alert_threshold=0.10  # 10% 낙폭 경고
)

# 포지션 관리자 초기화
position_manager = UnifiedPositionManager(initial_capital=100000)

# 리스크 관리자 초기화
risk_manager = AdvancedRiskManager(
    position_manager=position_manager,
    config=config
)
```

### VaR 계산

```python
from AuroraQ_Shared.risk_management import VaRCalculator
import numpy as np

# 수익률 데이터 (예시)
returns = np.random.normal(0.001, 0.02, 252)  # 1년치 일일 수익률

# VaR 계산기 초기화
var_calculator = VaRCalculator()

# 95% VaR 계산
var_result = var_calculator.calculate_var(
    returns, 
    method='historical',
    confidence_level=0.95,
    portfolio_value=100000
)

print(f"95% VaR: ${var_result['var']:,.2f} ({var_result['var_pct']:.2%})")
print(f"95% CVaR: ${var_result['cvar']:,.2f} ({var_result['cvar_pct']:.2%})")
```

### 동적 포지션 사이징

```python
# VaR 기반 포지션 크기 권고
sizing_recommendation = risk_manager.get_position_sizing_recommendation(
    symbol="AAPL",
    current_price=150.0,
    signal_confidence=0.8
)

print(f"권장 포지션 크기: {sizing_recommendation['recommended_size']:.2f}주")
print(f"조정 요인: {sizing_recommendation['adjustments']['final_adjustment']:.2f}")
```

### 리스크 지표 모니터링

```python
# 포트폴리오 스냅샷 업데이트
snapshot = risk_manager.update_portfolio_snapshot(
    total_equity=105000,
    cash=5000,
    positions={
        "AAPL": {"market_value": 50000, "size": 333.33},
        "MSFT": {"market_value": 30000, "size": 100},
        "GOOGL": {"market_value": 20000, "size": 8}
    },
    prices={"AAPL": 150, "MSFT": 300, "GOOGL": 2500}
)

# 리스크 지표 계산
metrics = risk_manager.calculate_risk_metrics(snapshot)

print(f"95% VaR: {metrics.var_95_pct:.2%}")
print(f"현재 낙폭: {metrics.current_drawdown:.2%}")
print(f"집중도 지수: {metrics.herfindahl_index:.3f}")
print(f"종합 리스크 점수: {metrics.overall_risk_score:.1f}")
```

### 리스크 알림 시스템

```python
# 리스크 콜백 함수 정의
def risk_alert_callback(metrics, alerts):
    for alert in alerts:
        print(f"⚠️ {alert.title}: {alert.description}")
        print(f"권고사항: {', '.join(alert.recommended_actions)}")

# 긴급 상황 콜백 함수 정의
def emergency_callback(critical_alerts):
    print(f"🚨 긴급: {len(critical_alerts)}개의 심각한 리스크 알림!")
    for alert in critical_alerts:
        print(f"- {alert.title}: {alert.description}")

# 콜백 등록
risk_manager.add_risk_callback(risk_alert_callback)
risk_manager.add_emergency_callback(emergency_callback)
```

### 포트폴리오 리스크 분석

```python
from AuroraQ_Shared.risk_management import PortfolioRiskAnalyzer
import pandas as pd

# 가격 히스토리 데이터 (예시)
price_history = pd.DataFrame({
    'AAPL': np.random.normal(150, 5, 252),
    'MSFT': np.random.normal(300, 10, 252),
    'GOOGL': np.random.normal(2500, 100, 252)
}, index=pd.date_range('2023-01-01', periods=252))

# 포트폴리오 분석기 초기화
analyzer = PortfolioRiskAnalyzer(config)

# 종합 리스크 분석
analysis = analyzer.analyze_portfolio_risk(snapshot, price_history)

print("=== 포트폴리오 리스크 분석 ===")
print(f"집중도 위험 수준: {analysis['concentration_analysis']['risk_level']}")
print(f"평균 상관관계: {analysis['correlation_analysis']['avg_correlation']:.3f}")
print(f"다각화 효과: {analysis['correlation_analysis']['diversification_analysis']['diversification_benefit']:.1f}%")
```

## 📊 리스크 대시보드

```python
# 리스크 대시보드 데이터 조회
dashboard = risk_manager.get_risk_dashboard()

print("=== 리스크 대시보드 ===")
print(f"현재 VaR: {dashboard['current_metrics']['var_95_pct']:.2%}")
print(f"활성 알림: {len(dashboard['active_alerts'])}개")
print(f"리스크 예산 사용률: {dashboard['risk_budget_utilization']:.1%}")

# 포지션 축소 권고 확인
if 'position_reduction_recommendation' in dashboard:
    reduction = dashboard['position_reduction_recommendation']
    if reduction['should_reduce']:
        print(f"⚠️ 포지션 축소 권고: {reduction['reason']}")
        print(f"권장 축소 비율: {reduction['reduction_percentage']:.1%}")
```

## ⚙️ 설정 옵션

### RiskConfig 주요 파라미터

```python
config = RiskConfig(
    # VaR 설정
    var_confidence_levels=[0.95, 0.99],
    var_lookback_period=252,
    var_limit_pct=0.05,
    
    # CVaR 설정
    cvar_confidence_level=0.95,
    cvar_limit_pct=0.08,
    
    # 낙폭 관리
    max_drawdown_limit=0.15,
    drawdown_alert_threshold=0.10,
    drawdown_position_reduction=0.5,
    
    # 포지션 집중도
    max_single_position_pct=0.20,
    max_sector_concentration=0.40,
    
    # 상관관계 관리
    max_correlation_threshold=0.7,
    correlation_lookback_period=60,
    
    # 변동성 관리
    volatility_threshold_multiplier=2.0,
    volatility_lookback_period=30,
    
    # 유동성 관리
    min_liquidity_ratio=0.1,
    liquidity_buffer_pct=0.05,
    
    # 스트레스 테스트
    stress_test_scenarios=[-0.1, -0.2, -0.3]
)
```

## 🔄 백테스트와 실시간 시스템 통합

이 리스크 관리 모듈은 AuroraQ_Backtest와 AuroraQ_Production 양쪽에서 공통으로 사용할 수 있도록 설계되었습니다:

```python
# 백테스트에서 사용
from AuroraQ_Backtest.core.backtest_engine import BacktestEngine
from AuroraQ_Shared.risk_management import AdvancedRiskManager

backtest_engine = BacktestEngine()
# 백테스트 엔진에 리스크 관리자 통합

# 실시간 시스템에서 사용
from AuroraQ_Production.core.realtime_system import RealtimeSystem
from AuroraQ_Shared.risk_management import AdvancedRiskManager

realtime_system = RealtimeSystem()
# 실시간 시스템에 리스크 관리자 통합
```

## 🎯 주요 특징

1. **실시간 모니터링**: 거래 실행 시마다 자동으로 리스크 지표 업데이트
2. **동적 조정**: VaR, 낙폭, 변동성 등에 따른 실시간 포지션 사이징
3. **다층 알림 시스템**: 경고, 위험, 심각 단계별 알림
4. **긴급 대응**: 심각한 리스크 상황 시 자동 포지션 축소
5. **포괄적 분석**: VaR부터 시나리오 분석까지 종합적 리스크 평가
6. **유연한 설정**: 전략별 리스크 프로파일에 맞는 맞춤 설정

## 📈 성능 최적화

- **메모리 효율성**: 최근 데이터만 유지하여 메모리 사용량 최적화
- **계산 캐싱**: 반복 계산 결과 캐싱으로 성능 향상
- **병렬 처리**: 독립적인 계산 작업의 병렬 처리
- **점진적 업데이트**: 전체 재계산 대신 증분 업데이트

이 모듈을 통해 백테스트와 실시간 거래 모두에서 일관되고 고도화된 리스크 관리를 구현할 수 있습니다.