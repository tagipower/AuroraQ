"""
AuroraQ 시스템 통합 모듈

백테스트와 실시간 거래 시스템의 리스크 관리 및 포지션 관리 통합
실시간 보정 시스템과 하이브리드 거래 시스템
"""

# 백테스트 통합 시스템
from .backtest_integration import (
    BacktestIntegration,
    BacktestRiskIntegration,
    create_simple_backtest,
    create_risk_aware_backtest,
    create_auto_calibrated_backtest,
    create_synchronized_backtest_environment,
    sync_backtest_with_realtime_parameters,
    run_comparative_analysis,
    quick_risk_backtest,
    create_calibrated_backtest
)

# 실시간 시스템 통합
from .production_integration import (
    ProductionIntegration,
    ProductionRiskIntegration,
    create_simple_production,
    create_production_risk_integration,
    start_risk_aware_trading
)

# 실시간 보정 시스템
from .realtime_calibration_system import (
    RealtimeCalibrationSystem,
    RealtimeCalibrationConfig,
    RealtimeCalibrationState
)

# 실시간 하이브리드 시스템
from .realtime_hybrid_system import (
    RealtimeHybridSystem,
    RealtimeSystemConfig,
    TradingSignal,
    SystemState
)

__all__ = [
    # 백테스트 통합
    'BacktestIntegration',
    'BacktestRiskIntegration',
    'create_simple_backtest',
    'create_risk_aware_backtest',
    'create_auto_calibrated_backtest',
    'create_synchronized_backtest_environment',
    'sync_backtest_with_realtime_parameters',
    'run_comparative_analysis',
    'quick_risk_backtest',
    'create_calibrated_backtest',
    
    # 실시간 통합
    'ProductionIntegration',
    'ProductionRiskIntegration',
    'create_simple_production',
    'create_production_risk_integration',
    'start_risk_aware_trading',
    
    # 실시간 보정 시스템
    'RealtimeCalibrationSystem',
    'RealtimeCalibrationConfig',
    'RealtimeCalibrationState',
    
    # 실시간 하이브리드 시스템
    'RealtimeHybridSystem',
    'RealtimeSystemConfig',
    'TradingSignal',
    'SystemState'
]