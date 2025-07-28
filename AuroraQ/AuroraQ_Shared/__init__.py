"""
AuroraQ 공유 모듈

백테스트와 실시간 거래 시스템 간 공통 모듈
"""

# 포지션 관리
from .position_management import (
    UnifiedPositionManager,
    Position,
    Trade,
    OrderSignal,
    PositionSide,
    OrderSide,
    TradeStatus
)

# 리스크 관리
from .risk_management import (
    AdvancedRiskManager,
    VaRCalculator,
    RiskMetrics,
    RiskConfig,
    RiskAlert,
    PortfolioRiskAnalyzer
)

# 시스템 통합
from .integration import (
    BacktestRiskIntegration,
    ProductionRiskIntegration,
    create_risk_aware_backtest,
    create_production_risk_integration,
    quick_risk_backtest,
    start_risk_aware_trading,
    create_calibrated_backtest
)

# 보정 시스템
from .calibration import (
    CalibrationManager,
    CalibrationConfig,
    CalibrationResult,
    ExecutionAnalyzer,
    ExecutionMetrics,
    MarketConditionDetector,
    RealTradeMonitor
)

__version__ = "1.0.0"

__all__ = [
    # 포지션 관리
    'UnifiedPositionManager',
    'Position',
    'Trade', 
    'OrderSignal',
    'PositionSide',
    'OrderSide',
    'TradeStatus',
    
    # 리스크 관리
    'AdvancedRiskManager',
    'VaRCalculator',
    'RiskMetrics',
    'RiskConfig', 
    'RiskAlert',
    'PortfolioRiskAnalyzer',
    
    # 시스템 통합
    'BacktestRiskIntegration',
    'ProductionRiskIntegration',
    'create_risk_aware_backtest',
    'create_production_risk_integration',
    'quick_risk_backtest',
    'start_risk_aware_trading',
    'create_calibrated_backtest',
    
    # 보정 시스템
    'CalibrationManager',
    'CalibrationConfig',
    'CalibrationResult',
    'ExecutionAnalyzer',
    'ExecutionMetrics',
    'MarketConditionDetector',
    'RealTradeMonitor'
]