"""
고도화된 리스크 관리 모듈
VaR, CVaR, MDD 기반 동적 리스크 관리
"""

# VaR 계산기는 독립적으로 동작 가능
try:
    from .var_calculator import VaRCalculator
except ImportError:
    VaRCalculator = None

# 리스크 모델들
try:
    from .risk_models import RiskMetrics, RiskConfig, RiskAlert
except ImportError:
    RiskMetrics = None
    RiskConfig = None
    RiskAlert = None

# 고급 리스크 관리자 (의존성이 많음)
try:
    from .advanced_risk_manager import AdvancedRiskManager
except ImportError:
    AdvancedRiskManager = None

# 포트폴리오 리스크 분석기
try:
    from .portfolio_risk_analyzer import PortfolioRiskAnalyzer
except ImportError:
    PortfolioRiskAnalyzer = None

__all__ = [
    'AdvancedRiskManager',
    'RiskMetrics', 
    'RiskConfig',
    'RiskAlert',
    'VaRCalculator',
    'PortfolioRiskAnalyzer'
]