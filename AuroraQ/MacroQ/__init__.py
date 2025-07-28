"""
MacroQ - Temporal Fusion Transformer 기반 중장기 포트폴리오 관리 시스템

다자산 지원 (주식, ETF, 채권, 암호화폐)
TFT 기반 멀티호라이즌 예측 (1주, 1개월, 3개월)
리스크 패리티 기반 포트폴리오 최적화
"""

__version__ = "1.0.0"
__author__ = "QuantumAI Team"

from .core.tft_engine.lightweight_tft import LightweightTFT
from .portfolio.optimizer import PortfolioOptimizer

__all__ = ['LightweightTFT', 'PortfolioOptimizer']