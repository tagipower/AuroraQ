"""
AuroraQ - AI 기반 단기 트레이딩 시스템

PPO 강화학습 + Rule 기반 전략을 통한 암호화폐 단기매매
SharedCore와 연동하여 감정 분석 및 거시경제 이벤트 활용
"""

__version__ = "2.0.0"
__author__ = "QuantumAI Team"

from .backtest.core.backtest_engine import BacktestEngine
from .production.core.realtime_system import RealtimeSystem

__all__ = ['BacktestEngine', 'RealtimeSystem']