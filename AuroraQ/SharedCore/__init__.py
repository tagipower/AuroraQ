"""
SharedCore - AuroraQ & MacroQ 공통 데이터 및 인프라 레이어

이 모듈은 두 AI Agent가 읽기 전용으로 참조하는 공통 데이터와 서비스를 제공합니다.
- 통합 데이터 레이어 (market_data, sentiment)
- 리스크 관리 시스템
- 모니터링 및 알림
- 공통 유틸리티
"""

__version__ = "2.0.0"
__author__ = "QuantumAI Team"

from .data_layer.unified_data_provider import UnifiedDataProvider
from .sentiment_engine.sentiment_aggregator import SentimentAggregator

__all__ = [
    'UnifiedDataProvider',
    'SentimentAggregator'
]