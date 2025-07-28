# SharedCore/sentiment_engine/fusion/__init__.py

"""
다중 소스 감정 점수 융합 관리자 모듈
"""

from .sentiment_fusion_manager import (
    SentimentFusionManager,
    FusionConfig,
    FusedSentiment,
    get_fusion_manager
)

__all__ = [
    'SentimentFusionManager',
    'FusionConfig',
    'FusedSentiment',
    'get_fusion_manager'
]