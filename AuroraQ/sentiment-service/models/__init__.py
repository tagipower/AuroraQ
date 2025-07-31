# models/__init__.py
"""Pydantic models for API requests and responses"""

from .sentiment_models import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    FusionRequest,
    FusionResponse,
    HealthResponse,
    ErrorResponse,
    SentimentLabel,
    NewsArticle,
    SentimentResult,
    FusedSentiment,
    ServiceStats,
    ModelInfo
)

# Import advanced components
from .keyword_scorer import KeywordScorer
from .advanced_keyword_scorer import (
    AdvancedKeywordScorer,
    MultiModalSentiment,
    AdvancedFeatures,
    EmotionalState,
    MarketRegime
)

__all__ = [
    'SentimentRequest',
    'SentimentResponse', 
    'BatchSentimentRequest',
    'BatchSentimentResponse',
    'FusionRequest',
    'FusionResponse',
    'HealthResponse',
    'ErrorResponse',
    'SentimentLabel',
    'NewsArticle',
    'SentimentResult',
    'FusedSentiment',
    'ServiceStats',
    'ModelInfo',
    # Advanced components
    'KeywordScorer',
    'AdvancedKeywordScorer',
    'MultiModalSentiment',
    'AdvancedFeatures',
    'EmotionalState',
    'MarketRegime'
]