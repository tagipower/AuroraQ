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
    FusedSentiment
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
    'FusedSentiment'
]