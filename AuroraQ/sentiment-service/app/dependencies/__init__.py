# app/dependencies/__init__.py
"""FastAPI dependencies for sentiment service"""

from .sentiment_deps import get_sentiment_analyzer, get_fusion_manager, get_cache_client

__all__ = [
    'get_sentiment_analyzer',
    'get_fusion_manager', 
    'get_cache_client'
]