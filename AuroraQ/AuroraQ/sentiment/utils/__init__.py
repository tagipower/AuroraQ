#!/usr/bin/env python3
"""
AuroraQ Sentiment Utils Module
뉴스 수집 및 처리를 위한 유틸리티 모듈
"""

# 중요도 점수화 시스템
try:
    from .news_importance_scorer import NewsImportanceScorer, calculate_news_importance
except ImportError:
    NewsImportanceScorer = None
    calculate_news_importance = None

# 중복 제거 시스템들
try:
    from .news_deduplicator import NewsDeduplicator, deduplicate_news_items
except ImportError:
    NewsDeduplicator = None
    deduplicate_news_items = None

try:
    from .high_performance_deduplicator import (
        HighPerformanceDeduplicator, 
        get_high_performance_deduplicator,
        deduplicate_news_optimized
    )
except ImportError:
    HighPerformanceDeduplicator = None
    get_high_performance_deduplicator = None
    deduplicate_news_optimized = None

# 사전 필터링 시스템
try:
    from .news_prefilter import NewsPreFilter, FilterDecision
except ImportError:
    NewsPreFilter = None
    FilterDecision = None

__all__ = [
    'NewsImportanceScorer',
    'calculate_news_importance',
    'NewsDeduplicator', 
    'deduplicate_news_items',
    'HighPerformanceDeduplicator',
    'get_high_performance_deduplicator',
    'deduplicate_news_optimized',
    'NewsPreFilter',
    'FilterDecision'
]