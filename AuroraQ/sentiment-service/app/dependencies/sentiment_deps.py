# app/dependencies/sentiment_deps.py
"""FastAPI dependencies for sentiment analysis components"""

import sys
import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from fastapi import Depends, HTTPException
import structlog

# Add SharedCore to path for import
project_root = Path(__file__).parent.parent.parent.parent
shared_core_path = project_root / "SharedCore"
sys.path.insert(0, str(shared_core_path))

try:
    from sentiment_engine.analyzers.finbert_analyzer import FinBERTAnalyzer, get_finbert_analyzer
    from sentiment_engine.fusion.sentiment_fusion_manager import SentimentFusionManager, get_fusion_manager
    from sentiment_engine.routing.sentiment_router import SentimentRouter, get_router
except ImportError as e:
    # Fallback imports or create minimal implementations
    print(f"Warning: Could not import SharedCore modules: {e}")
    
    # Create stub classes if SharedCore is not available
    class FinBERTAnalyzer:
        def __init__(self):
            self._initialized = False
        
        async def initialize(self):
            self._initialized = True
        
        async def analyze(self, text):
            return 0.5
        
        async def analyze_detailed(self, text):
            from collections import namedtuple
            Result = namedtuple('Result', ['sentiment_score', 'label', 'confidence', 'keywords', 'scenario_tag'])
            Label = namedtuple('Label', ['value'])
            return Result(0.5, Label('neutral'), 0.8, [], '')
        
        async def close(self):
            pass
    
    class SentimentFusionManager:
        def __init__(self):
            self._initialized = False
        
        async def initialize(self):
            self._initialized = True
        
        async def fuse(self, scores, symbol="BTC", timestamp=None):
            if not scores:
                return 0.5
            return sum(scores.values()) / len(scores)
        
        def get_statistics(self, symbol=None):
            return {"count": 0, "average_confidence": 0.8}
        
        async def close(self):
            pass
    
    class SentimentRouter:
        def __init__(self, mode="live"):
            self.mode = mode
            self._initialized = False
        
        async def initialize(self):
            self._initialized = True
        
        async def get_score(self, news_text=None, timestamp=None):
            return 0.5
        
        def get_mode(self):
            return self.mode
        
        async def close(self):
            pass
    
    async def get_finbert_analyzer():
        analyzer = FinBERTAnalyzer()
        await analyzer.initialize()
        return analyzer
    
    async def get_fusion_manager():
        manager = SentimentFusionManager()
        await manager.initialize()
        return manager
    
    async def get_router(mode="live", csv_path=None):
        router = SentimentRouter(mode)
        await router.initialize()
        return router

from utils.redis_client import get_cache
from config.settings import settings

logger = structlog.get_logger(__name__)

# Global instances to maintain state
_sentiment_analyzer: Optional[FinBERTAnalyzer] = None
_fusion_manager: Optional[SentimentFusionManager] = None
_sentiment_router: Optional[SentimentRouter] = None


@lru_cache()
async def get_sentiment_analyzer() -> FinBERTAnalyzer:
    """Get FinBERT sentiment analyzer instance"""
    global _sentiment_analyzer
    
    if _sentiment_analyzer is None:
        try:
            logger.info("Initializing FinBERT analyzer")
            _sentiment_analyzer = await get_finbert_analyzer()
            logger.info("FinBERT analyzer initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize FinBERT analyzer", error=str(e))
            raise HTTPException(
                status_code=503,
                detail="Sentiment analyzer service unavailable"
            )
    
    return _sentiment_analyzer


@lru_cache()
async def get_fusion_manager() -> SentimentFusionManager:
    """Get sentiment fusion manager instance"""
    global _fusion_manager
    
    if _fusion_manager is None:
        try:
            logger.info("Initializing sentiment fusion manager")
            _fusion_manager = await get_fusion_manager()
            logger.info("Sentiment fusion manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize fusion manager", error=str(e))
            raise HTTPException(
                status_code=503,
                detail="Fusion manager service unavailable"
            )
    
    return _fusion_manager


@lru_cache()
async def get_sentiment_router(mode: str = "live") -> SentimentRouter:
    """Get sentiment router instance"""
    global _sentiment_router
    
    if _sentiment_router is None or _sentiment_router.get_mode() != mode:
        try:
            logger.info("Initializing sentiment router", mode=mode)
            _sentiment_router = await get_router(mode)
            logger.info("Sentiment router initialized successfully", mode=mode)
            
        except Exception as e:
            logger.error("Failed to initialize sentiment router", mode=mode, error=str(e))
            raise HTTPException(
                status_code=503,
                detail="Sentiment router service unavailable"
            )
    
    return _sentiment_router


async def get_cache_client():
    """Get Redis cache client dependency"""
    try:
        cache = await get_cache()
        return cache
    except Exception as e:
        logger.error("Failed to get cache client", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Cache service unavailable"
        )


async def cleanup_dependencies():
    """Clean up all global dependencies"""
    global _sentiment_analyzer, _fusion_manager, _sentiment_router
    
    try:
        if _sentiment_analyzer:
            await _sentiment_analyzer.close()
            _sentiment_analyzer = None
        
        if _fusion_manager:
            await _fusion_manager.close()
            _fusion_manager = None
        
        if _sentiment_router:
            await _sentiment_router.close()
            _sentiment_router = None
        
        logger.info("Dependencies cleanup completed")
        
    except Exception as e:
        logger.error("Error during dependencies cleanup", error=str(e))


# Dependency functions for FastAPI
def analyzer_dependency() -> FinBERTAnalyzer:
    """FastAPI dependency for sentiment analyzer"""
    return Depends(get_sentiment_analyzer)


def fusion_dependency() -> SentimentFusionManager:
    """FastAPI dependency for fusion manager"""
    return Depends(get_fusion_manager)


def cache_dependency():
    """FastAPI dependency for cache client"""
    return Depends(get_cache_client)