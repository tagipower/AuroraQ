# SharedCore/sentiment_engine/sentiment_client.py
"""
Client for connecting to the independent sentiment service
Replaces direct usage of sentiment modules with REST API calls
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import aiohttp
import json
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class SentimentServiceConfig:
    """Configuration for sentiment service client"""
    base_url: str = "http://localhost:8000"
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes


class SentimentServiceClient:
    """
    High-performance client for the independent sentiment service
    Provides the same interface as the original sentiment modules but uses REST API
    """
    
    def __init__(self, config: Optional[SentimentServiceConfig] = None):
        self.config = config or SentimentServiceConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._local_cache: Dict[str, tuple[Any, float]] = {}  # Simple memory cache
        self.logger = logger
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
        return self._session
    
    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _cache_key(self, endpoint: str, **kwargs) -> str:
        """Generate cache key"""
        key_data = f"{endpoint}:{json.dumps(kwargs, sort_keys=True)}"
        return key_data
    
    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get from local cache"""
        if not self.config.enable_caching:
            return None
        
        if cache_key in self._local_cache:
            data, timestamp = self._local_cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                return data
            else:
                del self._local_cache[cache_key]
        return None
    
    def _set_cached(self, cache_key: str, data: Any):
        """Set local cache"""
        if self.config.enable_caching:
            self._local_cache[cache_key] = (data, time.time())
            
            # Simple cache cleanup (keep last 1000 entries)
            if len(self._local_cache) > 1000:
                oldest_keys = sorted(
                    self._local_cache.keys(),
                    key=lambda k: self._local_cache[k][1]
                )[:100]  # Remove oldest 100 entries
                for key in oldest_keys:
                    del self._local_cache[key]
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        
        # Check cache first
        if use_cache and method.upper() == 'GET':
            cache_key = self._cache_key(endpoint, **(data or {}))
            cached_result = self._get_cached(cache_key)
            if cached_result:
                return cached_result
        
        session = await self._get_session()
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with session.request(method, url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Cache successful GET requests
                        if use_cache and method.upper() == 'GET':
                            cache_key = self._cache_key(endpoint, **(data or {}))
                            self._set_cached(cache_key, result)
                        
                        return result
                    else:
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=error_text
                        )
                        
            except Exception as e:
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}"
                )
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        raise Exception("Max retry attempts exceeded")
    
    async def analyze_sentiment(
        self, 
        text: str, 
        symbol: str = "CRYPTO", 
        include_detailed: bool = False
    ) -> float:
        """
        Analyze sentiment of single text - compatible with original analyzer interface
        Returns: sentiment score (0.0 to 1.0)
        """
        try:
            result = await self._make_request(
                'POST', 
                '/api/v1/sentiment/analyze',
                {
                    'text': text,
                    'symbol': symbol,
                    'include_detailed': include_detailed
                }
            )
            return result['sentiment_score']
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return 0.5  # Neutral fallback
    
    async def analyze_sentiment_detailed(
        self, 
        text: str, 
        symbol: str = "CRYPTO"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment with detailed results
        Returns: full sentiment analysis result
        """
        try:
            result = await self._make_request(
                'POST',
                '/api/v1/sentiment/analyze',
                {
                    'text': text,
                    'symbol': symbol, 
                    'include_detailed': True
                }
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Detailed sentiment analysis failed: {e}")
            return {
                'sentiment_score': 0.5,
                'label': 'neutral',
                'confidence': 0.0,
                'keywords': [],
                'scenario_tag': '',
                'error': str(e)
            }
    
    async def analyze_batch_sentiment(
        self, 
        texts: List[str], 
        symbol: str = "CRYPTO",
        include_detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze multiple texts in batch - optimized for high throughput
        Returns: batch analysis results
        """
        try:
            result = await self._make_request(
                'POST',
                '/api/v1/sentiment/analyze/batch',
                {
                    'texts': texts,
                    'symbol': symbol,
                    'include_detailed': include_detailed
                }
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Batch sentiment analysis failed: {e}")
            # Return fallback results
            return {
                'results': [
                    {
                        'sentiment_score': 0.5,
                        'label': 'neutral',
                        'confidence': 0.0,
                        'error': str(e)
                    } for _ in texts
                ],
                'total_count': len(texts),
                'average_score': 0.5,
                'processing_time': 0.0
            }
    
    async def fuse_sentiment_scores(
        self,
        sentiment_scores: Dict[str, float],
        symbol: str = "BTCUSDT", 
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Fuse multiple sentiment scores - compatible with fusion manager interface
        Returns: fused sentiment analysis result
        """
        try:
            request_data = {
                'sentiment_scores': sentiment_scores,
                'symbol': symbol
            }
            
            if timestamp:
                request_data['timestamp'] = int(timestamp.timestamp())
            
            result = await self._make_request(
                'POST',
                '/api/v1/fusion/fuse',
                request_data
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Sentiment fusion failed: {e}")
            # Fallback: simple average
            if sentiment_scores:
                avg_score = sum(sentiment_scores.values()) / len(sentiment_scores)
            else:
                avg_score = 0.5
                
            return {
                'fused_score': avg_score,
                'confidence': 0.0,
                'trend': 'neutral',
                'volatility': 0.0,
                'raw_scores': sentiment_scores,
                'weights_used': {k: 1.0/len(sentiment_scores) for k in sentiment_scores},
                'sources_count': len(sentiment_scores),
                'error': str(e)
            }
    
    async def get_fusion_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get fusion statistics for symbol"""
        try:
            result = await self._make_request(
                'GET',
                f'/api/v1/fusion/statistics/{symbol}'
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get fusion statistics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            result = await self._make_request('GET', '/health', use_cache=False)
            return result
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}


# Compatibility layer - provides same interface as original modules
class SentimentAnalyzerProxy:
    """Proxy that provides the same interface as the original FinBERT analyzer"""
    
    def __init__(self, client: SentimentServiceClient):
        self.client = client
        self._initialized = True
    
    async def analyze(self, text: Union[str, dict]) -> float:
        """Analyze sentiment - compatible with original interface"""
        if isinstance(text, dict):
            text_content = text.get('content', '') or text.get('title', '')
        else:
            text_content = str(text)
            
        return await self.client.analyze_sentiment(text_content)
    
    async def analyze_detailed(self, text: Union[str, dict]) -> Any:
        """Analyze with detailed results"""
        if isinstance(text, dict):
            text_content = text.get('content', '') or text.get('title', '')
        else:
            text_content = str(text)
            
        result = await self.client.analyze_sentiment_detailed(text_content)
        
        # Return compatible object
        class DetailedResult:
            def __init__(self, data):
                self.sentiment_score = data['sentiment_score']
                self.confidence = data['confidence']
                self.keywords = data.get('keywords', [])
                self.scenario_tag = data.get('scenario_tag', '')
                self.label = type('Label', (), {'value': data['label']})()
                
        return DetailedResult(result)
    
    async def close(self):
        """Close resources"""
        await self.client.close()


class SentimentFusionManagerProxy:
    """Proxy that provides the same interface as the original fusion manager"""
    
    def __init__(self, client: SentimentServiceClient):
        self.client = client
        self._initialized = True
    
    async def fuse(
        self, 
        sentiment_scores: Dict[str, float],
        symbol: str = "BTCUSDT",
        timestamp: Optional[datetime] = None
    ) -> float:
        """Fuse sentiment scores - compatible with original interface"""
        result = await self.client.fuse_sentiment_scores(sentiment_scores, symbol, timestamp)
        return result['fused_score']
    
    def get_statistics(self, symbol: str = None) -> Dict[str, Any]:
        """Get statistics - note: this becomes async in the proxy"""
        # This is a limitation - original was sync, service is async
        # In practice, would need to be called differently
        return {'note': 'Use async get_statistics_async method'}
    
    async def get_statistics_async(self, symbol: str = None) -> Dict[str, Any]:
        """Async version of get_statistics"""
        if symbol:
            return await self.client.get_fusion_statistics(symbol)
        return {}
    
    async def close(self):
        """Close resources"""
        await self.client.close()


# Global instances for backward compatibility
_global_client: Optional[SentimentServiceClient] = None
_global_analyzer_proxy: Optional[SentimentAnalyzerProxy] = None
_global_fusion_proxy: Optional[SentimentFusionManagerProxy] = None


@lru_cache()
def get_sentiment_service_client(service_url: str = "http://localhost:8000") -> SentimentServiceClient:
    """Get global sentiment service client"""
    global _global_client
    
    if _global_client is None:
        config = SentimentServiceConfig(base_url=service_url)
        _global_client = SentimentServiceClient(config)
    
    return _global_client


async def get_analyzer_proxy() -> SentimentAnalyzerProxy:
    """Get sentiment analyzer proxy - drop-in replacement for FinBERT analyzer"""
    global _global_analyzer_proxy
    
    if _global_analyzer_proxy is None:
        client = get_sentiment_service_client()
        _global_analyzer_proxy = SentimentAnalyzerProxy(client)
    
    return _global_analyzer_proxy


async def get_fusion_manager_proxy() -> SentimentFusionManagerProxy:
    """Get fusion manager proxy - drop-in replacement for fusion manager"""
    global _global_fusion_proxy
    
    if _global_fusion_proxy is None:
        client = get_sentiment_service_client()
        _global_fusion_proxy = SentimentFusionManagerProxy(client)
    
    return _global_fusion_proxy


async def cleanup_service_connections():
    """Clean up all service connections"""
    global _global_client, _global_analyzer_proxy, _global_fusion_proxy
    
    if _global_client:
        await _global_client.close()
        _global_client = None
    
    _global_analyzer_proxy = None
    _global_fusion_proxy = None


# Example usage and migration guide
if __name__ == "__main__":
    import asyncio
    
    async def test_client():
        """Test the sentiment service client"""
        
        # Initialize client
        config = SentimentServiceConfig(
            base_url="http://localhost:8000",
            enable_caching=True,
            cache_ttl=300
        )
        client = SentimentServiceClient(config)
        
        try:
            # Test health check
            health = await client.health_check()
            print(f"Service health: {health['status']}")
            
            # Test single sentiment analysis
            sentiment = await client.analyze_sentiment(
                "Bitcoin price surges as institutional adoption increases",
                symbol="BTC"
            )
            print(f"Sentiment score: {sentiment}")
            
            # Test detailed analysis
            detailed = await client.analyze_sentiment_detailed(
                "Federal Reserve hints at rate cuts amid economic uncertainty",
                symbol="SPY"
            )
            print(f"Detailed analysis: {detailed}")
            
            # Test batch analysis
            texts = [
                "Cryptocurrency market shows strong bullish momentum",
                "Regulatory concerns weigh on digital asset prices",
                "Institutional investors increase crypto allocations"
            ]
            
            batch_result = await client.analyze_batch_sentiment(texts, symbol="CRYPTO")
            print(f"Batch analysis - Average sentiment: {batch_result['average_score']}")
            
            # Test sentiment fusion
            scores = {
                "news": 0.8,
                "social": 0.6,
                "technical": 0.7,
                "historical": 0.65
            }
            
            fusion_result = await client.fuse_sentiment_scores(scores, symbol="BTCUSDT")
            print(f"Fused sentiment: {fusion_result['fused_score']} (confidence: {fusion_result['confidence']})")
            
        finally:
            await client.close()
    
    # Run test
    asyncio.run(test_client())