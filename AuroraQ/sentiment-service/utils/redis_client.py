# utils/redis_client.py
"""Redis client configuration and utilities"""

import json
import asyncio
from typing import Optional, Any, Dict, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis
import structlog

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Global Redis connection pool
_redis_pool: Optional[Redis] = None


async def get_redis_client() -> Redis:
    """Get or create Redis client connection"""
    global _redis_pool
    
    if _redis_pool is None:
        try:
            # Create Redis client from URL
            _redis_pool = redis.from_url(
                settings.redis_url,
                db=settings.redis_db,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True
            )
            
            # Test connection
            await _redis_pool.ping()
            logger.info("Redis connection established", url=settings.redis_url)
            
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise
    
    return _redis_pool


async def close_redis_client():
    """Close Redis connection"""
    global _redis_pool
    
    if _redis_pool:
        try:
            await _redis_pool.aclose()
            _redis_pool = None
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error("Error closing Redis connection", error=str(e))


class RedisCache:
    """Redis-based caching utility"""
    
    def __init__(self, redis_client: Optional[Redis] = None, default_ttl: int = 300):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.logger = get_logger(self.__class__.__name__)
    
    async def _get_client(self) -> Redis:
        """Get Redis client"""
        if self.redis is None:
            self.redis = await get_redis_client()
        return self.redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            redis_client = await self._get_client()
            
            # Try to get the value
            cached_data = await redis_client.get(key)
            
            if cached_data is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                # Return as string if not JSON
                return cached_data
                
        except Exception as e:
            self.logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """Set value in cache"""
        try:
            redis_client = await self._get_client()
            ttl = ttl or self.default_ttl
            
            # Serialize value if needed
            if serialize:
                if isinstance(value, (dict, list)):
                    cached_value = json.dumps(value, default=str)
                else:
                    cached_value = str(value)
            else:
                cached_value = value
            
            # Set with TTL
            result = await redis_client.setex(key, ttl, cached_value)
            
            if result:
                self.logger.debug("Cache set successful", key=key, ttl=ttl)
            
            return bool(result)
            
        except Exception as e:
            self.logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            redis_client = await self._get_client()
            result = await redis_client.delete(key)
            
            if result:
                self.logger.debug("Cache delete successful", key=key)
            
            return bool(result)
            
        except Exception as e:
            self.logger.error("Cache delete failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_client = await self._get_client()
            result = await redis_client.exists(key)
            return bool(result)
            
        except Exception as e:
            self.logger.error("Cache exists check failed", key=key, error=str(e))
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            redis_client = await self._get_client()
            result = await redis_client.expire(key, ttl)
            return bool(result)
            
        except Exception as e:
            self.logger.error("Cache expire failed", key=key, error=str(e))
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for key"""
        try:
            redis_client = await self._get_client()
            ttl = await redis_client.ttl(key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            self.logger.error("Cache TTL check failed", key=key, error=str(e))
            return None
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value"""
        try:
            redis_client = await self._get_client()
            result = await redis_client.incrby(key, amount)
            return result
            
        except Exception as e:
            self.logger.error("Cache increment failed", key=key, error=str(e))
            return None
    
    async def get_multiple(self, keys: list[str]) -> Dict[str, Any]:
        """Get multiple values at once"""
        try:
            redis_client = await self._get_client()
            
            if not keys:
                return {}
            
            # Get all values
            values = await redis_client.mget(*keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value
            
            return result
            
        except Exception as e:
            self.logger.error("Cache get_multiple failed", keys=keys, error=str(e))
            return {}
    
    async def set_multiple(
        self, 
        data: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values at once"""
        try:
            redis_client = await self._get_client()
            ttl = ttl or self.default_ttl
            
            # Prepare data for mset
            serialized_data = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    serialized_data[key] = json.dumps(value, default=str)
                else:
                    serialized_data[key] = str(value)
            
            # Set all values
            result = await redis_client.mset(serialized_data)
            
            if result and ttl > 0:
                # Set TTL for all keys
                for key in data.keys():
                    await redis_client.expire(key, ttl)
            
            return bool(result)
            
        except Exception as e:
            self.logger.error("Cache set_multiple failed", error=str(e))
            return False


# Global cache instance
_global_cache: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """Get global cache instance"""
    global _global_cache
    
    if _global_cache is None:
        redis_client = await get_redis_client()
        _global_cache = RedisCache(redis_client, settings.cache_ttl)
    
    return _global_cache


def generate_cache_key(*parts: Union[str, int, float]) -> str:
    """Generate standardized cache key"""
    # Convert all parts to strings and join with colons
    key_parts = [str(part) for part in parts if part is not None]
    return ":".join(key_parts)


def generate_sentiment_cache_key(
    text_hash: str,
    model_name: str = "finbert",
    timestamp: Optional[datetime] = None
) -> str:
    """Generate cache key for sentiment analysis results"""
    if timestamp:
        # Round to hour for time-based caching
        hour_key = timestamp.strftime("%Y%m%d%H")
        return generate_cache_key("sentiment", model_name, text_hash, hour_key)
    else:
        return generate_cache_key("sentiment", model_name, text_hash)


def generate_fusion_cache_key(
    scores_hash: str,
    symbol: str,
    timestamp: Optional[datetime] = None
) -> str:
    """Generate cache key for fusion results"""
    if timestamp:
        # Round to minute for fusion caching
        minute_key = timestamp.strftime("%Y%m%d%H%M")
        return generate_cache_key("fusion", symbol, scores_hash, minute_key)
    else:
        return generate_cache_key("fusion", symbol, scores_hash)