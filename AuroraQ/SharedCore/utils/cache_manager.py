"""
모드별 최적화된 캐싱 전략
"""

import asyncio
import aioredis
import pickle
import json
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CacheMode(Enum):
    """캐시 모드"""
    AURORA_ONLY = "aurora"      # 암호화폐 + 감정분석만
    MACRO_ONLY = "macro"        # 거시경제 데이터만
    FULL = "full"              # 모든 데이터
    MINIMAL = "minimal"        # 최소 캐싱


@dataclass
class CacheConfig:
    """캐시 설정"""
    mode: CacheMode
    crypto_ttl: int = 60        # 암호화폐 데이터 (1분)
    sentiment_ttl: int = 1800   # 감정분석 (30분)
    macro_ttl: int = 3600       # 거시경제 (1시간)
    news_ttl: int = 7200        # 뉴스 (2시간)
    max_memory_mb: int = 512    # 최대 메모리 사용량
    cleanup_interval: int = 300  # 정리 주기 (5분)


class OptimizedCacheManager:
    """
    모드별 최적화된 캐시 매니저
    """
    
    def __init__(self, config: CacheConfig, redis_url: str = "redis://localhost:6379"):
        self.config = config
        self.redis_url = redis_url
        self.redis_client = None
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_mb': 0
        }
        
        # 모드별 키 패턴
        self.key_patterns = self._get_key_patterns()
        
    def _get_key_patterns(self) -> Dict[str, bool]:
        """모드별 캐시 키 패턴 설정"""
        patterns = {
            'crypto:*': False,
            'sentiment:*': False,
            'macro:*': False,
            'news:*': False,
            'events:*': False
        }
        
        if self.config.mode in [CacheMode.AURORA_ONLY, CacheMode.FULL]:
            patterns['crypto:*'] = True
            patterns['sentiment:*'] = True
            patterns['news:*'] = True
            
        if self.config.mode in [CacheMode.MACRO_ONLY, CacheMode.FULL]:
            patterns['macro:*'] = True
            patterns['events:*'] = True
            
        return patterns
    
    async def connect(self):
        """Redis 연결"""
        try:
            self.redis_client = await aioredis.create_redis_pool(self.redis_url)
            logger.info(f"Connected to Redis with {self.config.mode.value} cache mode")
            
            # 기존 캐시 정리 (모드에 맞지 않는 데이터)
            await self._cleanup_incompatible_cache()
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using memory cache only.")
            self.redis_client = None
    
    async def get(self, key: str, default: Any = None) -> Any:
        """캐시에서 데이터 조회"""
        # 키 패턴 검증
        if not self._is_key_allowed(key):
            logger.debug(f"Key {key} not allowed in {self.config.mode.value} mode")
            return default
            
        try:
            # Redis 우선 시도
            if self.redis_client:
                data = await self.redis_client.get(key)
                if data:
                    self.cache_stats['hits'] += 1
                    return self._deserialize(data)
            
            # 메모리 캐시 fallback
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry['expires'] > datetime.now():
                    self.cache_stats['hits'] += 1
                    return entry['data']
                else:
                    # 만료된 데이터 제거
                    del self.memory_cache[key]
            
            self.cache_stats['misses'] += 1
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    async def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 데이터 저장"""
        # 키 패턴 검증
        if not self._is_key_allowed(key):
            return False
            
        # TTL 결정
        if ttl is None:
            ttl = self._get_default_ttl(key)
            
        try:
            # Redis 저장
            if self.redis_client:
                serialized = self._serialize(data)
                await self.redis_client.setex(key, ttl, serialized)
            
            # 메모리 캐시 저장 (Redis 실패시 backup)
            self.memory_cache[key] = {
                'data': data,
                'expires': datetime.now() + timedelta(seconds=ttl)
            }
            
            # 메모리 사용량 체크
            await self._check_memory_usage()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """캐시에서 데이터 삭제"""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
            
            if key in self.memory_cache:
                del self.memory_cache[key]
                
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str):
        """패턴에 맞는 캐시 삭제"""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            # 메모리 캐시에서도 삭제
            keys_to_delete = []
            for key in self.memory_cache:
                if self._match_pattern(key, pattern):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.memory_cache[key]
                
            logger.info(f"Cleared cache pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
    
    def _is_key_allowed(self, key: str) -> bool:
        """키가 현재 모드에서 허용되는지 확인"""
        for pattern, allowed in self.key_patterns.items():
            if self._match_pattern(key, pattern):
                return allowed
        return False
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """패턴 매칭"""
        if pattern.endswith('*'):
            return key.startswith(pattern[:-1])
        return key == pattern
    
    def _get_default_ttl(self, key: str) -> int:
        """키별 기본 TTL 반환"""
        if key.startswith('crypto:'):
            return self.config.crypto_ttl
        elif key.startswith('sentiment:'):
            return self.config.sentiment_ttl
        elif key.startswith('macro:'):
            return self.config.macro_ttl
        elif key.startswith('news:'):
            return self.config.news_ttl
        else:
            return 300  # 기본 5분
    
    def _serialize(self, data: Any) -> bytes:
        """데이터 직렬화"""
        try:
            # JSON 시도 (빠름)
            return json.dumps(data, default=str).encode()
        except (TypeError, ValueError):
            # Pickle fallback (느리지만 모든 타입 지원)
            return pickle.dumps(data)
    
    def _deserialize(self, data: bytes) -> Any:
        """데이터 역직렬화"""
        try:
            # JSON 시도
            return json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Pickle fallback
            return pickle.loads(data)
    
    async def _check_memory_usage(self):
        """메모리 사용량 체크 및 정리"""
        # 대략적인 메모리 사용량 계산
        estimated_mb = len(str(self.memory_cache)) / 1024 / 1024
        self.cache_stats['memory_usage_mb'] = estimated_mb
        
        if estimated_mb > self.config.max_memory_mb:
            # LRU 방식으로 오래된 항목 제거
            now = datetime.now()
            items_to_remove = []
            
            for key, entry in self.memory_cache.items():
                if entry['expires'] < now:
                    items_to_remove.append(key)
            
            # 만료된 항목 제거
            for key in items_to_remove:
                del self.memory_cache[key]
                self.cache_stats['evictions'] += 1
            
            logger.info(f"Memory cache cleanup: removed {len(items_to_remove)} expired items")
    
    async def _cleanup_incompatible_cache(self):
        """현재 모드와 호환되지 않는 캐시 정리"""
        if not self.redis_client:
            return
            
        try:
            # 모든 키 패턴에 대해 체크
            for pattern, allowed in self.key_patterns.items():
                if not allowed:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
                        logger.info(f"Cleaned up incompatible cache pattern: {pattern}")
                        
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'hit_rate_percent': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'cache_mode': self.config.mode.value
        }
    
    async def close(self):
        """연결 종료"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
            logger.info("Cache manager closed")


def create_cache_manager(mode: str, **kwargs) -> OptimizedCacheManager:
    """모드별 캐시 매니저 생성"""
    cache_mode = CacheMode(mode)
    config = CacheConfig(mode=cache_mode, **kwargs)
    return OptimizedCacheManager(config)