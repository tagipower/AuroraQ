#!/usr/bin/env python3
"""
Content Cache Manager for AuroraQ Sentiment Service
5분 TTL 캐싱 및 원문 자동 폐기 시스템 (메타데이터만 보관)
"""

import asyncio
import redis.asyncio as redis
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContentMetadata:
    """원문 폐기 후 보관할 메타데이터"""
    content_hash: str
    title: str
    source: str
    published_at: datetime
    url: str
    category: str
    symbol: Optional[str]
    sentiment_score: Optional[float] = None
    confidence: Optional[float] = None
    keywords: List[str] = None
    entities: List[str] = None
    relevance_score: float = 0.5
    processing_timestamp: datetime = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.entities is None:
            self.entities = []
        if self.processing_timestamp is None:
            self.processing_timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        data = asdict(self)
        # datetime 객체를 문자열로 변환
        data['published_at'] = self.published_at.isoformat()
        data['processing_timestamp'] = self.processing_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentMetadata':
        """딕셔너리에서 복원"""
        # 문자열을 datetime 객체로 변환
        data['published_at'] = datetime.fromisoformat(data['published_at'])
        data['processing_timestamp'] = datetime.fromisoformat(data['processing_timestamp'])
        return cls(**data)

class ContentCacheManager:
    """컨텐츠 캐시 관리자"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        초기화
        
        Args:
            redis_url: Redis 연결 URL
        """
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        
        # 캐시 설정
        self.raw_content_ttl = 300  # 5분 (원문 캐시)
        self.metadata_ttl = 86400 * 7  # 7일 (메타데이터)
        self.processing_ttl = 3600  # 1시간 (처리 중 마킹)
        
        # 키 프리픽스
        self.raw_content_prefix = "raw_content:"
        self.metadata_prefix = "metadata:"
        self.processing_prefix = "processing:"
        self.stats_prefix = "cache_stats:"
        
        # 통계
        self.stats = {
            "total_cached": 0,
            "total_expired": 0,
            "metadata_preserved": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.redis = await redis.from_url(self.redis_url)
        await self._load_stats()
        logger.info("Content cache manager initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.redis:
            await self._save_stats()
            await self.redis.close()
            logger.info("Content cache manager closed")
    
    def _generate_content_hash(self, content: str, url: str) -> str:
        """컨텐츠 해시 생성"""
        hash_input = f"{content[:500]}{url}"  # 내용 일부 + URL
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    async def cache_raw_content(self,
                              content: str,
                              title: str,
                              url: str,
                              source: str,
                              published_at: datetime,
                              category: str = "general",
                              symbol: Optional[str] = None,
                              additional_metadata: Optional[Dict[str, Any]] = None) -> str:
        """원문 임시 캐싱 (5분 TTL)"""
        
        # 컨텐츠 해시 생성
        content_hash = self._generate_content_hash(content, url)
        
        # 원문 캐시 키
        raw_key = f"{self.raw_content_prefix}{content_hash}"
        
        # 원문 데이터 구조
        raw_data = {
            "content": content,
            "title": title,
            "url": url,
            "source": source,
            "published_at": published_at.isoformat(),
            "category": category,
            "symbol": symbol,
            "cached_at": datetime.now().isoformat(),
            "content_hash": content_hash
        }
        
        # 추가 메타데이터 병합
        if additional_metadata:
            raw_data.update(additional_metadata)
        
        try:
            # Redis에 원문 저장 (5분 TTL)
            await self.redis.setex(
                raw_key,
                self.raw_content_ttl,
                json.dumps(raw_data, default=str)
            )
            
            # 통계 업데이트
            self.stats["total_cached"] += 1
            
            logger.debug(f"Raw content cached: {content_hash} (TTL: {self.raw_content_ttl}s)")
            return content_hash
            
        except Exception as e:
            logger.error(f"Failed to cache raw content: {e}")
            return content_hash
    
    async def get_raw_content(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """원문 조회 (5분 내에만 가능)"""
        
        raw_key = f"{self.raw_content_prefix}{content_hash}"
        
        try:
            cached_data = await self.redis.get(raw_key)
            
            if cached_data:
                self.stats["cache_hits"] += 1
                logger.debug(f"Raw content cache hit: {content_hash}")
                return json.loads(cached_data)
            else:
                self.stats["cache_misses"] += 1
                logger.debug(f"Raw content cache miss: {content_hash}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get raw content: {e}")
            return None
    
    async def preserve_metadata(self,
                              content_hash: str,
                              metadata: ContentMetadata) -> bool:
        """메타데이터 영구 보관 (원문 폐기 후)"""
        
        metadata_key = f"{self.metadata_prefix}{content_hash}"
        
        try:
            # 메타데이터를 JSON으로 저장 (7일 TTL)
            await self.redis.setex(
                metadata_key,
                self.metadata_ttl,
                json.dumps(metadata.to_dict())
            )
            
            self.stats["metadata_preserved"] += 1
            
            logger.debug(f"Metadata preserved: {content_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to preserve metadata: {e}")
            return False
    
    async def get_metadata(self, content_hash: str) -> Optional[ContentMetadata]:
        """메타데이터 조회"""
        
        metadata_key = f"{self.metadata_prefix}{content_hash}"
        
        try:
            cached_data = await self.redis.get(metadata_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return ContentMetadata.from_dict(data)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return None
    
    async def process_and_preserve(self,
                                 content_hash: str,
                                 sentiment_score: float,
                                 confidence: float,
                                 keywords: List[str],
                                 entities: List[str] = None) -> bool:
        """감정 분석 후 메타데이터 보관 및 원문 폐기"""
        
        # 원문 조회
        raw_data = await self.get_raw_content(content_hash)
        
        if not raw_data:
            logger.warning(f"Raw content not found for processing: {content_hash}")
            return False
        
        try:
            # 메타데이터 생성
            metadata = ContentMetadata(
                content_hash=content_hash,
                title=raw_data["title"],
                source=raw_data["source"],
                published_at=datetime.fromisoformat(raw_data["published_at"]),
                url=raw_data["url"],
                category=raw_data["category"],
                symbol=raw_data.get("symbol"),
                sentiment_score=sentiment_score,
                confidence=confidence,
                keywords=keywords,
                entities=entities or [],
                relevance_score=raw_data.get("relevance_score", 0.5)
            )
            
            # 메타데이터 보관
            success = await self.preserve_metadata(content_hash, metadata)
            
            if success:
                # 원문 즉시 삭제 (TTL 만료 대기하지 않음)
                raw_key = f"{self.raw_content_prefix}{content_hash}"
                await self.redis.delete(raw_key)
                
                logger.info(f"Content processed and raw data purged: {content_hash}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to process and preserve content: {e}")
            return False
    
    async def mark_processing(self, content_hash: str) -> bool:
        """처리 중 마킹 (중복 처리 방지)"""
        
        processing_key = f"{self.processing_prefix}{content_hash}"
        
        try:
            # SET if not exists with TTL
            result = await self.redis.set(
                processing_key,
                json.dumps({
                    "started_at": datetime.now().isoformat(),
                    "content_hash": content_hash
                }),
                ex=self.processing_ttl,
                nx=True  # Only set if key doesn't exist
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to mark processing: {e}")
            return False
    
    async def unmark_processing(self, content_hash: str) -> bool:
        """처리 완료 마킹 해제"""
        
        processing_key = f"{self.processing_prefix}{content_hash}"
        
        try:
            await self.redis.delete(processing_key)
            return True
            
        except Exception as e:
            logger.error(f"Failed to unmark processing: {e}")
            return False
    
    async def is_processing(self, content_hash: str) -> bool:
        """처리 중인지 확인"""
        
        processing_key = f"{self.processing_prefix}{content_hash}"
        
        try:
            return bool(await self.redis.exists(processing_key))
            
        except Exception as e:
            logger.error(f"Failed to check processing status: {e}")
            return False
    
    async def cleanup_expired_content(self) -> int:
        """만료된 원문 정리 (백그라운드 작업)"""
        
        try:
            # 만료된 원문 키 찾기
            pattern = f"{self.raw_content_prefix}*"
            expired_keys = []
            
            async for key in self.redis.scan_iter(match=pattern):
                ttl = await self.redis.ttl(key)
                if ttl <= 0:  # 만료된 키
                    expired_keys.append(key)
            
            if expired_keys:
                await self.redis.delete(*expired_keys)
                self.stats["total_expired"] += len(expired_keys)
                
                logger.info(f"Cleaned up {len(expired_keys)} expired raw content items")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired content: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        
        try:
            # Redis 메모리 정보
            info = await self.redis.info("memory")
            
            # 키 개수 계산
            raw_count = 0
            metadata_count = 0
            processing_count = 0
            
            async for key in self.redis.scan_iter(match=f"{self.raw_content_prefix}*"):
                raw_count += 1
            
            async for key in self.redis.scan_iter(match=f"{self.metadata_prefix}*"):
                metadata_count += 1
            
            async for key in self.redis.scan_iter(match=f"{self.processing_prefix}*"):
                processing_count += 1
            
            return {
                "memory_usage": info.get("used_memory_human", "0B"),
                "raw_content_items": raw_count,
                "metadata_items": metadata_count,
                "processing_items": processing_count,
                "raw_content_ttl": self.raw_content_ttl,
                "metadata_ttl": self.metadata_ttl,
                **self.stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return self.stats
    
    async def _load_stats(self):
        """통계 로드"""
        try:
            stats_key = f"{self.stats_prefix}main"
            cached_stats = await self.redis.get(stats_key)
            
            if cached_stats:
                self.stats.update(json.loads(cached_stats))
                
        except Exception as e:
            logger.error(f"Failed to load stats: {e}")
    
    async def _save_stats(self):
        """통계 저장"""
        try:
            stats_key = f"{self.stats_prefix}main"
            await self.redis.setex(
                stats_key,
                86400,  # 24시간
                json.dumps(self.stats)
            )
            
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")


# 백그라운드 정리 작업
class CacheCleanupService:
    """캐시 정리 서비스"""
    
    def __init__(self, cache_manager: ContentCacheManager):
        self.cache_manager = cache_manager
        self.cleanup_interval = 300  # 5분마다 정리
        self.running = False
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """정리 서비스 시작"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Cache cleanup service started")
    
    async def stop(self):
        """정리 서비스 중지"""
        self.running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cache cleanup service stopped")
    
    async def _cleanup_loop(self):
        """정리 루프"""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                if not self.running:
                    break
                
                # 만료된 컨텐츠 정리
                cleaned = await self.cache_manager.cleanup_expired_content()
                
                if cleaned > 0:
                    logger.info(f"Cleanup cycle completed: {cleaned} items removed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기


# 테스트 코드
if __name__ == "__main__":
    import time
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_cache_manager():
        """캐시 매니저 테스트"""
        
        async with ContentCacheManager() as cache_manager:
            print("=== 컨텐츠 캐시 매니저 테스트 ===")
            
            # 1. 원문 캐싱
            print("\n1. 원문 캐싱 테스트")
            content_hash = await cache_manager.cache_raw_content(
                content="Bitcoin surges to new highs as institutional adoption grows...",
                title="Bitcoin Reaches New All-Time High",
                url="https://example.com/bitcoin-news",
                source="crypto_news",
                published_at=datetime.now(),
                category="crypto",
                symbol="BTC"
            )
            print(f"캐시된 컨텐츠 해시: {content_hash}")
            
            # 2. 원문 조회
            print("\n2. 원문 조회 테스트")
            raw_data = await cache_manager.get_raw_content(content_hash)
            if raw_data:
                print(f"원문 조회 성공: {raw_data['title']}")
            else:
                print("원문 조회 실패")
            
            # 3. 처리 마킹
            print("\n3. 처리 마킹 테스트")
            marked = await cache_manager.mark_processing(content_hash)
            print(f"처리 마킹: {marked}")
            
            is_processing = await cache_manager.is_processing(content_hash)
            print(f"처리 중 확인: {is_processing}")
            
            # 4. 감정 분석 후 메타데이터 보관
            print("\n4. 메타데이터 보관 및 원문 폐기 테스트")
            processed = await cache_manager.process_and_preserve(
                content_hash=content_hash,
                sentiment_score=0.75,
                confidence=0.85,
                keywords=["bitcoin", "surge", "institutional", "adoption"],
                entities=["bitcoin", "cryptocurrency"]
            )
            print(f"처리 및 보관: {processed}")
            
            # 5. 처리 마킹 해제
            await cache_manager.unmark_processing(content_hash)
            
            # 6. 원문 재조회 (폐기되었으므로 None)
            print("\n5. 원문 폐기 확인")
            raw_data_after = await cache_manager.get_raw_content(content_hash)
            print(f"원문 폐기 후 조회: {raw_data_after is None}")
            
            # 7. 메타데이터 조회
            print("\n6. 메타데이터 조회 테스트")
            metadata = await cache_manager.get_metadata(content_hash)
            if metadata:
                print(f"메타데이터 조회 성공:")
                print(f"  제목: {metadata.title}")
                print(f"  감정 점수: {metadata.sentiment_score}")
                print(f"  신뢰도: {metadata.confidence}")
                print(f"  키워드: {metadata.keywords}")
            else:
                print("메타데이터 조회 실패")
            
            # 8. 통계 확인
            print("\n7. 캐시 통계")
            stats = await cache_manager.get_cache_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    # 테스트 실행
    asyncio.run(test_cache_manager())