#!/usr/bin/env python3
"""
FinBERT Batch Processor for AuroraQ Sentiment Service
15분 간격 배치 처리로 CPU 부담 완화 및 정확도 향상
"""

import asyncio
import time
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

# Transformers 라이브러리
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, Pipeline
)
import numpy as np

# 로컬 임포트
from ..utils.content_cache_manager import ContentCacheManager, ContentMetadata
from ..collectors.enhanced_news_collector import NewsItem
from ..utils.data_quality_validator import DataQualityValidator, ContentItem, ContentType

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """처리 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class BatchItem:
    """배치 처리 아이템"""
    content_hash: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    symbol: Optional[str] = None
    category: str = "general"
    priority: int = 1  # 1=high, 2=medium, 3=low
    status: ProcessingStatus = ProcessingStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

@dataclass
class BatchResult:
    """배치 처리 결과"""
    content_hash: str
    sentiment_score: float
    confidence: float
    label: str
    keywords: List[str]
    processing_time: float
    model_version: str
    batch_id: str
    processed_at: datetime

class FinBERTBatchProcessor:
    """FinBERT 배치 프로세서"""
    
    def __init__(self, 
                 model_name: str = "ProsusAI/finbert",
                 cache_manager: Optional[ContentCacheManager] = None,
                 max_batch_size: int = 16,
                 max_sequence_length: int = 512):
        """
        초기화
        
        Args:
            model_name: FinBERT 모델 이름
            cache_manager: 캐시 매니저
            max_batch_size: 최대 배치 크기
            max_sequence_length: 최대 시퀀스 길이
        """
        self.model_name = model_name
        self.cache_manager = cache_manager
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        
        # 모델 관련
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.pipeline: Optional[Pipeline] = None
        self.model_version: str = "finbert_v1.0"
        self.device = "cpu"  # VPS에서 CPU 사용
        self.model_lock = threading.Lock()
        
        # 배치 관리
        self.batch_queue: List[BatchItem] = []
        self.processing_queue: List[BatchItem] = []
        self.completed_batches: List[BatchResult] = []
        self.failed_items: List[BatchItem] = []
        
        # 스케줄링
        self.batch_interval = 900  # 15분 (900초)
        self.last_batch_time = 0
        self.is_running = False
        self.processor_task: Optional[asyncio.Task] = None
        
        # 성능 통계
        self.stats = {
            "total_processed": 0,
            "total_batches": 0,
            "total_errors": 0,
            "avg_processing_time": 0.0,
            "model_load_time": 0.0,
            "last_batch_time": None,
            "items_per_batch_avg": 0.0
        }
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.initialize_model()
        await self.start_processor()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop_processor()
        self.thread_pool.shutdown(wait=True)
    
    async def initialize_model(self) -> bool:
        """모델 초기화 (단일 인스턴스)"""
        
        if self.model is not None:
            logger.info("FinBERT model already loaded")
            return True
        
        logger.info(f"Loading FinBERT model: {self.model_name}")
        start_time = time.time()
        
        try:
            # CPU에서 실행하도록 설정
            torch.set_num_threads(2)  # CPU 스레드 수 제한
            
            def load_model():
                """스레드에서 모델 로드"""
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # CPU에서는 float32 사용
                    device_map=None  # CPU 사용
                )
                
                # CPU로 이동
                model = model.to(self.device)
                model.eval()  # 평가 모드
                
                # Pipeline 생성
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU 사용
                    truncation=True,
                    max_length=self.max_sequence_length
                )
                
                return tokenizer, model, sentiment_pipeline
            
            # 스레드 풀에서 모델 로드
            future = self.thread_pool.submit(load_model)
            self.tokenizer, self.model, self.pipeline = await asyncio.wrap_future(future)
            
            load_time = time.time() - start_time
            self.stats["model_load_time"] = load_time
            
            logger.info(f"FinBERT model loaded in {load_time:.2f}s on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            return False
    
    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """간단한 키워드 추출 (TF-IDF 대신 빈도 기반)"""
        
        # 금융 관련 중요 키워드
        important_keywords = {
            "bitcoin", "ethereum", "crypto", "cryptocurrency", "blockchain",
            "stock", "market", "trading", "investment", "price", "surge",
            "drop", "rally", "crash", "bullish", "bearish", "federal",
            "regulation", "policy", "economic", "inflation", "interest"
        }
        
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            # 특수문자 제거
            clean_word = ''.join(c for c in word if c.isalnum())
            
            if len(clean_word) >= 3:
                # 중요 키워드에 가중치 부여
                weight = 2 if clean_word in important_keywords else 1
                word_freq[clean_word] = word_freq.get(clean_word, 0) + weight
        
        # 빈도순 정렬 후 상위 키워드 반환
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def _process_single_text(self, text: str) -> Tuple[float, float, str]:
        """단일 텍스트 처리 (스레드 안전)"""
        
        with self.model_lock:
            try:
                if not self.pipeline:
                    raise ValueError("Model not initialized")
                
                # FinBERT 분석
                result = self.pipeline(text)[0]
                
                label = result["label"].lower()
                confidence = float(result["score"])
                
                # 점수 정규화 (-1.0 ~ 1.0)
                if label == "positive":
                    sentiment_score = confidence
                elif label == "negative":
                    sentiment_score = -confidence
                else:  # neutral
                    sentiment_score = 0.0
                
                return sentiment_score, confidence, label
                
            except Exception as e:
                logger.error(f"FinBERT processing failed: {e}")
                return 0.0, 0.0, "neutral"
    
    async def process_batch(self, batch_items: List[BatchItem]) -> List[BatchResult]:
        """배치 처리"""
        
        if not batch_items:
            return []
        
        batch_id = hashlib.md5(
            f"{datetime.now().isoformat()}{len(batch_items)}".encode()
        ).hexdigest()[:8]
        
        logger.info(f"Processing batch {batch_id} with {len(batch_items)} items")
        start_time = time.time()
        
        results = []
        
        # 배치 아이템들을 처리
        for item in batch_items:
            item_start = time.time()
            item.status = ProcessingStatus.PROCESSING
            
            try:
                # 텍스트 준비
                full_text = f"{item.title} {item.content}"
                if len(full_text) > self.max_sequence_length * 4:  # 대략적인 토큰 수 제한
                    full_text = full_text[:self.max_sequence_length * 4]
                
                # FinBERT 분석 (스레드 풀에서 실행)
                future = self.thread_pool.submit(self._process_single_text, full_text)
                sentiment_score, confidence, label = await asyncio.wrap_future(future)
                
                # 키워드 추출
                keywords = self._extract_keywords(full_text)
                
                processing_time = time.time() - item_start
                
                # 결과 생성
                result = BatchResult(
                    content_hash=item.content_hash,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    label=label,
                    keywords=keywords,
                    processing_time=processing_time,
                    model_version=self.model_version,
                    batch_id=batch_id,
                    processed_at=datetime.now()
                )
                
                results.append(result)
                item.status = ProcessingStatus.COMPLETED
                item.processing_time = processing_time
                
                # 캐시 매니저에 결과 저장
                if self.cache_manager:
                    await self.cache_manager.process_and_preserve(
                        content_hash=item.content_hash,
                        sentiment_score=sentiment_score,
                        confidence=confidence,
                        keywords=keywords
                    )
                
                logger.debug(f"Processed {item.content_hash}: "
                           f"{label} ({sentiment_score:.3f}, {confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to process item {item.content_hash}: {e}")
                item.status = ProcessingStatus.FAILED
                item.error_message = str(e)
                item.retry_count += 1
                
                # 재시도 가능한 아이템은 큐에 다시 추가
                if item.retry_count < 3:
                    self.batch_queue.append(item)
                else:
                    self.failed_items.append(item)
        
        batch_time = time.time() - start_time
        
        # 통계 업데이트
        self.stats["total_processed"] += len([r for r in results])
        self.stats["total_batches"] += 1
        self.stats["total_errors"] += len(batch_items) - len(results)
        
        if results:
            avg_time = sum(r.processing_time for r in results) / len(results)
            current_avg = self.stats["avg_processing_time"]
            total_processed = self.stats["total_processed"]
            
            # 이동 평균 업데이트
            self.stats["avg_processing_time"] = (
                (current_avg * (total_processed - len(results)) + avg_time * len(results))
                / total_processed
            )
        
        self.stats["last_batch_time"] = datetime.now().isoformat()
        self.stats["items_per_batch_avg"] = (
            self.stats["total_processed"] / self.stats["total_batches"]
        )
        
        logger.info(f"Batch {batch_id} completed: "
                   f"{len(results)} processed, {len(batch_items) - len(results)} failed "
                   f"in {batch_time:.2f}s")
        
        self.completed_batches.extend(results)
        return results
    
    async def add_to_queue(self, 
                          content_hash: str,
                          title: str,
                          content: str,
                          source: str,
                          url: str,
                          published_at: datetime,
                          symbol: Optional[str] = None,
                          category: str = "general",
                          priority: int = 1) -> bool:
        """큐에 아이템 추가"""
        
        # 중복 확인
        existing_hashes = {item.content_hash for item in self.batch_queue}
        if content_hash in existing_hashes:
            logger.debug(f"Item already in queue: {content_hash}")
            return False
        
        batch_item = BatchItem(
            content_hash=content_hash,
            title=title,
            content=content,
            source=source,
            url=url,
            published_at=published_at,
            symbol=symbol,
            category=category,
            priority=priority
        )
        
        self.batch_queue.append(batch_item)
        logger.debug(f"Added to batch queue: {content_hash} (queue size: {len(self.batch_queue)})")
        
        return True
    
    async def add_news_items(self, news_items: List[NewsItem]) -> int:
        """뉴스 아이템들을 배치 큐에 추가"""
        
        added_count = 0
        
        for news in news_items:
            # 우선순위 계산
            priority = 1 if news.relevance_score > 0.7 else 2
            
            success = await self.add_to_queue(
                content_hash=news.hash_id,
                title=news.title,
                content=news.content,
                source=news.source.value,
                url=news.url,
                published_at=news.published_at,
                symbol=news.symbol,
                category=news.category,
                priority=priority
            )
            
            if success:
                added_count += 1
        
        logger.info(f"Added {added_count}/{len(news_items)} news items to batch queue")
        return added_count
    
    def _prepare_batch(self) -> List[BatchItem]:
        """다음 배치 준비"""
        
        if not self.batch_queue:
            return []
        
        # 우선순위별 정렬 (priority 낮은 순서가 높은 우선순위)
        self.batch_queue.sort(key=lambda x: (x.priority, x.published_at), reverse=True)
        
        # 배치 크기만큼 아이템 추출
        batch_size = min(len(self.batch_queue), self.max_batch_size)
        batch_items = self.batch_queue[:batch_size]
        self.batch_queue = self.batch_queue[batch_size:]
        
        return batch_items
    
    async def run_batch_cycle(self) -> bool:
        """배치 사이클 실행"""
        
        try:
            # 배치 준비
            batch_items = self._prepare_batch()
            
            if not batch_items:
                logger.debug("No items in batch queue")
                return False
            
            # 배치 처리
            results = await self.process_batch(batch_items)
            
            if results:
                logger.info(f"Batch cycle completed: {len(results)} items processed")
                return True
            else:
                logger.warning("Batch cycle completed with no results")
                return False
                
        except Exception as e:
            logger.error(f"Batch cycle failed: {e}")
            return False
    
    async def start_processor(self):
        """배치 프로세서 시작"""
        
        if self.is_running:
            logger.warning("Batch processor already running")
            return
        
        if not self.model:
            if not await self.initialize_model():
                raise RuntimeError("Failed to initialize FinBERT model")
        
        self.is_running = True
        self.processor_task = asyncio.create_task(self._processor_loop())
        
        logger.info(f"FinBERT batch processor started (interval: {self.batch_interval}s)")
    
    async def stop_processor(self):
        """배치 프로세서 중지"""
        
        self.is_running = False
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("FinBERT batch processor stopped")
    
    async def _processor_loop(self):
        """프로세서 메인 루프"""
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 15분 간격 체크 또는 큐가 가득 찬 경우
                should_process = (
                    (current_time - self.last_batch_time) >= self.batch_interval or
                    len(self.batch_queue) >= self.max_batch_size
                )
                
                if should_process:
                    success = await self.run_batch_cycle()
                    if success:
                        self.last_batch_time = current_time
                
                # 다음 체크까지 대기 (10초 간격)
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processor loop error: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """프로세서 통계 반환"""
        
        return {
            "is_running": self.is_running,
            "model_loaded": self.model is not None,
            "queue_size": len(self.batch_queue),
            "processing_queue_size": len(self.processing_queue),
            "completed_batches": len(self.completed_batches),
            "failed_items": len(self.failed_items),
            "batch_interval_seconds": self.batch_interval,
            "max_batch_size": self.max_batch_size,
            "last_batch_time": self.last_batch_time,
            "next_batch_in_seconds": max(0, self.batch_interval - (time.time() - self.last_batch_time)),
            **self.stats
        }
    
    async def force_batch_run(self) -> bool:
        """강제 배치 실행 (테스트/디버깅용)"""
        
        logger.info("Forcing batch run...")
        return await self.run_batch_cycle()


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_finbert_processor():
        """FinBERT 배치 프로세서 테스트"""
        
        # 캐시 매니저 없이 테스트
        async with FinBERTBatchProcessor() as processor:
            print("=== FinBERT 배치 프로세서 테스트 ===")
            
            # 테스트 아이템들 추가
            test_items = [
                {
                    "content_hash": "test1",
                    "title": "Bitcoin Surges to New All-Time High",
                    "content": "Bitcoin reached a new record as institutional adoption grows",
                    "source": "reuters",
                    "url": "https://reuters.com/test1",
                    "published_at": datetime.now(),
                    "symbol": "BTC"
                },
                {
                    "content_hash": "test2", 
                    "title": "Market Crash Fears Grip Investors",
                    "content": "Economic uncertainty leads to massive selloff across markets",
                    "source": "bloomberg",
                    "url": "https://bloomberg.com/test2",
                    "published_at": datetime.now(),
                    "symbol": "MARKET"
                }
            ]
            
            print(f"\n1. 아이템 큐에 추가...")
            for item in test_items:
                await processor.add_to_queue(**item)
            
            print(f"큐 크기: {len(processor.batch_queue)}")
            
            # 강제 배치 실행
            print(f"\n2. 배치 처리 실행...")
            success = await processor.force_batch_run()
            print(f"배치 처리 성공: {success}")
            
            # 결과 확인
            print(f"\n3. 처리 결과:")
            for result in processor.completed_batches:
                print(f"  {result.content_hash}: {result.label} "
                      f"({result.sentiment_score:.3f}, {result.confidence:.3f})")
                print(f"    키워드: {result.keywords}")
                print(f"    처리시간: {result.processing_time:.3f}s")
            
            # 통계
            print(f"\n4. 프로세서 통계:")
            stats = processor.get_processor_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    # 테스트 실행
    asyncio.run(test_finbert_processor())